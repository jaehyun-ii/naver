"""구조화 사례 생성 — 수치 경계값·예외 mutation + modality-aware 질문 (결정적).

LLM이 라벨을 결정하지 않는다: mutation 계획(facts·의도 라벨)은 코드가 만들고,
적용성 축 라벨은 condition_evaluator가 계산해 계획과 일치함을 검증한다
(불일치 = 케이스 제외). modality(규범 성격)·question_intent(질문 목적)를 분리해
intent별 명시 질문을 렌더하고, 다축 라벨(applicability/normative/compliance/primary)
을 부여한다 — 금지 조건 충족을 NON_COMPLIANT로 자동 변환하지 않는다.

LLM의 역할은 두 곳뿐 — ① 예외 텍스트의 조건 구조화(EXCEPTION_STRUCT),
② 생성된 사례의 intent 명시 블라인드 재판정(VERIFY_CASE v2).

    python -m scripts.aireg_kr.build_cases --structure-exceptions   # LLM ①
    python -m scripts.aireg_kr.build_cases --build                  # 결정적
    python -m scripts.aireg_kr.build_cases --verify                 # LLM ②
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from . import prompts
from .answer_format import ANSWER_FORMAT_JUDGMENT
from .common import (OUT_DIR, VERIFY_LLM_BASE, VERIFY_LLM_MODEL, append_jsonl,
                     chat_json, gen_meta, load_done, load_jsonl, log, pmap)
from .condition_evaluator import (EVALUATOR_VERSION, OPERATOR_ALIASES,
                                  SUPPORTED_OPS, evaluate_rule,
                                  evaluation_to_dict, parse_number_with_unit)
from .labels import compare_primary_labels
from .modality import (DEFAULT_INTENT, INTENT_LABELS, QuestionIntent,
                       RuleModality, derive_axis_labels,
                       modality_for_requirement)
from .question_renderer import render_question

CASES = OUT_DIR / "cases.jsonl"
EXCEPTION_STRUCTS = OUT_DIR / "exception_structs.jsonl"
CASE_VERIFICATIONS = OUT_DIR / "case_verifications.jsonl"

# 경계값 변형: operator → [(variant, delta, 적용성 축 기대)]
BOUNDARY_PLAN = {
    "gte": [("below", -1, "NOT_APPLICABLE"), ("at", 0, "APPLICABLE"), ("above", +1, "APPLICABLE")],
    "gt": [("below", -1, "NOT_APPLICABLE"), ("at", 0, "NOT_APPLICABLE"), ("above", +1, "APPLICABLE")],
    "lte": [("below", -1, "APPLICABLE"), ("at", 0, "APPLICABLE"), ("above", +1, "NOT_APPLICABLE")],
    "lt": [("below", -1, "APPLICABLE"), ("at", 0, "NOT_APPLICABLE"), ("above", +1, "NOT_APPLICABLE")],
}

# intent → 검증 프롬프트용 주의문(§7.2)
INTENT_NOTES = {
    QuestionIntent.ACTION_PROHIBITED:
        "금지 조건이 충족되면 primary label은 PROHIBITED입니다. 금지 조건 충족을 "
        "NON_COMPLIANT와 동일시하지 마십시오.",
    QuestionIntent.OBLIGATION_TRIGGERED:
        "의무 발생 조건이 충족되면 OBLIGATION_TRIGGERED입니다. 실제 이행 여부(준수)는 "
        "이 질문의 대상이 아닙니다.",
    QuestionIntent.ACTION_PERMITTED:
        "허용 조건이 충족되면 PERMITTED입니다.",
    QuestionIntent.EXCEPTION_APPLIES:
        "예외 세부 조건이 전부 충족될 때만 EXCEPTION_APPLIES입니다.",
    QuestionIntent.RULE_APPLIES:
        "규정의 적용 대상 여부만 판정합니다. 준수 여부는 묻지 않습니다.",
    QuestionIntent.COMPLIANCE_JUDGMENT:
        "실제 행위·구성 facts를 근거로 준수 여부를 판정합니다. 행위 facts가 없으면 "
        "INSUFFICIENT_INFORMATION입니다.",
    QuestionIntent.REQUIREMENT_SATISFIED:
        "요건 충족 여부만 판정합니다.",
}


def allowed_labels_of(intent: QuestionIntent) -> str:
    pos, neg = INTENT_LABELS[intent]
    return f"{pos}|{neg}|INSUFFICIENT_INFORMATION"


# ── 카드 → 평가용 조건 정규화 ────────────────────────────────────────────
def _eval_cond(c: dict) -> dict | None:
    """카드 v2 조건 → evaluator 조건({field,operator,value,unit}). 미지원은 None."""
    op = OPERATOR_ALIASES.get(str(c.get("operator") or ""), str(c.get("operator") or ""))
    if op not in SUPPORTED_OPS:
        return None
    field = str(c.get("field") or c.get("subject") or "").strip()
    if not field:
        return None
    value = c.get("value")
    # LLM이 문자열 비교에 eq/ne를 쓰는 경우 — 수치가 아니면 equals 의미로 정규화
    if op in ("eq", "ne") and parse_number_with_unit(value)[0] is None:
        op = "equals" if op == "eq" else "not_equals"
    # 문장형(prose) 값의 문자열 비교는 의미 없는 케이스를 만든다 — 구조화 불가 처리
    if op in ("equals", "not_equals") and isinstance(value, str) \
            and (len(value) > 25 or len(value.split()) >= 4):
        return None
    return {"field": field, "operator": op, "value": value,
            "unit": c.get("unit") or None, "evidence": c.get("evidence", "")}


def satisfying_fact(cond: dict):
    """조건을 충족하는 사례 값 조립 — 다른 조건 고정용."""
    op, v = cond["operator"], cond["value"]
    num, _u = parse_number_with_unit(v)
    if op in ("gte", "lte", "eq") and num is not None:
        return num
    if op == "gt" and num is not None:
        return int(num + 1) if float(num).is_integer() else None  # 실수 step 불명확
    if op == "lt" and num is not None:
        return int(num - 1) if float(num).is_integer() else None
    if op == "equals":
        return v
    if op == "in" and isinstance(v, (list, tuple)) and v:
        return v[0]
    if op == "is_true":
        return True
    if op == "is_false":
        return False
    if op == "exists":
        return "제공됨"
    return None


def _violating_fact(cond: dict):
    """조건을 미충족하는 값."""
    op, v = cond["operator"], cond["value"]
    num, _u = parse_number_with_unit(v)
    if op == "gte" and num is not None:
        return num - 1
    if op == "gt" and num is not None:
        return num
    if op == "lte" and num is not None:
        return num + 1
    if op == "lt" and num is not None:
        return num
    if op == "eq" and num is not None:
        return num + 1
    if op == "equals":
        return f"not_{v}"
    if op == "in" and isinstance(v, (list, tuple)):
        return "other_value"
    if op == "is_true":
        return False
    if op == "is_false":
        return True
    return None


def _int_threshold(cond: dict) -> int | None:
    """정수 임계값만 경계값 생성 대상(실수는 step 불명확 → 제외)."""
    f, _unit = parse_number_with_unit(cond["value"])
    if f is None:
        return None
    return int(f) if float(f).is_integer() else None


# ── 렌더 ────────────────────────────────────────────────────────────────
def render_case_text(rule: dict, facts: dict, names: dict[str, str] | None = None) -> str:
    """결정적 사례문 렌더 — 라벨 힌트 없이 사실만 나열.

    facts는 {"applicability_facts": {...}, "behavior_facts": {...}} 2계층.
    """
    lines = [f"대상: {rule['section_path'].split('>')[-1].strip()} 관련 설비/선박."]

    def emit(layer: dict, header: str | None = None):
        items = []
        for field_name, fv in layer.items():
            if field_name == "applicability_asserted":
                lines.insert(1, "- 본 선박/설비는 이 요건의 적용 대상이다.")
                continue
            val, unit = (fv.get("value"), fv.get("unit")) if isinstance(fv, dict) else (fv, None)
            shown = ("예" if val is True else "아니오" if val is False else val)
            label = (names or {}).get(field_name, field_name)
            items.append(f"- {label}: {shown}{' ' + str(unit) if unit else ''}")
        if items and header:
            lines.append(header)
        lines.extend(items)

    emit(facts.get("applicability_facts") or {})
    behavior = facts.get("behavior_facts") or {}
    if behavior:
        lines.append("실제 행위·구성 (설치·운용 기록 확인 결과):")
        for field_name, fv in behavior.items():
            val = fv.get("value") if isinstance(fv, dict) else fv
            certainty = (fv.get("certainty") if isinstance(fv, dict) else None) or "CONFIRMED"
            label = (names or {}).get(field_name, field_name)
            if certainty == "CONFIRMED" and val is True:
                lines.append(f"- {label}: 현재 선박에서 해당 행위가 실제로 수행되고 있음이 "
                             "기록으로 확인되었다.")
            elif certainty == "CONFIRMED" and val is False:
                lines.append(f"- {label}: 현재 선박 구성에서 해당 행위는 수행되지 않으며, "
                             "설치·운용 기록을 확인한 결과 수행된 사실이 없음이 확인되었다.")
            else:  # 비확정(PLANNED 등)은 준수 판정 사례에 쓰지 않지만 방어적 렌더
                lines.append(f"- {label}: {val} (확실성: {certainty})")
    return "\n".join(lines)


# 확정 사실을 약화시키는 표현(§2.3 금지 목록) — 렌더러 회귀 가드
_WEAKENING = ("예정", "바람직", "검토 중", "계획에는", "여부는 아니오")


def fact_realization_checks(case_text: str, behavior_facts: dict,
                            names: dict[str, str] | None = None) -> list[dict]:
    """§2.4 — 사례문이 behavior facts를 확정적으로 실현했는지 결정적 검증."""
    checks = []
    for field_name, fv in (behavior_facts or {}).items():
        val = fv.get("value") if isinstance(fv, dict) else fv
        certainty = (fv.get("certainty") if isinstance(fv, dict) else None) or "CONFIRMED"
        label = (names or {}).get(field_name, field_name)
        # 라벨 표기 줄 찾기
        line = next((ln for ln in case_text.splitlines() if label in ln), "")
        if certainty == "CONFIRMED" and val is True:
            realized = "실제로 수행되고 있음" in line and "확인" in line
        elif certainty == "CONFIRMED" and val is False:
            realized = "수행되지 않으며" in line and "확인" in line
        else:
            realized = bool(line)
        weakened = any(w in line for w in _WEAKENING)
        contradicted = (val is False and "수행되고 있음" in line) or                        (val is True and "수행되지 않" in line)
        checks.append({"fact_path": f"behavior_facts.{field_name}",
                       "expected_value": val, "expected_certainty": certainty,
                       "realized": realized, "realized_as": line.strip(),
                       "contradicted": contradicted, "weakened": weakened})
    return checks


def build_prompt_text(rule: dict, case_text: str, question: str) -> str:
    return ("당신은 선급 규정 검토 전문가입니다. 아래 규정 조항과 사례 사실을 검토하여 "
            "질문에 답하십시오. 예외·면제는 세부 조건이 전부 충족될 때만 적용되며, "
            "결정적 정보가 없으면 '판단 불가'와 함께 필요한 정보를 명시하십시오.\n\n"
            f"[규정 조항] {rule['section_path']}\n{rule['article_text']}\n\n"
            f"[사례 사실]\n{case_text}\n\n[질문]\n{question}")


def _case_row(rule: dict, req_id: str, kind: str, variant: str,
              app_facts: dict, intended_applicability: str, mutation: dict, *,
              modality: RuleModality, intent: QuestionIntent,
              behavior_facts: dict | None = None,
              behavior_performed: bool | None = None,
              names: dict[str, str] | None = None) -> dict | None:
    """facts → 평가 → 다축 라벨 → intent 질문 렌더. 계획-평가 불일치는 제외."""
    spec = mutation.pop("_eval_spec")
    ev = evaluate_rule(spec, app_facts)
    if ev.derived_label != intended_applicability:
        log(f"  !! 계획-평가 라벨 불일치({kind}/{variant}): "
            f"의도 {intended_applicability} vs 평가 {ev.derived_label} — 제외")
        return None
    labels = derive_axis_labels(applicability_label=ev.derived_label,
                                modality=modality, intent=intent,
                                behavior_performed=behavior_performed)
    facts = {"applicability_facts": app_facts,
             "behavior_facts": behavior_facts or {}}
    case_text = render_case_text(rule, facts, names)
    fact_checks = fact_realization_checks(case_text, behavior_facts or {}, names)
    if any(not c["realized"] or c["contradicted"] or c["weakened"] for c in fact_checks):
        log(f"  !! fact realization 실패({kind}/{variant}) — 제외")
        return None
    question = render_question(modality=modality.value, intent=intent.value,
                               rule_card=rule, case_facts=facts)
    cid = f"{rule['id']}::{req_id}::{kind}::{variant}"
    return {
        "case_id": cid, "rule_id": rule["id"], "requirement_id": req_id,
        "case_kind": kind, "variant": variant,
        "rule_modality": modality.value, "question_intent": intent.value,
        "task_type": ("exception_judgment" if kind.startswith("exception")
                      else "compliance_judgment" if intent == QuestionIntent.COMPLIANCE_JUDGMENT
                      else "condition_verification"),
        "answer_format": ANSWER_FORMAT_JUDGMENT,
        "case_facts": facts, "case_text": case_text,
        "question": question,
        "prompt": build_prompt_text(rule, case_text, question),
        "labels": labels,
        "intended_label": labels["primary_label"],       # 질문(intent)에 대한 정답
        "program_label": labels["primary_label"],        # 평가기 유도 primary
        "program_applicability_label": ev.derived_label,  # 적용성 축 원값
        "program_evaluation": evaluation_to_dict(ev) | {
            "applicability_label": ev.derived_label,
            "normative_label": labels["normative_label"],
            "compliance_label": labels["compliance_label"],
            "primary_label": labels["primary_label"],
            "question_intent": intent.value, "rule_modality": modality.value},
        "fact_realization_checks": fact_checks,
        "mutation": mutation,
        "_gen": gen_meta("CASE_BUILDER", model="deterministic") | {"prompt": "CASE_BUILDER/v2"},
    }


# ── 경계값 생성 (§10 + modality) ────────────────────────────────────────
def boundary_cases(rule: dict, stats: Counter,
                   condition_gate: dict | None = None) -> list[dict]:
    out = []
    for req in (rule.get("card") or {}).get("requirements") or []:
        conds = [(c, _eval_cond(c)) for c in req.get("conditions") or []]
        if not conds:
            continue
        # 조건 자족성 게이트(§유보충돌 교정): ACCEPT 요건만 경계값·준수 케이스 생성.
        # 게이트 미수행(None)은 하위 호환으로 통과, 판정 존재 시 ACCEPT만.
        if condition_gate is not None:
            gid = f"{rule['id']}::{req.get('requirement_id', 'R?')}"
            g = condition_gate.get(gid)
            if g is None or g.get("decision") not in ("ACCEPT", "CORRECTED"):
                stats["boundary_skip_condition_gate"] += 1
                continue
            if g.get("decision") == "CORRECTED" and g.get("corrected_conditions"):
                # 게이트 교정 통과분 — 보강된 조건 집합으로 케이스 생성
                req = dict(req, conditions=g["corrected_conditions"])
                conds = [(c, _eval_cond(c)) for c in req["conditions"]]
                stats["boundary_used_corrected_conditions"] += 1
        if any(ec is None for _, ec in conds):
            stats["boundary_skip_unsupported_sibling"] += 1
            continue
        modality, _ev = modality_for_requirement(req, rule.get("article_text", ""))
        intent = DEFAULT_INTENT[modality]
        stats[f"modality_{modality.value}"] += 0  # 분포 키 노출
        evals = [ec for _, ec in conds]
        req_id = req.get("requirement_id", "R?")
        compliance_done = False
        for i, target in enumerate(evals):
            plan = BOUNDARY_PLAN.get(target["operator"])
            thr = _int_threshold(target)
            if not plan or thr is None:
                if plan and thr is None:
                    stats["boundary_skip_non_integer_step"] += 1
                continue
            for variant, delta, app_label in plan:
                app_facts = {}
                ok = True
                for j, other in enumerate(evals):
                    if j == i:
                        continue
                    val = satisfying_fact(other)
                    if val is None:
                        ok = False
                        break
                    app_facts[other["field"]] = {"value": val, "unit": other["unit"]}
                if not ok:
                    stats["boundary_skip_unfixable_sibling"] += 1
                    continue
                app_facts[target["field"]] = {"value": thr + delta, "unit": target["unit"]}
                mutation = {"type": "numeric_boundary", "field": target["field"],
                            "operator": target["operator"], "threshold": thr,
                            "variant": f"{variant}_boundary" if variant != "at" else "at_boundary",
                            "source_rule_id": rule["id"],
                            "_eval_spec": {"conditions": evals, "condition_logic": "AND"}}
                row = _case_row(rule, req_id, "numeric_boundary",
                                f"{target['field']}_{variant}", app_facts, app_label,
                                mutation, modality=modality, intent=intent)
                if row:
                    out.append(row)
                    stats["boundary_generated"] += 1
                # 금지·의무 규칙 + 조건 충족 변형 발생 시(요건당 1회) —
                # behavior facts 부여 compliance 판정 사례 파생
                if row and app_label == "APPLICABLE" and not compliance_done \
                        and modality in (RuleModality.PROHIBITION, RuleModality.MANDATORY):
                    out += _compliance_variants(rule, req_id, evals, app_facts,
                                                modality, stats)
                    compliance_done = True
    return out


def _compliance_variants(rule: dict, req_id: str, evals: list[dict], app_facts: dict,
                         modality: RuleModality, stats: Counter) -> list[dict]:
    """§3.6/§6.2 — behavior facts가 있을 때만 compliance 판정 사례 생성.

    금지: 수행→NON_COMPLIANT, 미수행→COMPLIANT, 없음→생성 안 함(질문 자체 금지).
    """
    field = ("prohibited_action_performed" if modality == RuleModality.PROHIBITION
             else "required_action_performed")
    ko = ("금지된 사용/행위 수행 여부" if modality == RuleModality.PROHIBITION
          else "의무 조치 이행 여부")
    out = []
    for performed, tag in ((True, "performed"), (False, "not_performed")):
        mutation = {"type": "compliance_behavior", "behavior_field": field,
                    "performed": performed, "source_rule_id": rule["id"],
                    "_eval_spec": {"conditions": evals, "condition_logic": "AND"}}
        # §2.2 — 값과 확실성 분리: 준수 판정 사례는 CONFIRMED facts만 사용
        behavior = {field: {"value": performed, "unit": None,
                            "certainty": "CONFIRMED",
                            "time_scope": "CURRENT_CONFIGURATION",
                            "evidence_type": "EXPLICIT_STATEMENT"}}
        row = _case_row(rule, req_id, "compliance_behavior", tag,
                        dict(app_facts), "APPLICABLE", mutation,
                        modality=modality, intent=QuestionIntent.COMPLIANCE_JUDGMENT,
                        behavior_facts=behavior,
                        behavior_performed=performed, names={field: ko})
        if row:
            out.append(row)
            stats["compliance_behavior_generated"] += 1
    return out


# ── 예외 구조화 (§11 — LLM 단계) ────────────────────────────────────────
def structure_exceptions(limit: int = 0, workers: int = 1) -> None:
    rules = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    done = load_done(EXCEPTION_STRUCTS)
    todo = []
    for rule in rules:
        for req in (rule.get("card") or {}).get("requirements") or []:
            ex_text = (req.get("exception") or "").strip()
            if not ex_text or len(ex_text) < 10:
                continue
            sid = f"{rule['id']}::{req.get('requirement_id', 'R?')}"
            if sid not in done:
                todo.append((sid, rule, req, ex_text))
    if limit:
        todo = todo[:limit]

    def process(job: tuple) -> None:
        sid, rule, req, ex_text = job
        try:
            obj = chat_json(prompts.EXCEPTION_STRUCT.format(
                article_text=rule["article_text"],
                requirement_text=req.get("requirement_text", ""),
                exception_text=ex_text), max_tokens=2200)
            supported = [c for c in obj.get("exception_conditions") or []
                         if _eval_cond(c)]
            usable = bool(obj.get("structurable")) and len(supported) >= 2
            append_jsonl(EXCEPTION_STRUCTS, {
                "id": sid, "rule_id": rule["id"],
                "requirement_id": req.get("requirement_id", "R?"),
                "exception_text": ex_text, "struct": obj,
                "n_supported": len(supported), "usable": usable,
                "_gen": gen_meta("EXCEPTION_STRUCT")})
            log(f"구조화 {sid}: 지원 조건 {len(supported)}개 usable={usable}")
        except Exception as e:  # noqa: BLE001
            log(f"구조화 실패 {sid}: {e}")

    pmap(process, todo, workers=workers)


# ── 예외 케이스 생성 (§11 — 결정적 mutation) ────────────────────────────
def exception_cases(rule: dict, struct_row: dict, stats: Counter) -> list[dict]:
    struct = struct_row["struct"]
    ex_conds = [_eval_cond(c) for c in struct.get("exception_conditions") or []]
    ex_conds = [c for c in ex_conds if c]
    if len(ex_conds) < 2:
        return []
    ex_logic = str(struct.get("exception_logic") or "AND").upper()
    if ex_logic != "AND":
        stats["exception_skip_non_and_logic"] += 1
        return []
    base = [{"field": "applicability_asserted", "operator": "is_true", "value": None,
             "evidence": "사례가 본 요건의 적용 대상임을 명시"}]
    spec = {"conditions": base, "condition_logic": "AND",
            "exception_conditions": ex_conds, "exception_logic": "AND"}
    names = {str(c.get("field")): str(c.get("field_ko"))
             for c in struct.get("exception_conditions") or []
             if c.get("field") and c.get("field_ko")}

    def facts_for(met: list[bool], omit: list[int]) -> dict:
        facts = {"applicability_asserted": True}
        for k, (cond, m) in enumerate(zip(ex_conds, met)):
            if k in omit:
                continue
            val = satisfying_fact(cond) if m else _violating_fact(cond)
            if val is None:
                return {}
            facts[cond["field"]] = {"value": val, "unit": cond["unit"]}
        return facts

    plans = [
        ("exception_all_met", [True] * len(ex_conds), [], "EXCEPTION_APPLIES"),
        ("exception_none_met", [False] * len(ex_conds), [], "APPLICABLE"),
        ("exception_partially_met", [True] + [False] * (len(ex_conds) - 1), [], "APPLICABLE"),
        ("exception_missing_information", [True] * len(ex_conds), [len(ex_conds) - 1],
         "INSUFFICIENT_INFORMATION"),
    ]
    out = []
    for kind, met, omit, intended_app in plans:
        app_facts = facts_for(met, omit)
        if not app_facts:
            stats["exception_skip_unfabricatable"] += 1
            continue
        mutation = {"type": kind, "exception_fields": [c["field"] for c in ex_conds],
                    "met_plan": met,
                    "omitted_fields": [ex_conds[k]["field"] for k in omit],
                    "source_rule_id": rule["id"], "_eval_spec": spec}
        row = _case_row(rule, struct_row["requirement_id"], kind, "v1", app_facts,
                        intended_app, mutation,
                        modality=RuleModality.EXEMPTION,
                        intent=QuestionIntent.EXCEPTION_APPLIES, names=names)
        if row:
            out.append(row)
            stats[f"{kind}_generated"] += 1
    return out


# ── 블라인드 검증 — 2-pass(기본) / single-pass(하위 호환 --single-pass) ──
def _required_fields(case: dict) -> list[str]:
    pe = case.get("program_evaluation") or {}
    fields = [c["field"] for c in (pe.get("condition_results") or [])
              + (pe.get("exception_results") or [])
              if c.get("field") and c["field"] != "applicability_asserted"]
    fields += list((case.get("case_facts") or {}).get("behavior_facts") or {})
    return sorted(set(fields))


def _condition_table(case: dict) -> str:
    """§5C — verifier가 숫자를 자연어로 재계산하지 않도록 구조화 조건표 제공.
    프로그램의 최종 라벨은 포함하지 않는다."""
    pe = case.get("program_evaluation") or {}
    rows = []
    for c in (pe.get("condition_results") or []) + (pe.get("exception_results") or []):
        if c.get("field") == "applicability_asserted":
            continue
        rows.append({"field": c.get("field"), "operator": c.get("operator"),
                     "threshold": c.get("expected_value"), "unit": c.get("unit"),
                     "case_value": c.get("actual_value")})
    return json.dumps(rows, ensure_ascii=False)


def _norm_intent(raw) -> str | None:
    """모델이 '"action_prohibited 예상"'처럼 안내문을 복사하는 경우 정규화."""
    if not raw:
        return None
    raw = str(raw)
    for qi in QuestionIntent:
        if qi.value in raw:
            return qi.value
    return raw.strip()


def _judge_once(case: dict, rule: dict, intent: QuestionIntent, facts_out: dict) -> dict:
    return chat_json(prompts.VERIFY_JUDGE.format(
        rule_modality=case.get("rule_modality", "unknown"),
        question_intent=intent.value,
        intent_note=INTENT_NOTES.get(intent, ""),
        allowed_labels=allowed_labels_of(intent),
        article_text=rule["article_text"],
        condition_table=_condition_table(case),
        extracted_facts=json.dumps(facts_out, ensure_ascii=False),
        question=case.get("question", "")), max_tokens=1800,
        base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)


def verify_case_two_pass(case: dict, rule: dict) -> dict:
    """Pass1 사실 추출 → 프로그램 facts 대조 → Pass2 판정. §3.3/§3.4 규칙 적용."""
    intent = QuestionIntent(case.get("question_intent", "rule_applies"))
    required = _required_fields(case)
    p1 = chat_json(prompts.VERIFY_FACTS.format(
        required_fields=json.dumps(required, ensure_ascii=False),
        case_text=case["case_text"]), max_tokens=1600,
        base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)
    # 프로그램 facts와 Pass1 대조 — 값 반전은 실현/추출 충돌
    flat = {}
    cf = case.get("case_facts") or {}
    for layer in ("applicability_facts", "behavior_facts"):
        for k, v in (cf.get(layer) or {}).items():
            flat[k] = v.get("value") if isinstance(v, dict) else v
    conflicts = []
    for k, v in (p1.get("extracted_facts") or {}).items():
        if k in flat and isinstance(flat[k], bool):
            pv = str(v).strip().lower() in ("true", "예", "1", "yes")
            if pv != flat[k]:
                conflicts.append(k)
    p2 = _judge_once(case, rule, intent, p1)
    label = p2.get("primary_label")
    missing = p2.get("missing_fields") or []
    invalid = label == "INSUFFICIENT_INFORMATION" and not missing
    if invalid:  # §3.3 — missing_fields 없는 유보는 무효 → 1회 재판정
        p2 = _judge_once(case, rule, intent, p1)
        label = p2.get("primary_label")
        missing = p2.get("missing_fields") or []
        invalid = label == "INSUFFICIENT_INFORMATION" and not missing
    # §3.4 — 필수 fact가 전부 사례에 존재(구조상 항상 명시)하는데 유보 → 충돌
    program_present = all(f in flat for f in required)
    omitted = [f for f in required if f not in flat]  # missing_information 변형은 의도적 누락
    abstention_conflict = (label == "INSUFFICIENT_INFORMATION" and program_present
                           and not omitted and case.get("program_label")
                           != "INSUFFICIENT_INFORMATION")
    if abstention_conflict:
        # 명시 사실 유보는 1회 재판정 — 재시도에서 확정 라벨이 나오면 채택
        p2b = _judge_once(case, rule, intent, p1)
        label_b = p2b.get("primary_label")
        if label_b and label_b != "INSUFFICIENT_INFORMATION":
            p2, label, missing = p2b, label_b, p2b.get("missing_fields") or []
            abstention_conflict = False
    return {"pass1": p1, "pass2": p2, "verifier_label": label,
            "interpreted_intent": _norm_intent(p2.get("interpreted_intent")),
            "fact_conflicts": conflicts,
            "insufficient_invalid": invalid,
            "abstention_conflict": abstention_conflict}


def verify_cases(limit: int = 0, only_ids: set[str] | None = None,
                 single_pass: bool = False, workers: int = 1) -> None:
    rules = {r["id"]: r for r in load_jsonl(OUT_DIR / "rule_cards.jsonl")}
    done = load_done(CASE_VERIFICATIONS, key="case_id")
    todo = [c for c in load_jsonl(CASES)
            if c["case_id"] not in done and c["rule_id"] in rules
            and (only_ids is None or c["case_id"] in only_ids)]
    if limit:
        todo = todo[:limit]

    def process(case: dict) -> None:
        rule = rules[case["rule_id"]]
        intent = QuestionIntent(case.get("question_intent", "rule_applies"))
        try:
            if single_pass:  # 하위 호환 경로
                v = chat_json(prompts.VERIFY_CASE.format(
                    rule_modality=case.get("rule_modality", "unknown"),
                    question_intent=intent.value,
                    intent_note=INTENT_NOTES.get(intent, ""),
                    allowed_labels=allowed_labels_of(intent),
                    article_text=rule["article_text"], case_text=case["case_text"],
                    question=case.get("question", "")), max_tokens=1800,
                    base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)
                out = {"verifier_label": v.get("primary_label"),
                       "interpreted_intent": _norm_intent(v.get("interpreted_intent")),
                       "pass2": v, "fact_conflicts": [],
                       "insufficient_invalid": False, "abstention_conflict": False}
            else:
                out = verify_case_two_pass(case, rule)
            lc = compare_primary_labels(
                question_intent=intent.value,
                generation_primary_label=case["intended_label"],
                verifier_primary_label=out["verifier_label"],
                program_primary_label=case["program_label"],
                verifier_interpreted_intent=out.get("interpreted_intent"))
            if out["abstention_conflict"]:
                lc = lc | {"agreement": "VERIFIER_ABSTENTION_CONFLICT",
                           "action": "REVERIFY_OR_EXPERT"}
            if out["fact_conflicts"]:
                lc = lc | {"agreement": "FACT_REALIZATION_OR_EXTRACTION_CONFLICT",
                           "action": "INSPECT_CASE"}
            append_jsonl(CASE_VERIFICATIONS, {
                "case_id": case["case_id"], "rule_id": case["rule_id"],
                "question_intent": intent.value,
                "rule_modality": case.get("rule_modality"),
                "verifier_label": out["verifier_label"],
                "interpreted_intent": out.get("interpreted_intent"),
                "two_pass": not single_pass,
                "fact_conflicts": out["fact_conflicts"],
                "insufficient_invalid": out["insufficient_invalid"],
                "abstention_conflict": out["abstention_conflict"],
                "verification": out.get("pass2"), "pass1": out.get("pass1"),
                "label_consistency": lc,
                "_gen": gen_meta("VERIFY_JUDGE" if not single_pass else "VERIFY_CASE",
                                 model=VERIFY_LLM_MODEL)})
            log(f"검증 {case['case_id']}: {case['intended_label']}/"
                f"{out['verifier_label']}/{case['program_label']} → {lc['agreement']}")
        except Exception as e:  # noqa: BLE001
            log(f"검증 실패 {case['case_id']}: {e}")

    pmap(process, todo, workers=workers)


def build(stats: Counter) -> None:
    rules = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    # §1 게이트: EXCEPTION_STRUCT_VERIFY ACCEPT 구조만 케이스 생성에 사용
    gate = {v["id"]: v for v in load_jsonl(OUT_DIR / "exception_struct_verifications.jsonl")}
    cgate_rows = load_jsonl(OUT_DIR / "condition_gate_verifications.jsonl")
    condition_gate = {v["id"]: v for v in cgate_rows} if cgate_rows else None
    structs = {}
    for srow in load_jsonl(EXCEPTION_STRUCTS):
        if not srow.get("usable"):
            continue
        g = gate.get(srow["id"])
        if not g or g.get("decision") != "ACCEPT":
            stats["exception_struct_gate_blocked"] += 1
            continue
        structs[srow["rule_id"]] = {**srow, "struct": g["final_struct"]}
    done = load_done(CASES, key="case_id")
    n = 0
    for rule in rules:
        rows = boundary_cases(rule, stats, condition_gate)
        if rule["id"] in structs:
            rows += exception_cases(rule, structs[rule["id"]], stats)
        for row in rows:
            if row["case_id"] in done:
                continue
            append_jsonl(CASES, row)
            done.add(row["case_id"])
            n += 1
    log(f"케이스 {n}건 생성 → {CASES} | 통계 {dict(stats)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="경계값·예외·compliance 구조화 케이스 생성")
    ap.add_argument("--structure-exceptions", action="store_true", help="LLM 예외 구조화")
    ap.add_argument("--build", action="store_true", help="결정적 케이스 생성")
    ap.add_argument("--verify", action="store_true", help="LLM 블라인드 검증(2-pass 기본)")
    ap.add_argument("--single-pass", action="store_true", help="구형 단일 패스 검증")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AIREG_WORKERS", "1")))
    args = ap.parse_args(argv)
    if args.structure_exceptions:
        structure_exceptions(args.limit, workers=args.workers)
    if args.build:
        build(Counter())
    if args.verify:
        verify_cases(args.limit, single_pass=args.single_pass,
                     workers=args.workers)
    if not (args.structure_exceptions or args.build or args.verify):
        log("단계 플래그(--structure-exceptions/--build/--verify)를 지정하십시오")


if __name__ == "__main__":
    main()
