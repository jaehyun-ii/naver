"""예외 구조 2차 품질 게이트 (§1) — 원문 대조 검증 + 정적 검사 + CORRECT 1회.

EXCEPTION_STRUCT 산출물을 즉시 쓰지 않고, (a) 프로그램 정적 검사(숫자·단위·
evidence 부분문자열·중복·빈 조건), (b) LLM 원문 대조(EXCEPTION_STRUCT_VERIFY),
(c) CORRECT 시 수정 구조 재검증(최대 1회)을 통과한 구조만 케이스 생성에 쓴다.

    python -m scripts.aireg_kr.exception_gate [--limit N]
    → data_aireg/exception_struct_verifications.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re

from . import prompts
from .common import (OUT_DIR, VERIFY_LLM_BASE, VERIFY_LLM_MODEL, append_jsonl,
                     chat_json, gen_meta, load_done, load_jsonl, log, pmap)
from .condition_evaluator import parse_number_with_unit

STRUCT_VERIFICATIONS = OUT_DIR / "exception_struct_verifications.jsonl"
CONDITION_GATES = OUT_DIR / "condition_gate_verifications.jsonl"

# 상대 기준·열거 분기 정적 검출(§조건 게이트) — LLM 게이트 전 선행 차단
RE_RELATIVE = re.compile(r"의\s*[0-9.]+\s*배|[0-9.]+\s*배(?:의|로)|퍼센트|%의")
RE_ENUM_BRANCH = re.compile(r"종류에\s*따라|각각\s*다음|다음\s*표에\s*따|등급별|재질별")


def static_condition_checks(conditions: list[dict], requirement_text: str,
                            article_text: str) -> list[str]:
    """조건 자족성 정적 검사 — 상대 기준·열거 분기 힌트를 코드로 선행 차단."""
    failures = []
    for c in conditions:
        val = str(c.get("value") or "")
        if RE_RELATIVE.search(val):
            failures.append(f"상대 기준 값({val[:30]}) — 절대 경계값 생성 불가")
    blob = f"{requirement_text} {article_text[:800]}"
    if RE_ENUM_BRANCH.search(blob):
        subjects = " ".join(str(c.get("field") or c.get("subject") or "") for c in conditions)
        if not re.search(r"종류|재질|등급|type|grade|material", subjects, re.I):
            failures.append("원문이 종류/재질별 상이 기준인데 종류 조건 부재")
    return failures


def _llm_gate_conditions(rule: dict, req: dict, conds: list[dict]) -> dict:
    return chat_json(prompts.CONDITION_GATE.format(
        article_text=rule["article_text"],
        requirement_text=req.get("requirement_text", ""),
        conditions=json.dumps(conds, ensure_ascii=False)),
        max_tokens=1400, base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)


def correct_conditions(rule: dict, req: dict, conds: list[dict],
                       missing: list[str]) -> list[dict] | None:
    """REJECT 조건 집합을 원문 기반으로 보강(교정) — 재게이트 통과 시에만 채택.

    교정도 검증기(gemma 등)로 수행하되, 보강 결과는 정적+LLM 게이트를 다시
    통과해야 한다(예외 게이트의 CORRECT 재검증 원칙과 동형, 1회 시도).
    """
    v = chat_json(prompts.CONDITION_CORRECT.format(
        article_text=rule["article_text"],
        requirement_text=req.get("requirement_text", ""),
        conditions=json.dumps(conds, ensure_ascii=False),
        missing_variables=json.dumps(missing, ensure_ascii=False)),
        max_tokens=1800, base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)
    if not v.get("correctable") or not v.get("conditions"):
        return None
    fixed = v["conditions"]
    # 재게이트: 정적 → LLM (통과해야 채택)
    if static_condition_checks(fixed, req.get("requirement_text", ""),
                               rule.get("article_text", "")):
        return None
    g = _llm_gate_conditions(rule, req, fixed)
    if str(g.get("decision") or "").upper() != "ACCEPT" or g.get("relative_threshold"):
        return None
    return fixed


def gate_conditions(limit: int = 0, redo: bool = False, correct: bool = True,
                    workers: int = 1) -> None:
    """요건별 조건 자족성 게이트 — 경계값·준수 케이스 생성 자격 판정.

    REJECT(변수 누락) 시 correct=True면 원문 기반 보강을 1회 시도하고,
    재게이트 통과분은 decision=CORRECTED + corrected_conditions로 기록한다.
    상대 기준·정성 조건은 교정 대상이 아니다(정당 제외)."""
    rules = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    done = set() if redo else load_done(CONDITION_GATES)
    from .modality import RuleModality, modality_for_requirement
    todo = []
    for rule in rules:
        for req in (rule.get("card") or {}).get("requirements") or []:
            conds = req.get("conditions") or []
            if not conds:
                continue
            # 정의·절차형 요건은 판정 케이스 대상이 아님 — 게이트 스킵(정보형 트랙 소관)
            mod, _ = modality_for_requirement(req, rule.get("article_text", ""))
            if mod in (RuleModality.DEFINITION, RuleModality.PROCEDURE):
                continue
            gid = f"{rule['id']}::{req.get('requirement_id', 'R?')}"
            if gid not in done:
                todo.append((gid, rule, req, conds))
    if limit:
        todo = todo[:limit]

    def process(job: tuple) -> None:
        gid, rule, req, conds = job
        static_fail = static_condition_checks(conds, req.get("requirement_text", ""),
                                              rule.get("article_text", ""))
        if static_fail:
            # 상대 기준은 교정 불가(정당 제외). 열거 분기(종류 조건 부재)는
            # 교정 프롬프트가 정확히 다루는 케이스 — 교정 루프로 라우팅.
            corrected = None
            if correct and not any("상대 기준" in f for f in static_fail):
                try:
                    corrected = correct_conditions(
                        rule, req, conds,
                        [f for f in static_fail if "종류" in f] or static_fail)
                except Exception as e:  # noqa: BLE001 — 일시 LLM 실패는 REJECT로 강등
                    log(f"조건게이트 교정 실패 {gid}: {e}")
            append_jsonl(CONDITION_GATES, {
                "id": gid, "rule_id": rule["id"],
                "requirement_id": req.get("requirement_id", "R?"),
                "decision": "CORRECTED" if corrected else "REJECT",
                "corrected_conditions": corrected,
                "static_failures": static_fail,
                "_gen": gen_meta("CONDITION_GATE",
                                 model=VERIFY_LLM_MODEL if corrected else "deterministic")})
            log(f"조건게이트 {gid}: {'CORRECTED' if corrected else 'REJECT'}(정적) "
                f"{static_fail[:1]}")
            return
        try:
            v = _llm_gate_conditions(rule, req, conds)
            decision = str(v.get("decision") or "").upper()
            if v.get("relative_threshold") or (v.get("enum_dependent")
                                               and v.get("missing_variables")):
                decision = "REJECT"
            corrected = None
            if decision != "ACCEPT" and correct and not v.get("relative_threshold") \
                    and v.get("missing_variables"):
                corrected = correct_conditions(rule, req, conds,
                                               v.get("missing_variables") or [])
                if corrected:
                    decision = "CORRECTED"
            append_jsonl(CONDITION_GATES, {
                "id": gid, "rule_id": rule["id"],
                "requirement_id": req.get("requirement_id", "R?"),
                "decision": decision or "REJECT",
                "corrected_conditions": corrected,
                "missing_variables": v.get("missing_variables") or [],
                "relative_threshold": bool(v.get("relative_threshold")),
                "enum_dependent": bool(v.get("enum_dependent")),
                "reason": v.get("reason"),
                "_gen": gen_meta("CONDITION_GATE", model=VERIFY_LLM_MODEL)})
            log(f"조건게이트 {gid}: {decision} {v.get('missing_variables') or ''}")
        except Exception as e:  # noqa: BLE001
            log(f"조건게이트 실패 {gid}: {e}")

    pmap(process, todo, workers=workers)

MAX_CORRECTION_ATTEMPTS = 1


def static_struct_checks(struct: dict, source_text: str) -> list[str]:
    """원문 기반 정적 검사(§1.5) — 실패 항목 목록 반환(빈 목록 = 통과)."""
    failures: list[str] = []
    conds = struct.get("exception_conditions") or []
    if not conds:
        return ["빈 예외 조건"]
    seen = set()
    src_norm = re.sub(r"\s+", "", source_text)
    for i, c in enumerate(conds):
        key = (c.get("field"), str(c.get("operator")), str(c.get("value")))
        if key in seen:
            failures.append(f"중복 조건: {c.get('field')}")
        seen.add(key)
        if not str(c.get("field") or "").strip():
            failures.append(f"조건 {i}: field 없음")
        num, _unit = parse_number_with_unit(c.get("value"))
        if num is not None and str(int(num) if float(num).is_integer() else num) \
                .replace(".", "") not in src_norm.replace(",", "").replace(".", ""):
            failures.append(f"조건 {i}: 수치 {num}가 원문에 없음")
        unit = str(c.get("unit") or "").strip()
        if unit and unit.lower() not in source_text.lower() \
                and unit not in ("조", "개", "기"):  # 단위 명사는 표기 변형 허용
            failures.append(f"조건 {i}: 단위 {unit!r}가 원문에 없음")
        ev = re.sub(r"\s+", "", str(c.get("evidence") or ""))
        if ev and len(ev) > 5 and ev not in src_norm:
            failures.append(f"조건 {i}: evidence가 원문 부분문자열이 아님")
    return failures


def verify_struct(rule: dict, struct_row: dict) -> dict:
    """LLM 원문 대조 + CORRECT 재검증 루프. 최종 판정 행 반환."""
    req_id = struct_row["requirement_id"]
    req = next((r for r in (rule.get("card") or {}).get("requirements") or []
                if r.get("requirement_id") == req_id), {})
    candidate = struct_row["struct"]
    attempts = 0
    history = []
    # evidence는 조항 원문 또는 카드 예외 텍스트(요약)에서 인용될 수 있다 — 합집합
    # 대비로 정적 검사. 원문 충실성의 의미적 최종 판단은 LLM 게이트가 수행한다.
    source = rule["article_text"] + "\n" + str(struct_row.get("exception_text") or "")
    while True:
        static_fail = static_struct_checks(candidate, source)
        if static_fail:
            return {"decision": "REJECT", "final_struct": None, "attempts": attempts,
                    "static_failures": static_fail, "history": history,
                    "review_reason": "정적 검사 실패"}
        v = chat_json(prompts.EXCEPTION_STRUCT_VERIFY.format(
            source_text=rule["article_text"],
            requirement_text=req.get("requirement_text", ""),
            exception_text=struct_row.get("exception_text", ""),
            candidate_structure=json.dumps(candidate, ensure_ascii=False)),
            max_tokens=2400)
        history.append({k: v.get(k) for k in ("decision", "missing_conditions",
                                              "unsupported_conditions", "review_reason")})
        decision = str(v.get("decision") or "").upper()
        if decision == "ACCEPT":
            return {"decision": "ACCEPT", "final_struct": candidate, "attempts": attempts,
                    "static_failures": [], "history": history,
                    "evidence_mapping": v.get("evidence_mapping") or []}
        corrected = v.get("corrected_exception_structure")
        if decision == "CORRECT" and corrected and attempts < MAX_CORRECTION_ATTEMPTS:
            candidate = corrected
            attempts += 1
            continue  # 수정 구조도 동일 검증(정적+LLM)을 다시 통과해야 함
        return {"decision": "EXPERT_REVIEW_REQUIRED" if decision == "CORRECT" else "REJECT",
                "final_struct": None, "attempts": attempts, "static_failures": [],
                "history": history, "review_reason": v.get("review_reason")}


def main() -> None:
    ap = argparse.ArgumentParser(description="예외 구조·조건 자족성 게이트")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="기존 판정 무시하고 재검증")
    ap.add_argument("--conditions", action="store_true",
                    help="요건 조건 자족성 게이트 실행(경계값·준수 케이스용)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AIREG_WORKERS", "1")))
    args = ap.parse_args()
    if args.conditions:
        gate_conditions(args.limit, args.redo, workers=args.workers)
        return
    rules = {r["id"]: r for r in load_jsonl(OUT_DIR / "rule_cards.jsonl")}
    done = set() if args.redo else load_done(STRUCT_VERIFICATIONS)
    todo = [s for s in load_jsonl(OUT_DIR / "exception_structs.jsonl")
            if s.get("usable") and s["id"] not in done and s["rule_id"] in rules]
    if args.limit:
        todo = todo[:args.limit]

    def process(srow: dict) -> None:
        try:
            out = verify_struct(rules[srow["rule_id"]], srow)
            append_jsonl(STRUCT_VERIFICATIONS, {
                "id": srow["id"], "rule_id": srow["rule_id"],
                "requirement_id": srow["requirement_id"], **out,
                "_gen": gen_meta("EXCEPTION_STRUCT_VERIFY")})
            log(f"게이트 {srow['id']}: {out['decision']} (attempts {out['attempts']})")
        except Exception as e:  # noqa: BLE001
            log(f"게이트 실패 {srow['id']}: {e}")

    pmap(process, todo, workers=args.workers)


if __name__ == "__main__":
    main()
