"""GRPO 도메인 보상 — 프로그램 평가 결과 연계 (run_grpo_peft reward kwargs 사용).

모든 함수는 TRL 계약(callable(prompts, completions, **kw) -> list[float])을 따르며,
추가 컬럼은 GRPO 데이터 행에서 자동 전달된다(list, completions와 정렬).
program 평가 불가(evaluable=False) 행에는 그룹 내 상수(0.0)를 반환해
advantage에 영향을 주지 않는다 — UNKNOWN을 정답처럼 강제하지 않는다.
"""
from __future__ import annotations

from .answer_format import parse_decision_to_label, parse_regulation_answer
from .condition_evaluator import normalize_scalar
from .labels import canonicalize_label
from .modality import INTENT_LABELS, QuestionIntent

# 응답 '판단' 섹션 → 정규 라벨 폴백(구형 관용 표현 — LABEL_DISPLAY 우선 후 사용)
_LEGACY_PATTERNS = [
    ("판단 불가", "INSUFFICIENT_INFORMATION"),
    ("예외가 적용", "EXCEPTION_APPLIES"),
    ("적용되지 않", "NOT_APPLICABLE"),
]


def _completion_text(c) -> str:
    if isinstance(c, str):
        return c
    if c and isinstance(c, list) and isinstance(c[-1], dict):
        return c[-1].get("content", "")
    return ""


def extract_decision_label(text: str) -> str | None:
    """판단 섹션 → 표준 라벨. LABEL_DISPLAY(다축 표현) 우선, 구형 관용 폴백."""
    decision = parse_regulation_answer(text)["sections"].get("decision", "")
    if not decision:
        return None
    label = parse_decision_to_label(decision)
    if label:
        return label
    for pat, mapped in _LEGACY_PATTERNS:
        if pat in decision:
            return mapped
    return None


def _match_label(pred: str | None, gold: str | None) -> bool:
    """표준 라벨 일치 — 다축 라벨은 정확 일치, 구형 4라벨은 canonical 동치 허용."""
    if pred is None or gold is None:
        return False
    if pred == gold:
        return True
    return canonicalize_label(pred) == canonicalize_label(gold) != "UNKNOWN" \
        and gold in ("APPLICABLE", "NOT_APPLICABLE", "EXCEPTION_APPLIES",
                     "INSUFFICIENT_INFORMATION", "compliant", "non_compliant", "uncertain")


def program_label_reward(prompts, completions, program_label=None,
                         primary_label=None, evaluable=None, **kwargs) -> list[float]:
    """응답 판단 라벨 == 프로그램 평가 primary 라벨 → 1.0.

    primary_label이 있으면 그것을 정답 축으로 사용(§12), 없으면 구형
    program_label(하위 호환). evaluable=False 또는 UNKNOWN 행은 0.0(비활성 —
    그룹 내 상수라 GRPO advantage 무영향).
    """
    out = []
    for i, c in enumerate(completions):
        gold = (primary_label[i] if primary_label and primary_label[i]
                else (program_label[i] if program_label else None))
        ev = evaluable[i] if evaluable is not None else bool(gold)
        # UNKNOWN/NOT_ASSESSED를 정답처럼 강제하지 않음 — 비활성(그룹 내 상수)
        if not ev or not gold or gold in ("UNKNOWN", "NOT_ASSESSED"):
            out.append(0.0)
            continue
        out.append(1.0 if _match_label(extract_decision_label(_completion_text(c)), gold)
                   else 0.0)
    return out


_ALL_PRIMARY = {lab for pair in INTENT_LABELS.values() for lab in pair} | {
    "INSUFFICIENT_INFORMATION"}


def intent_alignment_reward(prompts, completions, question_intent=None,
                            rule_modality=None, primary_label=None,
                            **kwargs) -> list[float]:
    """의미 축 검증 보상 — 응답 라벨이 질문 intent의 라벨 축에 속하는지(§12.1).

    - intent 축 라벨(예: action_prohibited에 금지됨/금지되지 않음): 1.0
    - 판단 불가: 1.0(어느 축에서든 유효한 답)
    - 다른 축 라벨(금지 질문에 '적용'만): 0.25 (부분 — 축 혼동)
    - 라벨 추출 불가: 0.0
    intent 메타가 없는 행은 0.0(그룹 내 상수 — 무영향).
    """
    out = []
    for i, c in enumerate(completions):
        intent = question_intent[i] if question_intent else None
        if not intent:
            out.append(0.0)
            continue
        try:
            pos, neg = INTENT_LABELS[QuestionIntent(intent)]
        except (KeyError, ValueError):
            out.append(0.0)
            continue
        pred = extract_decision_label(_completion_text(c))
        if pred is None:
            out.append(0.0)
        elif pred in (pos, neg, "INSUFFICIENT_INFORMATION"):
            out.append(1.0)
        else:
            out.append(0.25)  # 표준 라벨이긴 하나 다른 의미 축
    return out


def condition_coverage_reward(prompts, completions, condition_results=None,
                              exception_results=None, **kwargs) -> list[float]:
    """조건 언급 보상(0~1) — field/값의 텍스트 포함 여부만 제한적으로 확인.

    의미 평가가 아니다. 정답 라벨 보상을 대체하지 않으며 낮은 가중치로만 사용.
    기본·예외 조건(condition_results + exception_results)을 합쳐 보되,
    케이스 빌더 내부 구성 필드(applicability_asserted)는 제외한다.
    조건이 없는 행은 0.0(그룹 내 상수).
    """
    out = []
    for i, c in enumerate(completions):
        conds = list((condition_results[i] if condition_results else None) or []) + \
            list((exception_results[i] if exception_results else None) or [])
        conds = [x for x in conds if x.get("field") != "applicability_asserted"]
        if not conds:
            out.append(0.0)
            continue
        text = _completion_text(c)
        hit = 0
        for cond in conds:
            tokens = [str(cond.get("field", "")),
                      str(normalize_scalar(cond.get("expected_value")) or "")]
            if any(t and t.lower().replace("_", " ") in text.lower().replace("_", " ")
                   for t in tokens):
                hit += 1
        out.append(round(hit / len(conds), 4))
    return out
