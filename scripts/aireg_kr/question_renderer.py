"""modality-aware 질문 렌더러 — 질문 목적이 라벨 의미와 1:1이 되도록 고정.

"이 규정이 적용됩니까?"류 중의적 질문(금지 조항에서 프로그램/검증기가 다른 축으로
해석)을 제거한다. intent별 템플릿은 여기서만 관리하고, LLM이 질문 목적을 바꾸지
못하게 생성물은 템플릿 기반으로만 만든다.
"""
from __future__ import annotations

from .modality import QuestionIntent, RuleModality

# intent → (질문 템플릿, 답 선택지 안내)
_TEMPLATES: dict[QuestionIntent, str] = {
    QuestionIntent.RULE_APPLIES:
        "이 규정의 적용 조건을 고려할 때, 위 사례는 이 규정의 적용 대상입니까?",
    QuestionIntent.OBLIGATION_TRIGGERED:
        "위 사례의 선박/설비는 이 규정에 따라 {target}을(를) 이행해야 합니까? (의무 발생 여부)",
    QuestionIntent.ACTION_PROHIBITED:
        "해당 조건에서 {target}의 사용(또는 해당 행위)은 이 규정에 의해 금지됩니까?",
    QuestionIntent.ACTION_PERMITTED:
        "해당 조건에서 {target}의 사용(또는 해당 행위)은 이 규정에 의해 허용됩니까?",
    QuestionIntent.EXCEPTION_APPLIES:
        "위 사례에 이 규정의 예외 또는 면제 조건이 적용됩니까?",
    QuestionIntent.REQUIREMENT_SATISFIED:
        "위 사례는 이 규정의 {target} 요건을 충족합니까?",
    QuestionIntent.COMPLIANCE_JUDGMENT:
        "위 사례의 실제 행위·구성을 고려할 때, 이 규정에 대한 준수 여부를 판정하십시오.",
}

# intent별 답변 선택지 안내(프롬프트에 명시해 라벨 축을 고정)
_ANSWER_GUIDE: dict[QuestionIntent, str] = {
    QuestionIntent.RULE_APPLIES: "적용 / 비적용 / 판단 불가",
    QuestionIntent.OBLIGATION_TRIGGERED: "의무 발생 / 의무 미발생 / 판단 불가",
    QuestionIntent.ACTION_PROHIBITED: "금지됨 / 금지되지 않음 / 판단 불가",
    QuestionIntent.ACTION_PERMITTED: "허용됨 / 허용되지 않음 / 판단 불가",
    QuestionIntent.EXCEPTION_APPLIES: "예외 적용 / 예외 미적용 / 판단 불가",
    QuestionIntent.REQUIREMENT_SATISFIED: "요건 충족 / 요건 미충족 / 판단 불가",
    QuestionIntent.COMPLIANCE_JUDGMENT: "적합 / 부적합 / 판단 불가",
}


def _target_of(rule_card: dict, target_subject: str | None) -> str:
    if target_subject:
        return target_subject
    card = rule_card.get("card") or rule_card
    eq = (card.get("applicability") or {}).get("system_or_equipment") or []
    if eq:
        return str(eq[0])
    return "해당 설비"


def render_question(*, modality: str, intent: str, rule_card: dict,
                    case_facts: dict, target_subject: str | None = None) -> str:
    """intent 템플릿 렌더 + 답 선택지 안내. 중의적 '적용됩니까' 질문을 대체.

    compliance 질문은 behavior facts가 있어야 한다 — 없으면 ValueError
    (호출부가 사전에 걸러야 하며, 임의 생성 방지용 최후 가드).
    """
    q_intent = QuestionIntent(intent)
    RuleModality(modality)  # 유효성만 검증
    if q_intent == QuestionIntent.COMPLIANCE_JUDGMENT and \
            not (case_facts.get("behavior_facts") or {}):
        raise ValueError("compliance 질문은 behavior_facts 없이 생성할 수 없습니다")
    template = _TEMPLATES[q_intent]
    question = template.format(target=_target_of(rule_card, target_subject))
    return f"{question}\n(답 선택지: {_ANSWER_GUIDE[q_intent]})"


def answer_guide(intent: str) -> str:
    return _ANSWER_GUIDE[QuestionIntent(intent)]
