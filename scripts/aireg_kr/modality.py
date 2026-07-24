"""규정 modality·question intent 표준화 — 질문 목적과 규범 성격의 명시적 분리.

3자 라벨 불일치의 주원인은 금지 조항에 "이 규정이 적용됩니까?"를 물으면서
프로그램(조건 충족=적용)과 검증기(행위 가능 여부)가 다른 축으로 답한 것이다.
여기서 modality(규범 성격)와 intent(질문 목적)를 별도 필드로 고정하고,
한국어 규범 표현의 정규화 패턴을 한 곳에서 관리한다.

모호하면 UNKNOWN — 강제 분류 금지.
"""
from __future__ import annotations

import re
from enum import Enum


class RuleModality(str, Enum):
    APPLICABILITY = "applicability"
    MANDATORY = "mandatory"
    PROHIBITION = "prohibition"
    PERMISSION = "permission"
    RECOMMENDATION = "recommendation"
    EXEMPTION = "exemption"
    DEFINITION = "definition"
    PROCEDURE = "procedure"
    UNKNOWN = "unknown"


class QuestionIntent(str, Enum):
    RULE_APPLIES = "rule_applies"
    OBLIGATION_TRIGGERED = "obligation_triggered"
    ACTION_PROHIBITED = "action_prohibited"
    ACTION_PERMITTED = "action_permitted"
    EXCEPTION_APPLIES = "exception_applies"
    REQUIREMENT_SATISFIED = "requirement_satisfied"
    COMPLIANCE_JUDGMENT = "compliance_judgment"
    INFORMATION_EXTRACTION = "information_extraction"


# 한국어 규범 표현 → modality. 우선순위 순서(금지가 의무 패턴보다 먼저 —
# "사용하지 아니하여야 한다"가 mandatory의 "하여야 한다"에 잡히지 않게).
_PATTERNS: list[tuple[RuleModality, re.Pattern]] = [
    (RuleModality.PROHIBITION, re.compile(
        r"하여서는\s*아니\s*된다|사용하지\s*아니하여야|금지한다|허용되지\s*않는다"
        r"|사용하여서는\s*아니")),
    (RuleModality.EXEMPTION, re.compile(
        r"그러하지\s*아니하다|적용하지\s*아니한다|면제할\s*수\s*있다|생략할\s*수\s*있다")),
    (RuleModality.RECOMMENDATION, re.compile(
        r"권고한다|바람직하다|가능한\s*한")),
    (RuleModality.PERMISSION, re.compile(
        r"(?:할|사용할|허용할|간주할|인정할)\s*수\s*있다")),
    (RuleModality.MANDATORY, re.compile(
        r"하여야\s*(?:한다|하며|하고)|비치하여야|설치하여야|이어야\s*한다|따라야\s*한다")),
    (RuleModality.APPLICABILITY, re.compile(
        r"에\s*적용한다|적용\s*대상은|다음\s*선박에\s*적용|적용범위")),
    (RuleModality.DEFINITION, re.compile(
        r"라\s*함은|을\s*말한다|를\s*말한다|다음과\s*같이\s*정의")),
    (RuleModality.PROCEDURE, re.compile(
        r"검사를\s*받아야|승인을\s*받아야|제출하여야|신청하여야")),
]

# 카드 v2 requirement.modality(한글) → RuleModality 매핑
_CARD_MODALITY = {"의무": RuleModality.MANDATORY, "금지": RuleModality.PROHIBITION,
                  "허용": RuleModality.PERMISSION, "권고": RuleModality.RECOMMENDATION}


def classify_modality(text: str | None) -> tuple[RuleModality, str | None]:
    """규범 문장 → (modality, 근거 매치 문자열). 복수 축이 겹치면 UNKNOWN.

    문자열 패턴만으로 확정 — 모호(서로 다른 modality 패턴이 함께 매치)하면
    UNKNOWN으로 남기고 강제 분류하지 않는다. 단, 금지+의무 공존은 규정문에서
    통상 금지 단서가 우세하므로 금지 우선이 아니라 UNKNOWN 처리한다(§11 중첩).
    """
    if not text:
        return RuleModality.UNKNOWN, None
    hits: list[tuple[RuleModality, str]] = []
    remaining = text
    for modality, pat in _PATTERNS:
        m = pat.search(remaining)
        if m:
            hits.append((modality, m.group(0)))
            # 우선순위 패턴이 소비한 구간은 마스킹 — "생략할 수 있다"(면제)가
            # "할 수 있다"(허용)로 이중 매치되지 않게 한다
            remaining = remaining[:m.start()] + " " * (m.end() - m.start()) \
                + remaining[m.end():]
    if not hits:
        return RuleModality.UNKNOWN, None
    kinds = {h[0] for h in hits}
    # 규범 축(의무/금지/허용/권고)이 2개 이상 겹치면 모호 — UNKNOWN
    normative = kinds & {RuleModality.MANDATORY, RuleModality.PROHIBITION,
                         RuleModality.PERMISSION, RuleModality.RECOMMENDATION}
    if len(normative) > 1:
        return RuleModality.UNKNOWN, "; ".join(h[1] for h in hits)
    # 규범 축이 있으면 그것이 우선(적용/정의/절차 문구와 공존 가능)
    if normative:
        mod = next(iter(normative))
        return mod, next(h[1] for h in hits if h[0] == mod)
    return hits[0][0], hits[0][1]


def modality_for_requirement(req: dict, article_text: str = "") -> tuple[RuleModality, str | None]:
    """요건 단위 modality — 카드 v2 modality 필드 우선, 없으면 요건 문장 분류.

    카드 값과 evidence 패턴이 모순되면 UNKNOWN(§11 evidence-modality 모순).
    """
    card_mod = _CARD_MODALITY.get(str(req.get("modality") or "").strip())
    text_mod, evidence = classify_modality(req.get("requirement_text") or article_text)
    if card_mod and text_mod is not RuleModality.UNKNOWN and card_mod != text_mod:
        return RuleModality.UNKNOWN, f"카드({card_mod.value})≠본문({text_mod.value}): {evidence}"
    return (card_mod or text_mod), evidence


# modality → 기본 question intent (사례 생성 기본값)
DEFAULT_INTENT = {
    RuleModality.APPLICABILITY: QuestionIntent.RULE_APPLIES,
    RuleModality.MANDATORY: QuestionIntent.OBLIGATION_TRIGGERED,
    RuleModality.PROHIBITION: QuestionIntent.ACTION_PROHIBITED,
    RuleModality.PERMISSION: QuestionIntent.ACTION_PERMITTED,
    RuleModality.EXEMPTION: QuestionIntent.EXCEPTION_APPLIES,
    RuleModality.RECOMMENDATION: QuestionIntent.REQUIREMENT_SATISFIED,
    RuleModality.DEFINITION: QuestionIntent.INFORMATION_EXTRACTION,
    RuleModality.PROCEDURE: QuestionIntent.REQUIREMENT_SATISFIED,
    RuleModality.UNKNOWN: QuestionIntent.RULE_APPLIES,
}

# intent별 primary 라벨 집합 (긍정, 부정)
INTENT_LABELS = {
    QuestionIntent.RULE_APPLIES: ("APPLICABLE", "NOT_APPLICABLE"),
    QuestionIntent.OBLIGATION_TRIGGERED: ("OBLIGATION_TRIGGERED", "OBLIGATION_NOT_TRIGGERED"),
    QuestionIntent.ACTION_PROHIBITED: ("PROHIBITED", "NOT_PROHIBITED"),
    QuestionIntent.ACTION_PERMITTED: ("PERMITTED", "NOT_PERMITTED"),
    QuestionIntent.EXCEPTION_APPLIES: ("EXCEPTION_APPLIES", "EXCEPTION_NOT_APPLIES"),
    QuestionIntent.REQUIREMENT_SATISFIED: ("REQUIREMENT_SATISFIED", "REQUIREMENT_NOT_SATISFIED"),
    QuestionIntent.COMPLIANCE_JUDGMENT: ("COMPLIANT", "NON_COMPLIANT"),
}


def derive_axis_labels(*, applicability_label: str, modality: RuleModality | str,
                       intent: QuestionIntent | str,
                       behavior_performed: bool | None = None) -> dict:
    """조건 평가 결과(적용성) + modality/intent (+행위 facts) → 다축 라벨.

    핵심 원칙(§5): 조건 충족을 준수/위반으로 자동 변환하지 않는다.
    - normative: 적용성에서 기계적으로 유도 가능(금지 조건 충족→PROHIBITED 등)
    - compliance: behavior_performed가 주어졌을 때만 판정, 아니면 NOT_ASSESSED
    """
    modality = RuleModality(modality)
    intent = QuestionIntent(intent)
    app = applicability_label

    # normative 축 — 적용성 3치의 modality별 사상
    if modality == RuleModality.EXEMPTION:
        # 평가기 라벨 의미: EXCEPTION_APPLIES=예외 성립, APPLICABLE=예외 불성립(원칙 적용)
        normative = {"EXCEPTION_APPLIES": "EXCEPTION_APPLIES",
                     "APPLICABLE": "EXCEPTION_NOT_APPLIES",
                     "NOT_APPLICABLE": "EXCEPTION_NOT_APPLIES",
                     "INSUFFICIENT_INFORMATION": "INSUFFICIENT_INFORMATION",
                     }.get(app, "UNKNOWN")
    else:
        pos, neg = INTENT_LABELS.get(
            {RuleModality.MANDATORY: QuestionIntent.OBLIGATION_TRIGGERED,
             RuleModality.PROHIBITION: QuestionIntent.ACTION_PROHIBITED,
             RuleModality.PERMISSION: QuestionIntent.ACTION_PERMITTED,
             }.get(modality, QuestionIntent.RULE_APPLIES),
            ("APPLICABLE", "NOT_APPLICABLE"))
        if app == "APPLICABLE":
            normative = pos
        elif app == "NOT_APPLICABLE":
            normative = neg
        elif app == "EXCEPTION_APPLIES":
            normative = "EXCEPTION_APPLIES"  # 비-exemption 규칙에 예외 성립
        elif app == "INSUFFICIENT_INFORMATION":
            normative = "INSUFFICIENT_INFORMATION"
        else:
            normative = "UNKNOWN"

    # compliance 축 — 행위 facts 없이는 판정하지 않는다
    if modality == RuleModality.PROHIBITION and normative == "PROHIBITED":
        if behavior_performed is True:
            compliance = "NON_COMPLIANT"
        elif behavior_performed is False:
            compliance = "COMPLIANT"
        else:
            compliance = "NOT_ASSESSED"
    elif modality == RuleModality.MANDATORY and normative == "OBLIGATION_TRIGGERED":
        if behavior_performed is True:
            compliance = "COMPLIANT"
        elif behavior_performed is False:
            compliance = "NON_COMPLIANT"
        else:
            compliance = "NOT_ASSESSED"
    else:
        compliance = "NOT_ASSESSED"

    labels = {"applicability_label": app, "normative_label": normative,
              "compliance_label": compliance}
    labels["primary_label"] = select_primary_label(
        question_intent=intent.value, **labels)
    return labels


def select_primary_label(*, question_intent: str, applicability_label: str,
                         normative_label: str, compliance_label: str) -> str:
    """question_intent에 직접 답하는 축의 라벨을 primary로 선택."""
    intent = QuestionIntent(question_intent)
    if intent == QuestionIntent.RULE_APPLIES:
        return applicability_label
    if intent == QuestionIntent.COMPLIANCE_JUDGMENT:
        return compliance_label
    if intent == QuestionIntent.INFORMATION_EXTRACTION:
        return "NOT_ASSESSED"
    return normative_label
