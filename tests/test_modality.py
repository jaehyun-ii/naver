"""modality 분류·intent 라벨 축·질문 렌더러·다축 라벨 검증."""
import pytest

from scripts.aireg_kr.modality import (QuestionIntent, RuleModality,
                                       classify_modality, derive_axis_labels,
                                       modality_for_requirement,
                                       select_primary_label)
from scripts.aireg_kr.question_renderer import render_question


# ── modality 분류 ────────────────────────────────────────────────────────
def test_korean_modality_patterns():
    cases = {
        "선박에는 1조의 주조타장치를 비치하여야 한다.": RuleModality.MANDATORY,
        "설계온도가 350 °C를 초과하는 관계통에 사용하여서는 아니 된다.": RuleModality.PROHIBITION,
        "동력장치 2조를 갖춘 경우 보조조타장치를 생략할 수 있다.": RuleModality.EXEMPTION,
        "우리 선급이 인정하는 재료를 사용할 수 있다.": RuleModality.PERMISSION,
        "다음 각 호의 선박에 적용한다.": RuleModality.APPLICABILITY,
        "\"주조타장치\"라 함은 타를 작동시키는 장치를 말한다.": RuleModality.DEFINITION,
        "완성 후 검사를 받아야 한다.": RuleModality.PROCEDURE,
        "가능한 한 낮은 위치에 설치한다.": RuleModality.RECOMMENDATION,
    }
    for text, want in cases.items():
        got, evidence = classify_modality(text)
        assert got == want, (text, got)
        assert evidence  # 근거 매치 보존


def test_mixed_normative_is_unknown():
    # 의무+금지 중첩 문장 — 강제 분류 금지(§11 중첩 → 검수 대상)
    text = "A를 설치하여야 하며, B는 사용하여서는 아니 된다."
    got, _ = classify_modality(text)
    assert got == RuleModality.UNKNOWN
    assert classify_modality("이 장치는 성능이 우수하다.")[0] == RuleModality.UNKNOWN
    assert classify_modality(None)[0] == RuleModality.UNKNOWN


def test_card_field_conflict_is_unknown():
    req = {"modality": "의무", "requirement_text": "사용하여서는 아니 된다."}
    got, reason = modality_for_requirement(req)
    assert got == RuleModality.UNKNOWN and "카드" in reason


# ── 질문 렌더러 ─────────────────────────────────────────────────────────
def test_render_question_per_intent():
    card = {"card": {"applicability": {"system_or_equipment": ["주조타장치"]}}}
    q_p = render_question(modality="prohibition", intent="action_prohibited",
                          rule_card=card, case_facts={})
    assert "금지됩니까" in q_p and "적용됩니까" not in q_p
    q_m = render_question(modality="mandatory", intent="obligation_triggered",
                          rule_card=card, case_facts={})
    assert "이행해야 합니까" in q_m and "의무" in q_m
    q_a = render_question(modality="applicability", intent="rule_applies",
                          rule_card=card, case_facts={})
    assert "적용 대상입니까" in q_a
    q_e = render_question(modality="exemption", intent="exception_applies",
                          rule_card=card, case_facts={})
    assert "예외" in q_e
    q_perm = render_question(modality="permission", intent="action_permitted",
                             rule_card=card, case_facts={})
    assert "허용됩니까" in q_perm


def test_compliance_question_requires_behavior_facts():
    card = {"card": {}}
    with pytest.raises(ValueError, match="behavior_facts"):
        render_question(modality="prohibition", intent="compliance_judgment",
                        rule_card=card, case_facts={"applicability_facts": {"x": 1}})
    q = render_question(modality="prohibition", intent="compliance_judgment",
                        rule_card=card,
                        case_facts={"behavior_facts": {"used": True}})
    assert "준수" in q


# ── 다축 라벨 (§4·§5) ───────────────────────────────────────────────────
def test_prohibition_condition_met_is_applicable_and_prohibited_not_noncompliant():
    labels = derive_axis_labels(applicability_label="APPLICABLE",
                                modality="prohibition", intent="action_prohibited")
    assert labels == {"applicability_label": "APPLICABLE",
                      "normative_label": "PROHIBITED",
                      "compliance_label": "NOT_ASSESSED",   # 자동 NON_COMPLIANT 금지
                      "primary_label": "PROHIBITED"}


def test_prohibition_with_behavior_facts():
    done = derive_axis_labels(applicability_label="APPLICABLE", modality="prohibition",
                              intent="compliance_judgment", behavior_performed=True)
    assert done["compliance_label"] == "NON_COMPLIANT" and done["primary_label"] == "NON_COMPLIANT"
    not_done = derive_axis_labels(applicability_label="APPLICABLE", modality="prohibition",
                                  intent="compliance_judgment", behavior_performed=False)
    assert not_done["compliance_label"] == "COMPLIANT"


def test_mandatory_axis():
    labels = derive_axis_labels(applicability_label="APPLICABLE",
                                modality="mandatory", intent="obligation_triggered")
    assert labels["normative_label"] == "OBLIGATION_TRIGGERED"
    assert labels["compliance_label"] == "NOT_ASSESSED"
    performed = derive_axis_labels(applicability_label="APPLICABLE", modality="mandatory",
                                   intent="compliance_judgment", behavior_performed=True)
    assert performed["compliance_label"] == "COMPLIANT"


def test_insufficient_propagates():
    labels = derive_axis_labels(applicability_label="INSUFFICIENT_INFORMATION",
                                modality="prohibition", intent="action_prohibited")
    assert labels["primary_label"] == "INSUFFICIENT_INFORMATION"


def test_select_primary_by_intent():
    kw = dict(applicability_label="APPLICABLE", normative_label="PROHIBITED",
              compliance_label="NOT_ASSESSED")
    assert select_primary_label(question_intent="rule_applies", **kw) == "APPLICABLE"
    assert select_primary_label(question_intent="action_prohibited", **kw) == "PROHIBITED"
    assert select_primary_label(question_intent="compliance_judgment", **kw) == "NOT_ASSESSED"


def test_intent_and_modality_are_separate_fields():
    # 같은 금지 규정에 두 가지 질문 — 조건 평가는 같아도 primary가 축별로 다름
    rule_applies = derive_axis_labels(applicability_label="APPLICABLE",
                                      modality="prohibition", intent="rule_applies")
    prohibited = derive_axis_labels(applicability_label="APPLICABLE",
                                    modality="prohibition", intent="action_prohibited")
    assert rule_applies["primary_label"] == "APPLICABLE"
    assert prohibited["primary_label"] == "PROHIBITED"
