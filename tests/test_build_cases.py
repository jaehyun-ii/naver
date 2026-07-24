"""build_cases v2 — modality-aware 질문·다축 라벨·mutation 결정성 (LLM 불필요)."""
from collections import Counter

from scripts.aireg_kr.build_cases import (boundary_cases, exception_cases,
                                          satisfying_fact)

RULE = {
    "id": "KR_TEST_RULE_A1", "section_path": "규칙 > 제5편 > 601. 시험",
    "article_text": "총톤수 10,000톤 이상인 유조선은 A를 설치하여야 한다. 다만 …",
    "card": {"requirements": [{
        "requirement_id": "R1",
        "requirement_text": "총톤수 10,000톤 이상인 유조선은 A를 설치하여야 한다.",
        "modality": "의무",
        "conditions": [
            {"subject": "ship_type", "operator": "equals", "value": "oil_tanker"},
            {"subject": "총톤수", "operator": "이상", "value": "10,000", "unit": "GT"},
        ],
        "condition_logic": "ALL",
    }]},
}

PROHIBIT_RULE = {
    "id": "KR_TEST_RULE_P1", "section_path": "규칙 > 제5편 > 103. 재료",
    "article_text": "설계온도가 350 °C를 초과하는 관계통에 사용하여서는 아니 된다.",
    "card": {"requirements": [{
        "requirement_id": "R1",
        "requirement_text": "설계온도가 350 °C를 초과하는 관계통에 사용하여서는 아니 된다.",
        "modality": "금지",
        "conditions": [{"subject": "설계온도", "operator": "초과",
                        "value": "350", "unit": "°C"}],
    }]},
}


def test_mandatory_boundary_axis_labels_and_question():
    stats = Counter()
    rows = [r for r in boundary_cases(RULE, stats) if r["case_kind"] == "numeric_boundary"]
    assert len(rows) == 3
    by = {r["variant"].rsplit("_", 1)[1]: r for r in rows}
    at = by["at"]
    assert at["rule_modality"] == "mandatory"
    assert at["question_intent"] == "obligation_triggered"
    assert "이행해야 합니까" in at["question"]           # 중의적 '적용됩니까' 아님
    assert at["labels"]["applicability_label"] == "APPLICABLE"
    assert at["labels"]["normative_label"] == "OBLIGATION_TRIGGERED"
    assert at["labels"]["compliance_label"] == "NOT_ASSESSED"
    assert at["program_label"] == at["intended_label"] == "OBLIGATION_TRIGGERED"
    assert by["below"]["program_label"] == "OBLIGATION_NOT_TRIGGERED"
    # facts 2계층 + provenance
    assert at["case_facts"]["applicability_facts"]["총톤수"]["value"] == 10000
    assert at["case_facts"]["behavior_facts"] == {}
    assert at["mutation"]["threshold"] == 10000 and "_eval_spec" not in at["mutation"]


def test_prohibition_boundary_not_noncompliant():
    stats = Counter()
    rows = boundary_cases(PROHIBIT_RULE, stats)
    at = next(r for r in rows if r["variant"].endswith("_at") is False
              and r["case_kind"] == "numeric_boundary" and "above" in r["variant"])
    # 초과(gt): above가 조건 충족 → 금지됨. NON_COMPLIANT 아님(§5)
    assert at["rule_modality"] == "prohibition"
    assert "금지됩니까" in at["question"]
    assert at["labels"]["normative_label"] == "PROHIBITED"
    assert at["labels"]["compliance_label"] == "NOT_ASSESSED"
    assert at["program_label"] == "PROHIBITED"


def test_prohibition_compliance_behavior_variants():
    stats = Counter()
    rows = [r for r in boundary_cases(PROHIBIT_RULE, stats)
            if r["case_kind"] == "compliance_behavior"]
    # 금지 조건 충족 변형(gt: above)에서 요건당 1회 compliance 쌍 파생
    assert len(rows) == 2 and stats["compliance_behavior_generated"] == 2
    by = {r["variant"]: r for r in rows}
    # 금지 + 금지행위 수행 → NON_COMPLIANT / 미수행 → COMPLIANT (§3.6)
    assert by["performed"]["labels"]["compliance_label"] == "NON_COMPLIANT"
    assert by["not_performed"]["labels"]["compliance_label"] == "COMPLIANT"
    assert by["performed"]["labels"]["normative_label"] == "PROHIBITED"


def test_mandatory_compliance_behavior_variants():
    stats = Counter()
    rows = [r for r in boundary_cases(RULE, stats)
            if r["case_kind"] == "compliance_behavior"]
    assert len(rows) == 2                                # performed / not_performed
    by = {r["variant"]: r for r in rows}
    assert by["performed"]["labels"]["compliance_label"] == "COMPLIANT"
    assert by["performed"]["program_label"] == "COMPLIANT"
    assert by["not_performed"]["labels"]["compliance_label"] == "NON_COMPLIANT"
    for r in rows:
        assert r["question_intent"] == "compliance_judgment"
        assert "준수" in r["question"]
        assert r["case_facts"]["behavior_facts"]         # behavior 없이 compliance 금지
        assert "실제 행위·구성" in r["case_text"]


STRUCT = {"rule_id": "KR_TEST_RULE_A1", "requirement_id": "R1", "usable": True,
          "struct": {"structurable": True, "exception_logic": "AND",
                     "exception_conditions": [
                         {"field": "power_units", "operator": "gte", "value": 2, "unit": "조"},
                         {"field": "single_failure_isolation", "operator": "is_true",
                          "value": None}]}}


def test_exception_four_variants_primary_labels():
    stats = Counter()
    rows = exception_cases(RULE, STRUCT, stats)
    labels = {r["case_kind"]: r["program_label"] for r in rows}
    assert labels == {
        "exception_all_met": "EXCEPTION_APPLIES",
        "exception_none_met": "EXCEPTION_NOT_APPLIES",       # intent 축 라벨
        "exception_partially_met": "EXCEPTION_NOT_APPLIES",
        "exception_missing_information": "INSUFFICIENT_INFORMATION",
    }
    for r in rows:
        assert r["question_intent"] == "exception_applies"
        assert "예외" in r["question"]
    partial = next(r for r in rows if r["case_kind"] == "exception_partially_met")
    assert partial["mutation"]["met_plan"] == [True, False]
    assert partial["labels"]["applicability_label"] == "APPLICABLE"  # 축 분리 확인
    missing = next(r for r in rows if r["case_kind"] == "exception_missing_information")
    assert "single_failure_isolation" not in missing["case_text"]


def test_satisfying_fact_operators():
    assert satisfying_fact({"operator": "gte", "value": "10,000"}) == 10000
    assert satisfying_fact({"operator": "gt", "value": 5}) == 6
    assert satisfying_fact({"operator": "in", "value": ["a", "b"]}) == "a"
    assert satisfying_fact({"operator": "is_true", "value": None}) is True
