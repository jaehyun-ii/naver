"""품질 게이트 — 예외 구조 정적 검사·fact realization·readiness 판정 (LLM 불필요)."""
from collections import Counter

from scripts.aireg_kr.build_cases import (boundary_cases, fact_realization_checks,
                                          render_case_text)
from scripts.aireg_kr.exception_gate import static_struct_checks
from scripts.aireg_kr.readiness import evaluate_bulk_generation_readiness

SRC = "동력장치 2조 이상을 갖추고 202.의 1항에 적합한 경우 보조조타장치를 생략할 수 있다."


# ── 예외 구조 정적 검사(§1.5) ───────────────────────────────────────────
def test_static_checks_pass():
    struct = {"exception_conditions": [
        {"field": "power_units", "operator": "gte", "value": 2, "unit": "조",
         "evidence": "동력장치 2조 이상"},
        {"field": "art202_conform", "operator": "is_true", "value": None,
         "evidence": "202.의 1항에 적합"}]}
    assert static_struct_checks(struct, SRC) == []


def test_static_checks_detect_fabrication():
    # 원문에 없는 수치
    s1 = {"exception_conditions": [{"field": "x", "operator": "gte", "value": 999,
                                    "evidence": ""}]}
    assert any("수치" in f for f in static_struct_checks(s1, SRC))
    # evidence가 원문 부분문자열이 아님
    s2 = {"exception_conditions": [{"field": "x", "operator": "gte", "value": 2,
                                    "evidence": "원문에 없는 근거 문장입니다"}]}
    assert any("evidence" in f for f in static_struct_checks(s2, SRC))
    # 중복·빈 조건
    c = {"field": "x", "operator": "gte", "value": 2, "evidence": ""}
    assert any("중복" in f for f in static_struct_checks(
        {"exception_conditions": [c, dict(c)]}, SRC))
    assert static_struct_checks({"exception_conditions": []}, SRC) == ["빈 예외 조건"]


# ── fact realization(§2) ────────────────────────────────────────────────
RULE = {"section_path": "규칙 > 103. 재료", "article_text": "…"}


def _bf(value, certainty="CONFIRMED"):
    return {"prohibited_action_performed": {
        "value": value, "certainty": certainty,
        "time_scope": "CURRENT_CONFIGURATION", "evidence_type": "EXPLICIT_STATEMENT"}}


def test_confirmed_false_rendered_explicitly():
    facts = {"applicability_facts": {"설계온도": {"value": 351, "unit": "°C"}},
             "behavior_facts": _bf(False)}
    text = render_case_text(RULE, facts, {"prohibited_action_performed": "금지 행위 수행"})
    assert "수행되지 않으며" in text and "확인되었다" in text
    assert "아니오" not in text and "예정" not in text          # 모호 표현 금지
    checks = fact_realization_checks(text, facts["behavior_facts"],
                                     {"prohibited_action_performed": "금지 행위 수행"})
    assert checks[0]["realized"] and not checks[0]["weakened"] \
        and not checks[0]["contradicted"]


def test_confirmed_true_rendered_explicitly():
    facts = {"applicability_facts": {}, "behavior_facts": _bf(True)}
    text = render_case_text(RULE, facts)
    assert "실제로 수행되고 있음" in text
    checks = fact_realization_checks(text, facts["behavior_facts"])
    assert checks[0]["realized"] and not checks[0]["contradicted"]


def test_weakened_and_contradicted_detection():
    bf = _bf(False)
    weak = "- prohibited_action_performed: 사용하지 않을 예정이다"
    checks = fact_realization_checks(weak, bf)
    assert checks[0]["weakened"] or not checks[0]["realized"]
    flipped = "- prohibited_action_performed: 현재 선박에서 해당 행위가 실제로 수행되고 있음이 기록으로 확인되었다."
    checks = fact_realization_checks(flipped, bf)
    assert checks[0]["contradicted"]


def test_compliance_case_carries_certainty():
    rule = {"id": "R", "section_path": "규칙 > 103. 재료",
            "article_text": "설계온도가 350 °C를 초과하는 관계통에 사용하여서는 아니 된다.",
            "card": {"requirements": [{"requirement_id": "R1", "modality": "금지",
                                       "requirement_text": "사용하여서는 아니 된다.",
                                       "conditions": [{"subject": "설계온도", "operator": "초과",
                                                       "value": "350", "unit": "°C"}]}]}}
    rows = [r for r in boundary_cases(rule, Counter())
            if r["case_kind"] == "compliance_behavior"]
    bf = rows[0]["case_facts"]["behavior_facts"]
    fv = next(iter(bf.values()))
    assert fv["certainty"] == "CONFIRMED" and fv["evidence_type"] == "EXPLICIT_STATEMENT"
    assert rows[0]["fact_realization_checks"][0]["realized"]


# ── readiness(§9) ───────────────────────────────────────────────────────
_GOOD = {"program_verifier_match_rate": 0.97, "prohibition_match_rate": 0.96,
         "exception_match_rate": 0.92, "semantic_axis_mismatch_rate": 0.0,
         "fact_realization_pass_rate": 1.0, "verifier_abstention_conflict_rate": 0.0,
         "expert_review_queue_rate": 0.05, "strict_derivable_rate": 0.9,
         "canonical_group_leakage": 0, "exact_duplicate_leakage": 0,
         # 스위트 QA 게이트(2026-07-21 추가) — CRITERIA와 키를 일치시킨다
         "suite_overall_accept_rate": 0.9, "suite_review_rate": 0.02,
         "suite_error_count": 0, "suite_min_track_samples": 120,
         "suite_track_gates_ok": 1}


def test_readiness_approved_and_not():
    assert evaluate_bulk_generation_readiness(dict(_GOOD))["status"] == "APPROVED"
    bad = dict(_GOOD, prohibition_match_rate=0.90)
    out = evaluate_bulk_generation_readiness(bad)
    assert out["status"] == "NOT_APPROVED" and "prohibition_match_rate" in out["blocking_reasons"]
    # leakage는 무조건 미승인
    leak = dict(_GOOD, canonical_group_leakage=1)
    assert evaluate_bulk_generation_readiness(leak)["status"] == "NOT_APPROVED"
    # 지표 None(분모 없음)은 미충족 처리
    none_m = dict(_GOOD, exception_match_rate=None)
    assert evaluate_bulk_generation_readiness(none_m)["status"] == "NOT_APPROVED"


def test_strict_excludes_new_conflicts():
    from scripts.aireg_kr.labels import should_include_for_training
    lc = {"agreement": "VERIFIER_ABSTENTION_CONFLICT"}
    inc, why = should_include_for_training(verifier_verdict=None, label_consistency=lc,
                                           label_policy="strict")
    assert not inc and why == "verifier_abstention_conflict"
    inc, _ = should_include_for_training(verifier_verdict=None, label_consistency=lc,
                                         label_policy="report-only")
    assert inc


# ── 조건 자족성 게이트(유보 충돌 교정) ──────────────────────────────────
def test_static_condition_checks():
    from scripts.aireg_kr.exception_gate import static_condition_checks
    # 상대 기준 값 → 차단
    fails = static_condition_checks(
        [{"subject": "시험 압력", "operator": "이상", "value": "설계압력의 1.5배"}],
        "수압시험을 하여야 한다", "…")
    assert any("상대 기준" in f for f in fails)
    # 원문이 종류별 상이 기준인데 종류 조건 없음 → 차단
    fails = static_condition_checks(
        [{"subject": "설계온도", "operator": "초과", "value": "350", "unit": "°C"}],
        "관은 종류에 따라 다음 표에 따른 온도를 초과하는 관계통에 사용하여서는 아니 된다",
        "제1종관 350°C, SPP 230°C …")
    assert any("종류" in f for f in fails)
    # 종류 조건이 함께 있으면 통과
    fails = static_condition_checks(
        [{"subject": "관 종류", "operator": "equals", "value": "SPP"},
         {"subject": "설계온도", "operator": "초과", "value": "230", "unit": "°C"}],
        "관은 종류에 따라 …", "…")
    assert fails == []


def test_boundary_respects_condition_gate():
    from collections import Counter
    from scripts.aireg_kr.build_cases import boundary_cases
    rule = {"id": "R", "section_path": "규칙 > 103.", "article_text": "…",
            "card": {"requirements": [{"requirement_id": "R1", "modality": "금지",
                                       "requirement_text": "사용하여서는 아니 된다.",
                                       "conditions": [{"subject": "설계온도", "operator": "초과",
                                                       "value": "350", "unit": "°C"}]}]}}
    stats = Counter()
    # 게이트 REJECT → 생성 안 함
    assert boundary_cases(rule, stats, {"R::R1": {"decision": "REJECT"}}) == []
    assert stats["boundary_skip_condition_gate"] == 1
    # 게이트 ACCEPT → 생성
    rows = boundary_cases(rule, Counter(), {"R::R1": {"decision": "ACCEPT"}})
    assert len(rows) >= 3
    # 게이트 미수행(None) → 하위 호환 통과
    assert len(boundary_cases(rule, Counter(), None)) >= 3
