"""labels(3자 비교·정책) + rewards(program_label/coverage) 단위 검증."""
from scripts.aireg_kr.answer_format import format_regulation_answer
from scripts.aireg_kr.labels import (canonicalize_label, compare_labels,
                                     should_include_for_training)
from scripts.aireg_kr.rewards import (condition_coverage_reward,
                                      extract_decision_label,
                                      program_label_reward)


# ── 3자 비교 ─────────────────────────────────────────────────────────────
def test_canonicalize():
    assert canonicalize_label("compliant") == "APPLICABLE"
    assert canonicalize_label("판단 불가") == "INSUFFICIENT_INFORMATION"
    assert canonicalize_label("uncertain") == "INSUFFICIENT_INFORMATION"
    assert canonicalize_label(None) == "UNKNOWN"


def test_three_way_agreements():
    assert compare_labels("APPLICABLE", "APPLICABLE", "APPLICABLE")["agreement"] == "THREE_WAY_MATCH"
    lc = compare_labels("EXCEPTION_APPLIES", "APPLICABLE", "APPLICABLE")
    assert lc["agreement"] == "GENERATOR_DISAGREES" and lc["action"] == "REJECT"
    lc = compare_labels("APPLICABLE", "NOT_APPLICABLE", "APPLICABLE")
    assert lc["agreement"] == "VERIFIER_DISAGREES" and lc["action"] == "REVERIFY"
    lc = compare_labels("APPLICABLE", "APPLICABLE", "NOT_APPLICABLE")
    assert lc["agreement"] == "PROGRAM_DISAGREES" and lc["action"] == "INSPECT_EVALUATOR"
    lc = compare_labels("APPLICABLE", "NOT_APPLICABLE", "EXCEPTION_APPLIES")
    assert lc["agreement"] == "ALL_DISAGREE"
    lc = compare_labels("compliant", "compliant", None)   # program UNKNOWN
    assert lc["agreement"] == "PROGRAM_UNEVALUABLE" and lc["action"] == "KEEP_EXISTING_POLICY"


def test_policy_strict_vs_report_only():
    mismatch = compare_labels("APPLICABLE", "NOT_APPLICABLE", "APPLICABLE")
    inc, why = should_include_for_training(verifier_verdict=None,
                                           label_consistency=mismatch,
                                           label_policy="strict")
    assert not inc and why == "program_verifier_label_mismatch"
    inc, why = should_include_for_training(verifier_verdict=None,
                                           label_consistency=mismatch,
                                           label_policy="report-only")
    assert inc and why == "report_only"
    # 공통: 블라인드 REJECT는 정책 무관 제외
    inc, why = should_include_for_training(verifier_verdict="REJECT",
                                           label_consistency=None,
                                           label_policy="report-only")
    assert not inc and why == "verifier_rejected"
    # program UNKNOWN → 기존 정책 유지
    keep = compare_labels("compliant", "compliant", "UNKNOWN")
    inc, why = should_include_for_training(verifier_verdict="ACCEPT",
                                           label_consistency=keep,
                                           label_policy="strict")
    assert inc and why == "program_unevaluable_keep_existing"
    # generator만 불일치는 strict/verifier-preferred 공통 제외
    gen_bad = compare_labels("EXCEPTION_APPLIES", "APPLICABLE", "APPLICABLE")
    inc, why = should_include_for_training(verifier_verdict="ACCEPT",
                                           label_consistency=gen_bad,
                                           label_policy="verifier-preferred")
    assert not inc and why == "generator_label_mismatch"


# ── rewards ─────────────────────────────────────────────────────────────
def _ans(decision):
    return format_regulation_answer(decision=decision, evidence="근거")


def test_extract_decision_label():
    assert extract_decision_label(_ans("적용")) == "APPLICABLE"
    assert extract_decision_label(_ans("적용되지 않음")) == "NOT_APPLICABLE"
    assert extract_decision_label(_ans("예외 적용")) == "EXCEPTION_APPLIES"
    assert extract_decision_label(_ans("판단 불가")) == "INSUFFICIENT_INFORMATION"
    assert extract_decision_label("자유 서술") is None


def test_program_label_reward_and_evaluable_gate():
    comps = [_ans("적용"), _ans("적용"), _ans("미적용")]
    out = program_label_reward(
        ["p"] * 3, comps,
        program_label=["APPLICABLE", "APPLICABLE", "APPLICABLE"],
        evaluable=[True, False, True])
    assert out == [1.0, 0.0, 0.0]     # 정답 / 비활성 / 오답
    # UNKNOWN 프로그램 라벨을 정답처럼 강제하지 않음
    out = program_label_reward(["p"], [_ans("판단 불가")],
                               program_label=["UNKNOWN"], evaluable=[True])
    assert out == [0.0]


def test_condition_coverage_reward():
    conds = [{"field": "gross_tonnage", "expected_value": "10,000"},
             {"field": "power_units", "expected_value": 2}]
    good = "조건 검토: gross tonnage 10000 확인, power_units 2조 충족"
    out = condition_coverage_reward(["p", "p"], [good, "무관한 답"],
                                    condition_results=[conds, conds])
    assert out[0] == 1.0 and out[1] == 0.0
    assert condition_coverage_reward(["p"], ["x"], condition_results=[[]]) == [0.0]


# ── primary 축 3자 비교(§9) ─────────────────────────────────────────────
def test_compare_primary_labels_axes():
    from scripts.aireg_kr.labels import compare_primary_labels
    kw = dict(question_intent="action_prohibited",
              generation_primary_label="PROHIBITED",
              program_primary_label="PROHIBITED")
    assert compare_primary_labels(**kw, verifier_primary_label="PROHIBITED")[
        "agreement"] == "THREE_WAY_MATCH"
    assert compare_primary_labels(**kw, verifier_primary_label="NOT_PROHIBITED")[
        "agreement"] == "VERIFIER_DISAGREES"
    # 검증기가 intent 축을 바꿔 해석 → 라벨 비교 대신 SEMANTIC_AXIS_MISMATCH
    out = compare_primary_labels(**kw, verifier_primary_label="APPLICABLE",
                                 verifier_interpreted_intent="rule_applies")
    assert out["agreement"] == "SEMANTIC_AXIS_MISMATCH" and out["action"] == "REVERIFY"
    # program NOT_ASSESSED → 기존 정책 유지
    assert compare_primary_labels(
        question_intent="compliance_judgment", generation_primary_label="COMPLIANT",
        verifier_primary_label="COMPLIANT", program_primary_label="NOT_ASSESSED")[
        "agreement"] == "PROGRAM_UNEVALUABLE"


def test_semantic_mismatch_strict_exclusion():
    from scripts.aireg_kr.labels import compare_primary_labels
    lc = compare_primary_labels(question_intent="action_prohibited",
                                generation_primary_label="PROHIBITED",
                                verifier_primary_label="APPLICABLE",
                                program_primary_label="PROHIBITED",
                                verifier_interpreted_intent="rule_applies")
    inc, why = should_include_for_training(verifier_verdict=None, label_consistency=lc,
                                           label_policy="strict")
    assert not inc and why == "semantic_axis_mismatch"
    inc, _ = should_include_for_training(verifier_verdict=None, label_consistency=lc,
                                         label_policy="report-only")
    assert inc


# ── intent alignment reward(§12.1) ──────────────────────────────────────
def test_intent_alignment_reward():
    from scripts.aireg_kr.rewards import intent_alignment_reward
    prohibited = _ans("금지됨")
    applied = _ans("적용")
    freeform = "자유 서술"
    out = intent_alignment_reward(
        ["p"] * 3, [prohibited, applied, freeform],
        question_intent=["action_prohibited"] * 3)
    assert out == [1.0, 0.25, 0.0]        # 정축 / 축 혼동 / 추출 불가
    obligation = _ans("의무 발생")
    assert intent_alignment_reward(["p"], [obligation],
                                   question_intent=["obligation_triggered"]) == [1.0]
    # intent 메타 없음 → 상수 0.0 (하위 호환)
    assert intent_alignment_reward(["p"], [prohibited]) == [0.0]
    # 판단 불가는 어느 축에서든 유효
    assert intent_alignment_reward(["p"], [_ans("판단 불가")],
                                   question_intent=["action_prohibited"]) == [1.0]


def test_program_label_reward_uses_primary_and_display_labels():
    from scripts.aireg_kr.rewards import program_label_reward
    out = program_label_reward(
        ["p"] * 3, [_ans("금지됨"), _ans("금지되지 않음"), _ans("적합")],
        primary_label=["PROHIBITED", "PROHIBITED", "NOT_ASSESSED"],
        evaluable=[True, True, True])
    assert out == [1.0, 0.0, 0.0]         # 정답 / 오답 / NOT_ASSESSED 비활성
