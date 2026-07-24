"""condition_evaluator v1 — 연산자·단위·논리·예외 결정성 검증."""
from scripts.aireg_kr.condition_evaluator import (EvaluationStatus, combine,
                                                  convert_value,
                                                  evaluate_condition,
                                                  evaluate_rule)

S, N, U, I = (EvaluationStatus.SATISFIED, EvaluationStatus.NOT_SATISFIED,
              EvaluationStatus.UNKNOWN, EvaluationStatus.INVALID)


def _c(op, value, field="gross_tonnage", unit="GT", **kw):
    return {"field": field, "operator": op, "value": value, "unit": unit, **kw}


# ── 단일 조건: 수치 경계 ────────────────────────────────────────────────
def test_numeric_boundaries():
    for op, cases in {
        "gte": [(9999, N), (10000, S), (10001, S)],
        "gt": [(9999, N), (10000, N), (10001, S)],
        "lte": [(9999, S), (10000, S), (10001, N)],
        "lt": [(9999, S), (10000, N), (10001, N)],
    }.items():
        for actual, want in cases:
            ev = evaluate_condition(_c(op, 10000), {"gross_tonnage": actual})
            assert ev.status == want, (op, actual, ev.reason)


def test_korean_operator_aliases():
    assert evaluate_condition(_c("이상", 10000), {"gross_tonnage": 10000}).status == S
    assert evaluate_condition(_c("미만", 10000), {"gross_tonnage": 10000}).status == N
    # "해당"/"기타"는 의미가 열려 있어 미지원 → UNKNOWN
    assert evaluate_condition(_c("해당", "x"), {"gross_tonnage": 1}).status == U


def test_between():
    c = _c("between_inclusive", [10, 20], field="speed", unit="kn")
    assert evaluate_condition(c, {"speed": 10}).status == S
    assert evaluate_condition(c, {"speed": 21}).status == N
    c2 = _c("between_exclusive", [10, 20], field="speed", unit="kn")
    assert evaluate_condition(c2, {"speed": 10}).status == N


def test_enum_string_bool_existence():
    c = {"field": "ship_type", "operator": "in", "value": ["oil_tanker", "chemical_tanker"]}
    assert evaluate_condition(c, {"ship_type": "Oil Tanker"}).status == S   # alias 정규화
    assert evaluate_condition(c, {"ship_type": "bulk_carrier"}).status == N
    eq = {"field": "ship_type", "operator": "equals", "value": "oil_tanker"}
    assert evaluate_condition(eq, {"ship_type": "OIL  TANKER"}).status == S
    b = {"field": "double_hull", "operator": "is_true", "value": None}
    assert evaluate_condition(b, {"double_hull": "TRUE"}).status == S
    assert evaluate_condition(b, {"double_hull": False}).status == N
    ex = {"field": "ship_type", "operator": "exists", "value": None}
    assert evaluate_condition(ex, {"ship_type": "x"}).status == S
    assert evaluate_condition(ex, {}).status == N


def test_missing_value_type_error_unsupported():
    assert evaluate_condition(_c("gte", 10000), {}).status == U               # 값 누락
    assert evaluate_condition(_c("gte", 10000), {"gross_tonnage": "미상"}).status == U
    assert evaluate_condition(_c("gte", "높음"), {"gross_tonnage": 1}).status == I  # 기대값 비수치
    assert evaluate_condition(_c("regex_match", "a"), {"gross_tonnage": 1}).status == U
    eq = {"field": "f", "operator": "equals", "value": "text"}
    assert evaluate_condition(eq, {"f": 123}).status == U                     # 자료형 불일치


def test_qualitative_marker_unknown():
    c = {"field": "강도", "operator": "gte", "value": "충분한 강도",
         "evidence": "충분한 강도를 가질 것"}
    ev = evaluate_condition(c, {"강도": 10})
    assert ev.status == U and "정성" in ev.reason


def test_thousands_separator_and_numeric_string():
    assert evaluate_condition(_c("gte", "10,000"), {"gross_tonnage": "12,000"}).status == S


# ── 단위 ────────────────────────────────────────────────────────────────
def test_unit_conversion_allowed_and_forbidden():
    assert convert_value(1500, "mm", "m") == 1.5
    assert convert_value(2, "t", "kg") == 2000
    assert convert_value(50, "percent", "decimal") == 0.5
    assert convert_value(1, "GT", "DWT") is None          # 금지 변환
    assert convert_value(1, "GT", "kg") is None
    c = _c("gte", 1.5, field="length", unit="m")
    assert evaluate_condition(c, {"length": {"value": 1500, "unit": "mm"}}).status == S
    bad = evaluate_condition(_c("gte", 10000, unit="GT"),
                             {"gross_tonnage": {"value": 10000, "unit": "DWT"}})
    assert bad.status == U and "단위" in bad.reason


def test_semantically_different_fields_not_substituted():
    # LOA 조건에 LBP 값을 대입하지 않는다 — 필드명이 다르면 값 없음(UNKNOWN)
    c = {"field": "length_loa", "operator": "gte", "value": 150, "unit": "m"}
    assert evaluate_condition(c, {"length_lbp": 160}).status == U


# ── 논리 결합 ────────────────────────────────────────────────────────────
def test_and_or_three_valued():
    assert combine([S, S], "AND") == S
    assert combine([S, N], "AND") == N
    assert combine([S, U], "AND") == U
    assert combine([N, U], "AND") == N        # NOT_SATISFIED 우선
    assert combine([N, S], "OR") == S
    assert combine([N, N], "OR") == N
    assert combine([N, U], "OR") == U
    assert combine([I, S], "AND") == U        # INVALID는 UNKNOWN 동급 보수 처리


def test_nested_tree():
    tree = {"logic": "AND", "conditions": [
        {"field": "ship_type", "operator": "equals", "value": "oil_tanker"},
        {"logic": "OR", "conditions": [
            {"field": "gross_tonnage", "operator": "gte", "value": 10000, "unit": "GT"},
            {"field": "length", "operator": "gte", "value": 150, "unit": "m"},
        ]},
    ]}
    card = {"condition_tree": tree}
    assert evaluate_rule(card, {"ship_type": "oil_tanker", "gross_tonnage": 9000,
                                "length": 160}).derived_label == "APPLICABLE"
    assert evaluate_rule(card, {"ship_type": "oil_tanker", "gross_tonnage": 9000,
                                "length": 100}).derived_label == "NOT_APPLICABLE"
    assert evaluate_rule(card, {"ship_type": "oil_tanker",
                                "gross_tonnage": 9000}).derived_label == "INSUFFICIENT_INFORMATION"


# ── 예외 평가 (§6.1 매핑) ───────────────────────────────────────────────
_EX_CARD = {
    "conditions": [{"field": "ship_type", "operator": "equals", "value": "oil_tanker"}],
    "exception_conditions": [
        {"field": "power_units", "operator": "gte", "value": 2, "unit": "조"},
        {"field": "single_failure_isolation", "operator": "is_true", "value": None},
    ],
    "exception_logic": "AND",
}


def test_exception_all_met():
    ev = evaluate_rule(_EX_CARD, {"ship_type": "oil_tanker", "power_units": 2,
                                  "single_failure_isolation": True})
    assert ev.derived_label == "EXCEPTION_APPLIES" and ev.exception_status == S


def test_exception_partially_met_means_applicable():
    ev = evaluate_rule(_EX_CARD, {"ship_type": "oil_tanker", "power_units": 2,
                                  "single_failure_isolation": False})
    assert ev.exception_status == N and ev.derived_label == "APPLICABLE"


def test_exception_missing_info():
    ev = evaluate_rule(_EX_CARD, {"ship_type": "oil_tanker", "power_units": 2})
    assert ev.exception_status == U and ev.derived_label == "INSUFFICIENT_INFORMATION"
    assert "single_failure_isolation" in ev.missing_fields


def test_base_not_satisfied_skips_exception_label():
    ev = evaluate_rule(_EX_CARD, {"ship_type": "bulk_carrier", "power_units": 2,
                                  "single_failure_isolation": True})
    assert ev.derived_label == "NOT_APPLICABLE"


def test_unstructured_card_unknown():
    ev = evaluate_rule({"requirements": [{"exception": "우리 선급이 인정하는 경우"}]}, {})
    assert ev.derived_label == "UNKNOWN" and "구조화 조건 없음" in ev.errors
