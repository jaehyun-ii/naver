"""결정적 조건 평가기 v1 — Rule Card 구조화 조건 × 사례 facts 프로그램 판정.

범용 규정 추론 엔진이 아니다. 기계적으로 안전하게 판정 가능한 조건(수치 비교·
열거형·불리언·존재 여부)만 평가하고, 정성적·비구조 조건은 반드시 UNKNOWN으로
남긴다. UNKNOWN을 NOT_SATISFIED로 강등하지 않는다.

원문 텍스트를 직접 해석하지 않는다 — Rule Card v2가 이미 정규화한 operator
(이상/이하/초과/미만/동일 enum 포함)만 고정 별칭표로 받아들인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

EVALUATOR_VERSION = "condition_evaluator_v1"


class EvaluationStatus(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ConditionEvaluation:
    field: str
    operator: str
    expected_value: Any
    actual_value: Any
    unit: str | None
    status: EvaluationStatus
    reason: str
    normalized_expected: Any = None
    normalized_actual: Any = None


@dataclass(frozen=True)
class RuleEvaluation:
    status: EvaluationStatus
    condition_logic: str
    condition_results: list[ConditionEvaluation]
    exception_status: EvaluationStatus | None
    exception_results: list[ConditionEvaluation]
    derived_label: str
    missing_fields: list[str] = field(default_factory=list)
    unsupported_conditions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── 연산자 ───────────────────────────────────────────────────────────────
NUMERIC_OPS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte",
                         "between_inclusive", "between_exclusive"})
STRING_OPS = frozenset({"equals", "not_equals", "in", "not_in"})
BOOL_OPS = frozenset({"is_true", "is_false"})
EXISTENCE_OPS = frozenset({"exists", "not_exists"})
SUPPORTED_OPS = NUMERIC_OPS | STRING_OPS | BOOL_OPS | EXISTENCE_OPS

# Rule Card v2 enum → 정식 연산자. 자연어 해석이 아니라 고정 정규화표다.
# "해당"/"기타"는 의미가 열려 있어 미지원(UNKNOWN) — 임의 매핑 금지.
OPERATOR_ALIASES = {
    "이상": "gte", "이하": "lte", "초과": "gt", "미만": "lt",
    "동일": "eq", "같다": "eq", "같지 않다": "ne",
}

# 정성 조건 마커 — 자동 판정 금지(UNKNOWN)
QUALITATIVE_MARKERS = (
    "인정하는 경우", "충분한", "가능한 한", "적절한", "필요하다고 판단",
    "동등한 수준", "통상적인", "현저한", "만족스럽", "특별히 고려",
)

# ── 단위 registry — 허용 변환만 수행(기준 단위 배율) ─────────────────────
UNIT_CONVERSIONS: dict[str, dict[str, float]] = {
    "length": {"mm": 0.001, "cm": 0.01, "m": 1.0},
    "mass": {"kg": 1.0, "t": 1000.0},
    "time": {"sec": 1.0, "s": 1.0, "min": 60.0, "hour": 3600.0, "h": 3600.0},
    "angle": {"degree": 1.0, "deg": 1.0, "도": 1.0},
    "ratio": {"percent": 0.01, "%": 0.01, "decimal": 1.0},
    "power": {"w": 1.0, "kw": 1000.0, "mw": 1_000_000.0},
    # GT·DWT·kn 등은 각자 고유 차원 — 상호 변환 금지, 동일 단위 비교만 허용
    "gross_tonnage": {"gt": 1.0, "톤": 1.0, "ton": 1.0},
    "deadweight": {"dwt": 1.0},
    "speed": {"kn": 1.0, "knot": 1.0},
    "pressure": {"mpa": 1.0},
    "temperature": {"°c": 1.0, "℃": 1.0},
    "count": {"조": 1.0, "개": 1.0, "기": 1.0, "척": 1.0},
}
_UNIT_DIM = {u: dim for dim, units in UNIT_CONVERSIONS.items() for u in units}

_NUM_WITH_UNIT = None  # 지연 컴파일(모듈 하단 re import 순서 유지)


def parse_number_with_unit(value) -> tuple[float | None, str | None]:
    """"425 °C"·"10,000 kW"류 문자열 → (수치, 단위 토큰). 형식이 다르면 (None, None).

    수치 뒤 단일 단위 토큰만 허용 — 그 밖의 문자열은 해석하지 않는다(임의 변환 금지).
    """
    import re
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None
    if not isinstance(value, str):
        return None, None
    m = re.fullmatch(r"\s*([-+]?[\d,]+(?:\.\d+)?)\s*([^\s\d,]{0,6})\s*", value)
    if not m:
        return None, None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None, None
    return num, (m.group(2) or None)


def _norm_unit(unit: str | None) -> str | None:
    if unit is None or not str(unit).strip():
        return None
    return str(unit).strip().lower()


def convert_value(value: float, from_unit: str, to_unit: str) -> float | None:
    """허용 변환표 내에서만 변환. 차원 불일치(GT↔DWT 등)·미등록 단위는 None."""
    fu, tu = _norm_unit(from_unit), _norm_unit(to_unit)
    if fu == tu:
        return value
    dim_f, dim_t = _UNIT_DIM.get(fu), _UNIT_DIM.get(tu)
    if dim_f is None or dim_f != dim_t:
        return None
    return value * UNIT_CONVERSIONS[dim_f][fu] / UNIT_CONVERSIONS[dim_t][tu]


# ── 값 정규화 ────────────────────────────────────────────────────────────
_BOOL_STRINGS = {"true": True, "false": False, "yes": True, "no": False,
                 "예": True, "아니오": False}


def normalize_scalar(value: Any) -> Any:
    """숫자 문자열·천단위 구분기호·불리언 문자열·문자열 케이스 정규화."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        low = s.lower()
        if low in _BOOL_STRINGS:
            return _BOOL_STRINGS[low]
        num = s.replace(",", "")
        try:
            f = float(num)
            return int(f) if f.is_integer() else f
        except ValueError:
            pass
        return " ".join(low.split()).replace(" ", "_")  # 열거형 alias: 소문자·언더스코어
    return value


def _as_number(value: Any) -> float | None:
    v = normalize_scalar(value)
    if isinstance(v, bool):
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _number_in_unit(value: Any, declared_unit: str | None) -> float | None:
    """수치 또는 "425 °C"류 수치+단위 문자열 → 선언 단위 기준 수치.

    내장 단위가 선언 단위와 다르면 허용 변환표로만 변환(불가 시 None).
    """
    n = _as_number(value)
    if n is not None:
        return n
    num, emb = parse_number_with_unit(value)
    if num is None:
        return None
    if emb and declared_unit and _norm_unit(emb) != _norm_unit(declared_unit):
        return convert_value(num, emb, declared_unit)
    return num


def _fact(case_facts: dict, field_name: str) -> tuple[Any, str | None, bool]:
    """사례 값 조회 — {"field": 값} 또는 {"field": {"value":…, "unit":…}} 허용."""
    if field_name not in case_facts:
        return None, None, False
    raw = case_facts[field_name]
    if isinstance(raw, dict) and "value" in raw:
        return raw.get("value"), raw.get("unit"), True
    return raw, None, True


# ── 단일 조건 평가 ───────────────────────────────────────────────────────
def evaluate_condition(condition: dict, case_facts: dict) -> ConditionEvaluation:
    """단일 조건 평가. 미지원·비교불능은 UNKNOWN, 구조 오류는 INVALID."""
    field_name = str(condition.get("field") or condition.get("subject") or "").strip()
    raw_op = str(condition.get("operator") or "").strip()
    op = OPERATOR_ALIASES.get(raw_op, raw_op)
    expected = condition.get("value") if "value" in condition else condition.get("expected_value")
    unit = condition.get("unit")

    def result(status: EvaluationStatus, reason: str, actual=None,
               ne=None, na=None) -> ConditionEvaluation:
        return ConditionEvaluation(field=field_name, operator=raw_op,
                                   expected_value=expected, actual_value=actual,
                                   unit=unit, status=status, reason=reason,
                                   normalized_expected=ne, normalized_actual=na)

    if not field_name:
        return result(EvaluationStatus.INVALID, "field 없음")
    if not op:
        return result(EvaluationStatus.UNKNOWN, "operator 없음")
    # 정성 조건 마커 — 값·필드 어디에 있든 자동 판정 금지
    blob = f"{field_name} {expected} {condition.get('evidence', '')}"
    for marker in QUALITATIVE_MARKERS:
        if marker in blob:
            return result(EvaluationStatus.UNKNOWN, f"정성 조건({marker}) — 자동 판정 제외")
    if op not in SUPPORTED_OPS:
        return result(EvaluationStatus.UNKNOWN, f"미지원 연산자: {raw_op!r}")

    actual, actual_unit, present = _fact(case_facts, field_name)

    if op in EXISTENCE_OPS:
        ok = present if op == "exists" else not present
        return result(EvaluationStatus.SATISFIED if ok else EvaluationStatus.NOT_SATISFIED,
                      f"필드 {'존재' if present else '부재'}", actual)
    if not present or actual is None or (isinstance(actual, str) and not actual.strip()):
        return result(EvaluationStatus.UNKNOWN, f"사례 값 없음: {field_name}")

    if op in NUMERIC_OPS:
        na = _as_number(actual)
        if na is None:
            na = _number_in_unit(actual, unit)
        if op in ("between_inclusive", "between_exclusive"):
            if not (isinstance(expected, (list, tuple)) and len(expected) == 2):
                return result(EvaluationStatus.INVALID, "between 기대값은 [하한, 상한]")
            lo, hi = (_number_in_unit(expected[0], unit),
                      _number_in_unit(expected[1], unit))
            if na is None or lo is None or hi is None:
                return result(EvaluationStatus.UNKNOWN, "수치 변환 불가", actual)
            na_c = _convert_actual(na, actual_unit, unit)
            if na_c is None:
                return result(EvaluationStatus.UNKNOWN,
                              f"단위 변환 불가: {actual_unit}→{unit}", actual)
            ok = (lo <= na_c <= hi) if op == "between_inclusive" else (lo < na_c < hi)
            return result(_sat(ok), f"{lo} {'≤' if op.endswith('inclusive') else '<'} "
                          f"{na_c} {'≤' if op.endswith('inclusive') else '<'} {hi}: {ok}",
                          actual, (lo, hi), na_c)
        ne = _number_in_unit(expected, unit)
        if ne is None:
            return result(EvaluationStatus.INVALID, f"기대값이 수치가 아님: {expected!r}", actual)
        if na is None:
            na = _number_in_unit(actual, unit)
        if na is None:
            return result(EvaluationStatus.UNKNOWN, f"사례 값이 수치가 아님: {actual!r}", actual)
        na_c = _convert_actual(na, actual_unit, unit)
        if na_c is None:
            return result(EvaluationStatus.UNKNOWN,
                          f"단위 변환 불가: {actual_unit}→{unit}", actual)
        ok = {"eq": na_c == ne, "ne": na_c != ne, "gt": na_c > ne,
              "gte": na_c >= ne, "lt": na_c < ne, "lte": na_c <= ne}[op]
        sym = {"eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[op]
        return result(_sat(ok), f"{na_c} {sym} {ne}: {ok}", actual, ne, na_c)

    if op in BOOL_OPS:
        na = normalize_scalar(actual)
        if not isinstance(na, bool):
            return result(EvaluationStatus.UNKNOWN, f"불리언 아님: {actual!r}", actual)
        ok = na if op == "is_true" else not na
        return result(_sat(ok), f"{op}: {na}", actual, op == "is_true", na)

    # 열거형·문자열
    na = normalize_scalar(actual)
    if op in ("in", "not_in"):
        if not isinstance(expected, (list, tuple)):
            return result(EvaluationStatus.INVALID, "in 기대값은 목록", actual)
        ne = [normalize_scalar(v) for v in expected]
        ok = na in ne if op == "in" else na not in ne
        return result(_sat(ok), f"{na!r} {'∈' if op == 'in' else '∉'} {ne}", actual, ne, na)
    ne = normalize_scalar(expected)
    if type(na) is not type(ne):
        return result(EvaluationStatus.UNKNOWN,
                      f"자료형 불일치: {type(na).__name__} vs {type(ne).__name__}", actual, ne, na)
    ok = (na == ne) if op == "equals" else (na != ne)
    return result(_sat(ok), f"{na!r} {'==' if op == 'equals' else '!='} {ne!r}", actual, ne, na)


def _sat(ok: bool) -> EvaluationStatus:
    return EvaluationStatus.SATISFIED if ok else EvaluationStatus.NOT_SATISFIED


def _convert_actual(actual: float, actual_unit: str | None,
                    expected_unit: str | None) -> float | None:
    if _norm_unit(expected_unit) is None or _norm_unit(actual_unit) is None:
        # 한쪽이라도 단위 미기재 — 동일 단위 가정이 아니라, 기대 단위만 있으면 무단위
        # 사례 값을 그대로 비교(사례 facts가 조건과 같은 단위로 구조화되는 관례).
        return actual
    return convert_value(actual, actual_unit, expected_unit)


# ── 논리 결합 ────────────────────────────────────────────────────────────
def combine(statuses: list[EvaluationStatus], logic: str) -> EvaluationStatus:
    """AND/OR 3치 논리. INVALID는 UNKNOWN과 동급으로 보수 처리."""
    if not statuses:
        return EvaluationStatus.UNKNOWN
    ss = [EvaluationStatus.UNKNOWN if s == EvaluationStatus.INVALID else s
          for s in statuses]
    if logic in ("AND", "ALL"):
        if any(s == EvaluationStatus.NOT_SATISFIED for s in ss):
            return EvaluationStatus.NOT_SATISFIED
        if any(s == EvaluationStatus.UNKNOWN for s in ss):
            return EvaluationStatus.UNKNOWN
        return EvaluationStatus.SATISFIED
    if logic in ("OR", "ANY"):
        if any(s == EvaluationStatus.SATISFIED for s in ss):
            return EvaluationStatus.SATISFIED
        if any(s == EvaluationStatus.UNKNOWN for s in ss):
            return EvaluationStatus.UNKNOWN
        return EvaluationStatus.NOT_SATISFIED
    return EvaluationStatus.UNKNOWN  # NESTED 등 명시 트리 없는 논리 — 추정 금지


def evaluate_tree(node: dict, case_facts: dict,
                  flat: list[ConditionEvaluation]) -> EvaluationStatus:
    """명시적 조건 트리({logic, conditions:[조건|트리…]}) 재귀 평가."""
    if "logic" in node and "conditions" in node:
        child = [evaluate_tree(c, case_facts, flat) for c in node["conditions"]]
        return combine(child, str(node["logic"]).upper())
    ev = evaluate_condition(node, case_facts)
    flat.append(ev)
    return ev.status


# ── 규칙 평가 ────────────────────────────────────────────────────────────
LABEL = {"applicable": "APPLICABLE", "not_applicable": "NOT_APPLICABLE",
         "exception": "EXCEPTION_APPLIES", "insufficient": "INSUFFICIENT_INFORMATION",
         "unknown": "UNKNOWN"}


def evaluate_rule(rule_card: dict, case_facts: dict) -> RuleEvaluation:
    """기본 적용 조건 + 예외 조건 → 최종 라벨.

    입력 형태(우선순위):
    - rule_card["condition_tree"]: 명시적 중첩 트리 — 그대로 재귀 평가
    - rule_card["conditions"] + rule_card["condition_logic"](기본 AND): 평면 목록
    예외: rule_card["exception_conditions"](+ exception_logic, 기본 AND).
    카드에 구조화 예외가 없고 자연어 exception 텍스트만 있으면 예외 평가는
    수행하지 않는다(임의 구조화 금지) — 기본 조건 결과만으로 라벨을 낸다.
    """
    errors: list[str] = []
    flat: list[ConditionEvaluation] = []
    logic = str(rule_card.get("condition_logic") or "AND").upper()
    if rule_card.get("condition_tree"):
        base = evaluate_tree(rule_card["condition_tree"], case_facts, flat)
        logic = str(rule_card["condition_tree"].get("logic", logic)).upper()
    elif rule_card.get("conditions"):
        for c in rule_card["conditions"]:
            flat.append(evaluate_condition(c, case_facts))
        base = combine([e.status for e in flat], logic)
    else:
        base = EvaluationStatus.UNKNOWN
        errors.append("구조화 조건 없음")

    ex_flat: list[ConditionEvaluation] = []
    ex_status: EvaluationStatus | None = None
    if rule_card.get("exception_conditions"):
        ex_logic = str(rule_card.get("exception_logic") or "AND").upper()
        for c in rule_card["exception_conditions"]:
            ex_flat.append(evaluate_condition(c, case_facts))
        ex_status = combine([e.status for e in ex_flat], ex_logic)

    # 최종 라벨 (§6.1 매핑)
    if base == EvaluationStatus.NOT_SATISFIED:
        label = LABEL["not_applicable"]
    elif base != EvaluationStatus.SATISFIED:
        label = (LABEL["insufficient"] if base == EvaluationStatus.UNKNOWN and flat
                 else LABEL["unknown"])
    elif ex_status is None:
        label = LABEL["applicable"]
    elif ex_status == EvaluationStatus.SATISFIED:
        label = LABEL["exception"]
    elif ex_status == EvaluationStatus.NOT_SATISFIED:
        label = LABEL["applicable"]     # 예외 불성립(부분충족 포함) → 원칙 적용
    else:
        label = LABEL["insufficient"]   # 예외 판단 불가

    missing = sorted({e.field for e in flat + ex_flat
                      if e.status == EvaluationStatus.UNKNOWN
                      and e.reason.startswith("사례 값 없음")})
    unsupported = sorted({f"{e.field}:{e.reason}" for e in flat + ex_flat
                          if e.status == EvaluationStatus.UNKNOWN
                          and not e.reason.startswith("사례 값 없음")})
    return RuleEvaluation(status=base, condition_logic=logic, condition_results=flat,
                          exception_status=ex_status, exception_results=ex_flat,
                          derived_label=label, missing_fields=missing,
                          unsupported_conditions=unsupported, errors=errors)


def evaluation_to_dict(ev: RuleEvaluation) -> dict:
    """Master Dataset program_evaluation 필드 직렬화."""
    def cond(e: ConditionEvaluation) -> dict:
        return {"field": e.field, "operator": e.operator,
                "expected_value": e.expected_value, "actual_value": e.actual_value,
                "unit": e.unit, "status": e.status.value, "reason": e.reason}
    evaluable = bool(ev.condition_results) and ev.derived_label != "UNKNOWN"
    return {
        "evaluable": evaluable,
        "status": ev.status.value,
        "derived_label": ev.derived_label,
        "condition_logic": ev.condition_logic,
        "condition_results": [cond(e) for e in ev.condition_results],
        "exception_status": ev.exception_status.value if ev.exception_status else None,
        "exception_results": [cond(e) for e in ev.exception_results],
        "missing_fields": ev.missing_fields,
        "unsupported_conditions": ev.unsupported_conditions,
        "errors": ev.errors,
        "evaluator_version": EVALUATOR_VERSION,
    }
