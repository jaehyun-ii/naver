"""라벨 정규화·3자 비교·학습 포함 정책 — build_master/build_cases/GRPO 공용.

세 라벨의 출처:
  A generation_label  — profile kind/mutation 계획이 의도한 라벨
  B verifier_label    — 블라인드 검증기가 원문·사례를 보고 독립 산출한 라벨
  C program_label     — condition_evaluator가 조건×facts로 계산한 라벨

비교는 공통 라벨 공간(적용성)으로 정규화해 수행한다. program이 UNKNOWN이면
기존 생성+블라인드 검증 정책을 그대로 유지한다(평가기 불가 ≠ 데이터 불량).
"""
from __future__ import annotations

# 판정 트랙(compliance)·적용성 트랙 라벨 → 공통 공간
CANONICAL_LABELS = ("APPLICABLE", "NOT_APPLICABLE", "EXCEPTION_APPLIES",
                    "INSUFFICIENT_INFORMATION", "UNKNOWN")
_CANON = {
    "applicable": "APPLICABLE", "compliant": "APPLICABLE", "적용": "APPLICABLE",
    "적합": "APPLICABLE",
    "not_applicable": "NOT_APPLICABLE", "non_compliant": "NOT_APPLICABLE",
    "미적용": "NOT_APPLICABLE", "부적합": "NOT_APPLICABLE",
    "exception_applies": "EXCEPTION_APPLIES", "예외 적용": "EXCEPTION_APPLIES",
    "insufficient_information": "INSUFFICIENT_INFORMATION",
    "uncertain": "INSUFFICIENT_INFORMATION", "판단 불가": "INSUFFICIENT_INFORMATION",
    "unknown": "UNKNOWN",
}


def canonicalize_label(label: str | None) -> str:
    if not label:
        return "UNKNOWN"
    key = str(label).strip()
    return _CANON.get(key) or _CANON.get(key.lower(), "UNKNOWN")


def compare_labels(generation_label: str | None, verifier_label: str | None,
                   program_label: str | None) -> dict:
    """3자 비교 → {generation/verifier/program_label, agreement, action}.

    program=UNKNOWN이면 PROGRAM_UNEVALUABLE — 기존(생성+검증) 정책 유지.
    """
    g, v, p = (canonicalize_label(generation_label),
               canonicalize_label(verifier_label), canonicalize_label(program_label))
    out = {"generation_label": g, "verifier_label": v, "program_label": p}
    if p == "UNKNOWN":
        agreement = "PROGRAM_UNEVALUABLE"
        action = "KEEP_EXISTING_POLICY"
    elif g == v == p:
        agreement, action = "THREE_WAY_MATCH", "ACCEPT"
    elif v == p:
        agreement, action = "GENERATOR_DISAGREES", "REJECT"       # 생성 오류 추정
    elif g == p:
        agreement, action = "VERIFIER_DISAGREES", "REVERIFY"      # 검증 재시도/검수
    elif g == v:
        agreement, action = "PROGRAM_DISAGREES", "INSPECT_EVALUATOR"  # 평가기/카드 점검
    else:
        agreement, action = "ALL_DISAGREE", "EXPERT_REVIEW_HIGH"
    return out | {"agreement": agreement, "action": action}


def compare_primary_labels(*, question_intent: str,
                           generation_primary_label: str | None,
                           verifier_primary_label: str | None,
                           program_primary_label: str | None,
                           verifier_interpreted_intent: str | None = None) -> dict:
    """primary_label 축 3자 비교(§9) — 다른 의미 축의 라벨을 직접 비교하지 않는다.

    검증기가 해석한 intent가 입력 intent와 다르면 라벨 비교 대신
    SEMANTIC_AXIS_MISMATCH로 분류(재검증/검수 대상).
    """
    out = {"question_intent": question_intent,
           "generation_primary_label": generation_primary_label,
           "verifier_primary_label": verifier_primary_label,
           "program_primary_label": program_primary_label}
    if verifier_interpreted_intent and verifier_interpreted_intent != question_intent:
        return out | {"agreement": "SEMANTIC_AXIS_MISMATCH", "action": "REVERIFY",
                      "verifier_interpreted_intent": verifier_interpreted_intent}
    g, v, p = generation_primary_label, verifier_primary_label, program_primary_label
    if not p or p in ("UNKNOWN", "NOT_ASSESSED"):
        agreement, action = "PROGRAM_UNEVALUABLE", "KEEP_EXISTING_POLICY"
    elif g == v == p:
        agreement, action = "THREE_WAY_MATCH", "ACCEPT"
    elif v == p:
        agreement, action = "GENERATOR_DISAGREES", "REJECT"
    elif g == p:
        agreement, action = "VERIFIER_DISAGREES", "REVERIFY"
    elif g == v:
        agreement, action = "PROGRAM_DISAGREES", "INSPECT_EVALUATOR"
    else:
        agreement, action = "ALL_DISAGREE", "EXPERT_REVIEW_HIGH"
    return out | {"agreement": agreement, "action": action}


LABEL_POLICIES = ("strict", "verifier-preferred", "report-only")


def should_include_for_training(*, verifier_verdict: str | None,
                                label_consistency: dict | None,
                                label_policy: str) -> tuple[bool, str]:
    """학습 파생 포함 여부 결정. (include, reason) 반환.

    - 모든 정책 공통: 블라인드 검증 REJECT는 제외(기존 동작 보존).
    - strict: program 평가 가능인데 verifier≠program이면 제외.
    - verifier-preferred: generator만 불일치(REJECT 액션)도 제외.
    - report-only: verdict 외 추가 제외 없음(불일치는 리포트만).
    """
    if verifier_verdict == "REJECT":
        return False, "verifier_rejected"
    lc = label_consistency or {}
    agreement = lc.get("agreement")
    if agreement == "VERIFIER_PENDING":
        # 블라인드 검증 미수행 케이스 — strict에선 검증 전 학습 제외
        if label_policy == "strict":
            return False, "verifier_pending"
        return True, "verifier_pending_non_strict"
    if agreement == "SEMANTIC_AXIS_MISMATCH":
        # 질문 의미 축 불일치 — strict에선 재검증 전 학습 제외
        if label_policy == "strict":
            return False, "semantic_axis_mismatch"
        return True, "semantic_axis_mismatch_non_strict"
    if agreement in ("VERIFIER_ABSTENTION_CONFLICT",
                     "FACT_REALIZATION_OR_EXTRACTION_CONFLICT"):
        if label_policy == "strict":
            return False, agreement.lower()
        return True, f"{agreement.lower()}_non_strict"
    if agreement in (None, "PROGRAM_UNEVALUABLE"):
        return True, "program_unevaluable_keep_existing"
    if label_policy == "report-only":
        return True, "report_only"
    if agreement == "GENERATOR_DISAGREES":
        return False, "generator_label_mismatch"
    if label_policy == "strict":
        v = lc.get("verifier_primary_label", lc.get("verifier_label"))
        p = lc.get("program_primary_label", lc.get("program_label"))
        if v != p:
            return False, "program_verifier_label_mismatch"
    if agreement == "ALL_DISAGREE":
        return False, "all_labels_disagree"
    return True, "accepted"
