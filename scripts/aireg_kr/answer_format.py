"""규정 답변 공통 형식 — SFT 정답·DPO chosen/rejected·GRPO 형식 보상·평가 파서 공유.

형식이 생성부·보상부·평가부에서 각자 놀면 SFT가 가르친 출력과 GRPO가 보상하는
출력이 충돌한다. 섹션 라벨·순서는 여기서만 정의하고, 답변 조립은
``format_regulation_answer``, 해석은 ``parse_regulation_answer``, GRPO 형식 보상은
``format_reward`` 하나로 통일한다.

규칙: 섹션 순서 고정(판단 → 근거 → 조건 검토 → 필요한 조치 → 추가 확인 정보),
빈 섹션은 생략(강제 출력 금지), '판단'은 필수.
"""
from __future__ import annotations

import re

# 순서가 곧 출력 순서 — 변경 시 VERSIONS의 관련 프롬프트 버전도 올릴 것
ANSWER_SECTION_LABELS = {
    "decision": "판단",
    "evidence": "근거",
    "condition_review": "조건 검토",
    "required_action": "필요한 조치",
    "missing_information": "추가 확인 정보",
}
_ORDER = list(ANSWER_SECTION_LABELS)
_LABEL_TO_KEY = {v: k for k, v in ANSWER_SECTION_LABELS.items()}

UNCERTAIN_DECISION = "판단 불가"

# 표준 영문 라벨 ↔ 한국어 '판단' 섹션 출력 표현 — intent별로 다른 표현을 강제해
# 서로 다른 질문 축에 같은 "적용" 표현을 쓰지 않는다(§8).
LABEL_DISPLAY = {
    "APPLICABLE": "적용", "NOT_APPLICABLE": "비적용",
    "OBLIGATION_TRIGGERED": "의무 발생", "OBLIGATION_NOT_TRIGGERED": "의무 미발생",
    "PROHIBITED": "금지됨", "NOT_PROHIBITED": "금지되지 않음",
    "PERMITTED": "허용됨", "NOT_PERMITTED": "허용되지 않음",
    "EXCEPTION_APPLIES": "예외 적용", "EXCEPTION_NOT_APPLIES": "예외 미적용",
    "REQUIREMENT_SATISFIED": "요건 충족", "REQUIREMENT_NOT_SATISFIED": "요건 미충족",
    "COMPLIANT": "적합", "NON_COMPLIANT": "부적합",
    "INSUFFICIENT_INFORMATION": UNCERTAIN_DECISION,
}
# 관용 부정형 별칭 — "미적용"이 "적용"에 오매칭되지 않도록 명시 등록
_DISPLAY_ALIASES = {
    "미적용": "NOT_APPLICABLE", "적용되지 않음": "NOT_APPLICABLE",
    "적용 대상이 아님": "NOT_APPLICABLE", "적용됨": "APPLICABLE",
    "의무 없음": "OBLIGATION_NOT_TRIGGERED",
    "예외가 적용": "EXCEPTION_APPLIES",
}
# 표시 표현 → 표준 라벨 (긴 표현 우선 매칭 — "금지되지 않음"이 "금지됨"보다 먼저)
DISPLAY_TO_LABEL = sorted(
    ([(v, k) for k, v in LABEL_DISPLAY.items()] + list(_DISPLAY_ALIASES.items())),
    key=lambda x: -len(x[0]))


def display_label(label: str) -> str:
    """표준 라벨 → 판단 섹션 한국어 표현(미등록은 원문 유지)."""
    return LABEL_DISPLAY.get(label, label)


def parse_decision_to_label(decision_text: str) -> str | None:
    """판단 섹션 텍스트 → 표준 라벨. 정확 일치 우선, 이후 긴 표현 우선 부분 일치."""
    t = (decision_text or "").strip()
    for disp, label in DISPLAY_TO_LABEL:
        if t == disp:
            return label
    for disp, label in DISPLAY_TO_LABEL:
        if disp in t:
            return label
    return None

# ── 태스크별 형식 라우팅 ─────────────────────────────────────────────────
# 판정형은 공통 규정 답변 형식(판단/근거/…)을 강제하고, 정보형은 태스크별
# 자연어 형식을 유지한다. 집합은 여기서만 관리(생성부·보상부·평가부 공유).
ANSWER_FORMAT_JUDGMENT = "regulation_judgment_v1"

REGULATION_JUDGMENT_TASKS = frozenset({
    "applicability", "compliance_judgment", "exception_judgment",
    "condition_verification", "insufficient_information",
    "violation_judgment", "requirement_satisfaction",
})
INFORMATION_TASKS = frozenset({
    "definition", "summary", "direct_qa", "extractive_qa", "comparison",
    "procedure", "calculation", "requirement_extraction", "cross_reference_lookup",
})


def select_answer_format(task_type: str, strict: bool = False) -> str | None:
    """task_type → 답변 형식 id. 판정형이면 공통 형식, 정보형이면 None(자연어).

    미등록 task_type은 strict면 오류, 아니면 경고성 None(기존 자연어 유지).
    """
    if task_type in REGULATION_JUDGMENT_TASKS:
        return ANSWER_FORMAT_JUDGMENT
    if task_type in INFORMATION_TASKS:
        return None
    if strict:
        raise ValueError(f"미등록 task_type: {task_type!r}")
    return None


def _fmt_value(value) -> str:
    """str은 그대로, list는 '- ' 불릿 목록으로."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return "\n".join(f"- {str(v).strip()}" for v in value if str(v).strip())


def format_regulation_answer(
    *,
    decision: str,
    evidence: str | list[str] | None = None,
    condition_review: str | list[str] | None = None,
    required_action: str | list[str] | None = None,
    missing_information: str | list[str] | None = None,
) -> str:
    """섹션 순서 고정·빈 섹션 생략으로 규정 답변 텍스트를 조립한다. decision 필수."""
    if not (decision or "").strip():
        raise ValueError("decision(판단)은 비울 수 없습니다")
    values = {
        "decision": decision,
        "evidence": evidence,
        "condition_review": condition_review,
        "required_action": required_action,
        "missing_information": missing_information,
    }
    parts = []
    for key in _ORDER:
        body = _fmt_value(values[key])
        if body:
            parts.append(f"{ANSWER_SECTION_LABELS[key]}:\n{body}")
    return "\n\n".join(parts)


_SECTION_RE = re.compile(
    rf"^({'|'.join(re.escape(v) for v in ANSWER_SECTION_LABELS.values())})\s*[:：]\s*",
    re.MULTILINE)


def parse_regulation_answer(text: str) -> dict:
    """답변 텍스트 → {"sections": {key: body}, "order": [key…], "duplicates": [key…]}.

    평가 시 섹션 파싱과 GRPO 형식 보상이 같은 해석을 쓰도록 하는 단일 파서.
    중복 섹션은 첫 등장만 sections에 남기고 duplicates에 기록한다.
    """
    matches = list(_SECTION_RE.finditer(text or ""))
    sections: dict[str, str] = {}
    order: list[str] = []
    duplicates: list[str] = []
    for i, m in enumerate(matches):
        key = _LABEL_TO_KEY[m.group(1)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if key in sections:
            duplicates.append(key)
        else:
            sections[key] = body
            order.append(key)
    return {"sections": sections, "order": order, "duplicates": duplicates}


def _completion_text(completion) -> str:
    """TRL completions 원소 — str(standard) 또는 [{role,content}](conversational)."""
    if isinstance(completion, str):
        return completion
    if completion and isinstance(completion, list):
        return completion[-1].get("content", "") if isinstance(completion[-1], dict) else ""
    return ""


def format_score(text: str) -> float:
    """형식 준수 점수(0~1) — 의미 정확성 평가가 아니다.

    - 판단 섹션 존재·비어 있지 않음: 0.4
    - 섹션 등장 순서가 정의 순서와 일치: 0.2
    - 중복 섹션 없음: 0.2
    - 판단 불가 답변이면 '추가 확인 정보' 존재, 그 외면 '근거' 존재: 0.2
    """
    parsed = parse_regulation_answer(text)
    sections, order = parsed["sections"], parsed["order"]
    if not sections:
        return 0.0  # 섹션 자체가 없는 자유 서술 — 형식 점수 없음
    score = 0.0
    if sections.get("decision", "").strip():
        score += 0.4
    if order and [k for k in _ORDER if k in order] == order:
        score += 0.2
    if not parsed["duplicates"]:
        score += 0.2
    if UNCERTAIN_DECISION in sections.get("decision", ""):
        if sections.get("missing_information", "").strip():
            score += 0.2
    elif sections.get("evidence", "").strip():
        score += 0.2
    return round(score, 4)


def format_reward(prompts, completions, answer_format=None, task_type=None,
                  **kwargs) -> list[float]:
    """GRPO 형식 준수 보상 — run_grpo_peft의 reward_funcs에 그대로 주입 가능.

    형식 보상은 **판정형 태스크에만** 적용한다. 행 메타(answer_format 우선,
    없으면 task_type)로 판별하며, 정보형에는 1.0(그룹 내 상수 → GRPO advantage
    무영향)을 반환해 '판단:' 섹션 부재를 감점하지 않는다. 메타가 전혀 없는
    prompt-only 데이터는 종전대로 전 행 채점(하위 호환).

    형식 보상은 전체 보상의 일부로만 사용한다(라벨 정확성·근거 충실성 보상과 결합).
    """
    out = []
    for i, c in enumerate(completions):
        if answer_format is not None:
            judge = answer_format[i] == ANSWER_FORMAT_JUDGMENT
        elif task_type is not None:
            judge = task_type[i] in REGULATION_JUDGMENT_TASKS
        else:
            judge = True
        out.append(format_score(_completion_text(c)) if judge else 1.0)
    return out
