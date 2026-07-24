"""answer_format — SFT·DPO·GRPO 공유 답변 형식과 형식 보상."""
import pytest

from scripts.aireg_kr.answer_format import (UNCERTAIN_DECISION, format_reward,
                                            format_regulation_answer,
                                            format_score,
                                            parse_regulation_answer)


def test_full_answer_sections_in_order():
    text = format_regulation_answer(
        decision="적합", evidence=["[문서 2] 규칙 > 제5편", "인용: 「…하여야 한다」"],
        condition_review="총톤수 조건 충족", required_action="보조조타장치 비치",
        missing_information=None)
    assert text.index("판단:") < text.index("근거:") < text.index("조건 검토:") \
        < text.index("필요한 조치:")
    assert "추가 확인 정보" not in text  # 빈 섹션 강제 출력 금지


def test_uncertain_answer():
    text = format_regulation_answer(
        decision=UNCERTAIN_DECISION,
        evidence="제공된 규정은 유조선에만 적용되지만 사례에는 선종이 없습니다.",
        condition_review=["총톤수 조건: 충족", "선종 조건: 확인 불가"],
        missing_information=["선종"])
    parsed = parse_regulation_answer(text)
    assert parsed["sections"]["decision"] == UNCERTAIN_DECISION
    assert "- 선종" in parsed["sections"]["missing_information"]
    assert parsed["sections"]["condition_review"].startswith("- 총톤수")


def test_decision_required():
    with pytest.raises(ValueError):
        format_regulation_answer(decision="  ")


def test_parse_duplicates_and_order():
    text = "근거:\nB\n\n판단:\n적합\n\n판단:\n부적합"
    parsed = parse_regulation_answer(text)
    assert parsed["order"] == ["evidence", "decision"]   # 등장 순서 그대로
    assert parsed["duplicates"] == ["decision"]
    assert parsed["sections"]["decision"] == "적합"       # 첫 등장 유지


def test_format_score_good_answer_is_1():
    good = format_regulation_answer(decision="적합", evidence="인용: 「…」")
    assert format_score(good) == 1.0
    good_uncertain = format_regulation_answer(
        decision=UNCERTAIN_DECISION, evidence="근거 문장", missing_information=["선종"])
    assert format_score(good_uncertain) == 1.0


def test_format_score_penalties():
    assert format_score("그냥 자유 서술 답변") == 0.0          # 섹션 없음 → 형식 점수 없음
    assert format_score("판단:\n적합") == 0.8                 # 근거 누락 -0.2
    assert format_score("판단:\n판단 불가\n\n근거:\nX") == 0.8  # 판단불가인데 추가확인정보 없음
    assert format_score("근거:\nB\n\n판단:\n적합") < 1.0        # 순서 위반


def test_format_reward_trl_shapes():
    good = format_regulation_answer(decision="적합", evidence="e")
    # standard(str) / conversational([{role,content}]) 모두 처리
    out = format_reward(["p", "p"], [good, [{"role": "assistant", "content": good}]],
                        gold_label=["compliant", "compliant"])
    assert out == [1.0, 1.0]


def test_select_answer_format_routing():
    from scripts.aireg_kr.answer_format import (ANSWER_FORMAT_JUDGMENT,
                                                select_answer_format)
    for t in ("applicability", "compliance_judgment", "exception_judgment",
              "insufficient_information"):
        assert select_answer_format(t) == ANSWER_FORMAT_JUDGMENT
    for t in ("direct_qa", "comparison", "cross_reference_lookup", "summary"):
        assert select_answer_format(t) is None
    assert select_answer_format("모르는_태스크") is None            # non-strict: 자연어 유지
    with pytest.raises(ValueError):
        select_answer_format("모르는_태스크", strict=True)


def test_format_reward_gated_by_task_metadata():
    from scripts.aireg_kr.answer_format import ANSWER_FORMAT_JUDGMENT
    free = "자연어 서술 답변입니다. 판단 섹션 없음."
    judged = format_regulation_answer(decision="적합", evidence="e")
    # answer_format 메타 우선: 정보형(None)은 감점 없이 1.0(그룹 내 상수)
    out = format_reward(["p"] * 3, [free, free, judged],
                        answer_format=[None, ANSWER_FORMAT_JUDGMENT, ANSWER_FORMAT_JUDGMENT])
    assert out == [1.0, 0.0, 1.0]
    # task_type 폴백 게이팅
    out = format_reward(["p"] * 2, [free, free],
                        task_type=["direct_qa", "compliance_judgment"])
    assert out == [1.0, 0.0]
    # 메타 없음(prompt-only) — 종전대로 전 행 채점
    assert format_reward(["p"], [free]) == [0.0]


def test_suite_applic_chosen_rejected_same_formatter():
    """판정형 스위트 QA: chosen/rejected 외형 동일 — 형식만으로 선호 학습 방지."""
    from scripts.aireg_kr.answer_format import parse_regulation_answer as parse
    chosen = format_regulation_answer(decision="적용", evidence=["[문서 1] 경로"],
                                      condition_review="근거 검토")
    rejected = format_regulation_answer(decision="미적용",
                                        evidence="선박 개요를 볼 때 그렇게 판단됩니다.")
    for text in (chosen, rejected):
        parsed = parse(text)
        assert parsed["order"][0] == "decision" and "evidence" in parsed["sections"]
    assert format_score(chosen) == format_score(rejected) == 1.0  # 형식 점수 동일


def test_roundtrip_with_build_training():
    """build_chosen 산출물이 공통 파서로 완전 해석되는지 — 형식 계약의 실사용 검증."""
    from scripts.aireg_kr.build_training import build_chosen, build_no_golden_answer
    rule = {"article_text": "선박에는 1조의 주조타장치를 비치하여야 한다. 이 요건은 모든 선박에 적용한다.",
            "section_path": "규칙 > 제5편 > 201. 조타장치의 수"}
    ex = {"target_label": "uncertain"}
    ann = {"compliance_reason": "결정적 정보 부족", "missing_information": ["선종"],
           "critical_evidence": []}
    parsed = parse_regulation_answer(build_chosen(rule, ex, ann, golden_pos=2))
    assert parsed["sections"]["decision"] == UNCERTAIN_DECISION
    assert "[문서 2]" in parsed["sections"]["evidence"]
    assert format_score(build_chosen(rule, ex, ann, 2)) == 1.0
    assert format_score(build_no_golden_answer(rule)) == 1.0
