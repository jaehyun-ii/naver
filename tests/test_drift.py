"""드리프트 탐지 테스트 — PSI/JS 기반 입력 분포 변화 (GPU·judge 불필요)."""

from __future__ import annotations

from llmops_core.monitoring import compute_drift, summarize

# 조선 도메인 기준선(짧은 한국어 QA류)
_REF = [
    "선급증서 유효기간이 지나면 재검사를 신청한다.",
    "공장인수시험 불합격 시 시정조치 요구서를 발행한다.",
    "재료증명서가 없으면 입고를 보류한다.",
    "도면 승인 의견서를 검토한다.",
    "선주 요구사항서를 확인한다.",
] * 6  # 30건


def test_same_distribution_no_drift():
    ref = summarize(_REF)
    rep = compute_drift(ref, _REF)  # 동일 분포
    assert rep.drift is False
    assert rep.scores["length_psi"] < 0.1
    assert rep.scores["vocab_js"] < 0.1


def test_shifted_distribution_detects_drift():
    ref = summarize(_REF)
    # 전혀 다른 도메인·언어·길이(영문 장문)
    drifted = [
        "This is a completely different English sentence about software engineering "
        "and distributed systems with much longer content than the reference set." for _ in range(30)
    ]
    rep = compute_drift(ref, drifted)
    assert rep.drift is True
    # 어휘 분포(언어 전환)와 문자비율이 크게 변함
    assert rep.scores["vocab_js"] > 0.3 or rep.scores["char_shift"] > 0.2


def test_length_drift_via_psi():
    ref = summarize(["짧은 문장." for _ in range(30)])
    long_texts = ["아주 긴 문장 " * 50 for _ in range(30)]  # 길이 분포 급변
    rep = compute_drift(ref, long_texts)
    assert rep.scores["length_psi"] > 0.25
    assert rep.drift is True


def test_summary_format():
    s = summarize(["가나다 abc 123"])
    assert s["n"] == 1
    assert len(s["length_hist"]) == 7  # bins + overflow
    assert set(s["char_ratios"]) == {"ko", "en", "digit", "other"}
    assert s["char_ratios"]["ko"] > 0 and s["char_ratios"]["en"] > 0


def test_thresholds_override():
    ref = summarize(_REF)
    # 매우 느슨한 임계값이면 약한 변화는 드리프트 아님
    rep = compute_drift(ref, _REF[:10], thresholds={"length_psi": 9.9, "vocab_js": 9.9,
                                                    "char_shift": 9.9})
    assert rep.drift is False
