"""상류 데이터 파이프라인(L1~L3) 단위 테스트 — 무거운 의존성 없이 실행 가능.

정규화/중복제거(L1), 품질 게이트(L2 네이티브 규칙), 포맷변환/결정적분할/핑거프린트(L3).
"""

from __future__ import annotations

import pytest

from llmops_core.common.errors import DataQualityFailed
from llmops_core.common.schemas import TextRecord
from llmops_core.dataset.build import (
    SplitConfig,
    filter_sft_by_length,
    split_sft,
    to_sft_examples,
)
from llmops_core.dataset.versioning import fingerprint
from llmops_core.ingestion.records import dumps_jsonl, loads_jsonl
from llmops_core.ingestion.text_ops import content_hash, exact_dedup, normalize_text
from llmops_core.quality import QualityRules, validate_records


# ── L1 정규화/중복제거 ──
def test_normalize_and_exact_dedup():
    assert normalize_text("  hello   world \n") == "hello world"
    # 정규화 후 동일 → 같은 해시
    assert content_hash("hello   world") == content_hash("hello world")

    recs = [
        TextRecord(id="1", text="같은  문장"),
        TextRecord(id="2", text="같은 문장"),  # 정규화하면 중복
        TextRecord(id="3", text="다른 문장"),
    ]
    kept, removed = exact_dedup(recs)
    assert removed == 1
    assert [r.id for r in kept] == ["1", "3"]


# ── L1 JSONL io 라운드트립 ──
def test_jsonl_roundtrip():
    recs = [TextRecord(id="1", text="a"), TextRecord(id="2", text="b", lang="ko")]
    back = loads_jsonl(dumps_jsonl(recs), TextRecord)
    assert [r.id for r in back] == ["1", "2"]
    assert back[1].lang == "ko"


# ── L2 품질 게이트 ──
def test_quality_gate_pass():
    recs = [TextRecord(id=str(i), text=f"유효한 문장 번호 {i} 입니다") for i in range(10)]
    report = validate_records(recs)
    assert report.passed
    assert report.num_records == 10


def test_quality_gate_fail_on_dup_ratio():
    # 전부 동일 → 중복비율 1.0 > 기본 0.2 → 게이트 실패
    recs = [TextRecord(id=str(i), text="동일한 문장 동일한 문장") for i in range(5)]
    with pytest.raises(DataQualityFailed):
        validate_records(recs, QualityRules(min_chars=1))


def test_quality_gate_fail_on_min_chars():
    recs = [TextRecord(id="1", text="짧음")]
    report = validate_records(recs, QualityRules(min_chars=100), raise_on_fail=False)
    assert not report.passed
    assert "min_chars" in report.failures


# ── L3 포맷 변환·필터 ──
def test_to_sft_and_filter():
    labeled = [
        {"text": "질문1", "response": "답변1"},
        {"text": "질문2", "response": ""},  # 빈 응답 → 변환 제외
        {"text": "질문3"},  # 응답 없음 → 제외
    ]
    examples = to_sft_examples(labeled, system="너는 도우미다")
    assert len(examples) == 1
    assert examples[0].messages[0].role == "system"
    assert examples[0].messages[-1].content == "답변1"

    long_ans = [{"text": "q", "response": "x" * 5}]
    assert filter_sft_by_length(to_sft_examples(long_ans), min_chars=10) == []


# ── L3 결정적 분할 ──
def test_stable_split_is_deterministic_and_covers_all():
    labeled = [{"text": f"q{i}", "response": f"a{i}"} for i in range(200)]
    examples = to_sft_examples(labeled)
    cfg = SplitConfig(val_ratio=0.1, test_ratio=0.1)
    a = split_sft(examples, cfg)
    b = split_sft(examples, cfg)
    # 같은 입력 → 같은 분할(재현성)
    assert [[e.messages for e in s] for s in a] == [[e.messages for e in s] for s in b]
    tr, va, te = a
    assert len(tr) + len(va) + len(te) == 200
    # 대략적 비율(해시 분포)
    assert len(te) > 0 and len(va) > 0 and len(tr) > len(va)


# ── L3 핑거프린트(순서 무관 안정성) ──
def test_fingerprint_order_independent():
    r1 = [TextRecord(id="1", text="a"), TextRecord(id="2", text="b")]
    r2 = [TextRecord(id="2", text="b"), TextRecord(id="1", text="a")]
    assert fingerprint(r1) == fingerprint(r2)
    r3 = [TextRecord(id="1", text="a"), TextRecord(id="2", text="c")]
    assert fingerprint(r1) != fingerprint(r3)
