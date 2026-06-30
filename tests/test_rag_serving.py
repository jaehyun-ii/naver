"""RAG 런타임 서빙 — 임베더(해시 폴백)·인메모리 벡터스토어·파이프라인 증강."""

from __future__ import annotations

from llmops_core.rag.embedder import HashingEmbedder
from llmops_core.rag.pipeline import RagPipeline
from llmops_core.rag.store import InMemoryVectorStore


def _pipeline(tmp_path=None):
    persist = str(tmp_path / "store.json") if tmp_path else None
    return RagPipeline(embedder=HashingEmbedder(dim=128),
                       store=InMemoryVectorStore(persist_path=persist))


def test_ingest_and_retrieve_relevant():
    rag = _pipeline()
    rag.ingest([
        "조선소 안전관리 절차는 선급기관 검사를 포함한다",
        "오늘 점심 메뉴는 김치찌개입니다",
        "선박 도크 진수 전 재검사 신청이 필요하다",
    ])
    hits = rag.retrieve("선급기관 재검사 신청", k=2)
    assert len(hits) == 2
    # 관련 문서가 무관 문서(점심 메뉴)보다 상위
    texts = [h.document.text for h in hits]
    assert any("선급" in t or "재검사" in t for t in texts)
    assert "점심" not in texts[0]


def test_augment_injects_context_messages():
    rag = _pipeline()
    rag.ingest(["선급기관에 재검사를 신청하세요"])
    messages, hits = rag.augment("재검사 절차는?", k=1)
    assert messages[0]["role"] == "system"
    assert "재검사" in messages[0]["content"]  # 컨텍스트 주입됨
    assert messages[1] == {"role": "user", "content": "재검사 절차는?"}
    assert len(hits) == 1


def test_persistence_roundtrip(tmp_path):
    rag = _pipeline(tmp_path)
    rag.ingest(["영속 테스트 문서"])
    # 동일 경로로 새 파이프라인 → 디스크에서 복원
    rag2 = _pipeline(tmp_path)
    assert rag2.store.count() == 1
    assert "영속" in rag2.retrieve("영속", k=1)[0].document.text


def test_hashing_embedder_deterministic_normalized():
    emb = HashingEmbedder(dim=64)
    a = emb.encode(["같은 문장"])[0]
    b = emb.encode(["같은 문장"])[0]
    assert a == b  # 결정적
    norm = sum(x * x for x in a) ** 0.5
    assert abs(norm - 1.0) < 1e-6  # L2 정규화


def test_stats():
    rag = _pipeline()
    rag.ingest(["a", "b"])
    s = rag.stats()
    assert s["documents"] == 2
    assert s["dim"] == 128


# ── 평가에 RAG 주입(--rag, 평가=서빙 정합) ──
def test_compose_system_combines_prompt_and_context():
    from llmops_core.evaluation.local_eval import _compose_system

    assert _compose_system(None, None) is None
    assert _compose_system("PROD", None) == "PROD"
    only_ctx = _compose_system(None, "문서A")
    assert "문서A" in only_ctx and "컨텍스트" in only_ctx
    both = _compose_system("PROD", "문서A")
    assert both.startswith("PROD")
    assert "문서A" in both  # prod 프롬프트 + RAG 컨텍스트 합성


def test_build_system_fn_injects_retrieved_context():
    from llmops_core.evaluation.local_eval import _build_system_fn

    rag = _pipeline()
    rag.ingest([
        "선급기관에 재검사를 신청해야 한다",
        "점심은 김치찌개",
    ])
    fn = _build_system_fn("기본지시", rag, top_k=1)
    sys_for_q = fn("재검사 절차?")
    assert sys_for_q.startswith("기본지시")
    assert "재검사" in sys_for_q  # 질문에 맞는 문서가 컨텍스트로 주입됨


def test_build_system_fn_without_rag_is_static():
    from llmops_core.evaluation.local_eval import _build_system_fn

    fn = _build_system_fn("기본지시", None, top_k=4)
    assert fn("아무질문") == "기본지시"
    assert fn("다른질문") == "기본지시"


# ── 서빙 RAG 주입(inject_context) + 평가=서빙 정합 ──
def test_inject_context_creates_system_when_absent():
    rag = _pipeline()
    rag.ingest(["선급기관 재검사 절차"])
    msgs = [{"role": "user", "content": "재검사 어떻게?"}]
    new, hits = rag.inject_context(msgs, k=1)
    assert new[0]["role"] == "system"
    assert "재검사" in new[0]["content"]
    assert new[-1] == {"role": "user", "content": "재검사 어떻게?"}
    assert len(hits) == 1


def test_inject_context_appends_to_existing_system():
    rag = _pipeline()
    rag.ingest(["도크 진수 전 재검사"])
    msgs = [{"role": "system", "content": "PROD지시"},
            {"role": "user", "content": "재검사?"}]
    new, _ = rag.inject_context(msgs, k=1)
    assert new[0]["content"].startswith("PROD지시")  # 기존 system 보존
    assert "재검사" in new[0]["content"]  # 컨텍스트 덧붙임


def test_inject_context_empty_kb_is_noop():
    rag = _pipeline()  # 비어 있음
    msgs = [{"role": "user", "content": "질문"}]
    new, hits = rag.inject_context(msgs, k=4)
    assert new == msgs
    assert hits == []


def test_eval_and_serving_share_compose():
    """평가(_compose_system)와 서빙(rag.compose_system)이 동일 출력(정합 보증)."""
    from llmops_core.evaluation.local_eval import _compose_system
    from llmops_core.rag import compose_system

    assert _compose_system("P", "C") == compose_system("P", "C")
