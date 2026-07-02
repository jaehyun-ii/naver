"""RAG 파이프라인 — 수집(임베딩·색인)·검색·컨텍스트 주입.

서빙 시점 흐름: 질의 → 임베딩 → 벡터검색(top-k) → 컨텍스트 조립 → 메시지에 주입.
augment()가 OpenAI 호환 messages를 돌려주므로 게이트웨이/체인에 그대로 흘려보낼 수 있다.
"""

from __future__ import annotations

from llmops_core.common.config import get_settings
from llmops_core.rag.embedder import Embedder, make_embedder
from llmops_core.rag.store import Document, Hit, make_vector_store

_DEFAULT_SYSTEM = (
    "다음 컨텍스트만 근거로 한국어로 답하세요. 컨텍스트에 없으면 모른다고 답하세요.\n\n"
    "[컨텍스트]\n{context}"
)
_CONTEXT_INSTRUCTION = "다음 컨텍스트를 참고해 답하세요."

# bge-m3 코사인 유사도 하한(무관 문서 컷). 해시 폴백(dev)은 0.0=미적용으로 둔다.
_DEFAULT_SCORE_THRESHOLD = 0.3


def context_block(context: str) -> str:
    return f"{_CONTEXT_INSTRUCTION}\n\n[컨텍스트]\n{context}"


def compose_system(base_system: str | None, context: str | None) -> str | None:
    """기존 시스템 프롬프트에 RAG 컨텍스트를 합성. 평가·서빙이 공유하는 단일 규약.

    base_system(예: prod 프롬프트)을 덮어쓰지 않고 컨텍스트 블록을 덧붙인다.
    """
    if not context:
        return base_system
    block = context_block(context)
    return f"{base_system}\n\n{block}" if base_system else block


class RagPipeline:
    """임베더 + 벡터스토어 묶음. 수집/검색/메시지 증강 제공."""

    def __init__(self, embedder: Embedder | None = None, store=None, *,
                 score_threshold: float | None = None) -> None:
        self.embedder = embedder or make_embedder()
        self.store = store or make_vector_store(self.embedder.dim)
        cfg = get_settings().rag
        self.top_k = cfg.top_k
        # 서빙(bge-m3)은 유사도 하한 적용, dev 해시 폴백은 미적용(결정적 베이스라인 보존).
        if score_threshold is not None:
            self.score_threshold = score_threshold
        elif cfg.embedder == "hashing":
            self.score_threshold = 0.0
        else:
            self.score_threshold = _DEFAULT_SCORE_THRESHOLD

    def ingest(self, texts: list[str], *, ids: list[str] | None = None,
               metadata: list[dict] | None = None) -> int:
        """텍스트들을 임베딩·색인. 색인된 (청크) 문서 수 반환.

        backend=qdrant면 헤비 경로(split_documents)로 청킹 후 durable 적재하고,
        dev 폴백(memory)은 원문을 그대로 색인한다.
        """
        if not texts:
            return 0
        if get_settings().rag.backend == "qdrant":
            return self._ingest_chunked(texts)
        ids = ids or [f"doc-{self.store.count() + i}" for i in range(len(texts))]
        metadata = metadata or [{} for _ in texts]
        docs = [Document(id=i, text=t, metadata=m)
                for i, t, m in zip(ids, texts, metadata)]
        vecs = self.embedder.encode(texts)
        self.store.add(docs, vecs)
        return len(docs)

    def _ingest_chunked(self, texts: list[str]) -> int:  # pragma: no cover
        """헤비 경로 청킹(SentenceSplitter) → 임베딩 → Qdrant durable 적재."""
        from llmops_core.rag.ingest import IngestConfig, split_documents

        cfg = get_settings().rag
        nodes = split_documents(texts, IngestConfig(tenant_id=cfg.collection))
        base = self.store.count()
        docs: list[Document] = []
        chunk_texts: list[str] = []
        for i, node in enumerate(nodes):
            text = node.get_content() if hasattr(node, "get_content") else node.text
            docs.append(Document(id=f"{cfg.collection}-{base + i}", text=text,
                                 metadata=dict(getattr(node, "metadata", {}) or {})))
            chunk_texts.append(text)
        self.store.add(docs, self.embedder.encode(chunk_texts))
        return len(docs)

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        """질의에 가장 가까운 top-k 문서(유사도 하한 이상만)."""
        qv = self.embedder.encode([query])[0]
        hits = self.store.search(qv, k or self.top_k)
        if self.score_threshold > 0.0:
            hits = [h for h in hits if h.score >= self.score_threshold]
        return hits

    def build_context(self, hits: list[Hit]) -> str:
        return "\n\n".join(f"- {h.document.text}" for h in hits)

    def augment(self, query: str, *, k: int | None = None,
                system_template: str | None = None) -> tuple[list[dict], list[Hit]]:
        """질의를 RAG 컨텍스트로 증강한 messages와 사용된 hits 반환."""
        hits = self.retrieve(query, k)
        context = self.build_context(hits)
        system = (system_template or _DEFAULT_SYSTEM).format(context=context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        return messages, hits

    def inject_context(self, messages: list[dict], *, k: int | None = None,
                       ) -> tuple[list[dict], list[Hit]]:
        """기존 messages(서빙 요청)에 RAG 컨텍스트를 주입한 새 messages 반환.

        마지막 user 메시지를 질의로 검색 → 컨텍스트를 system 메시지에 합성(기존 system이
        있으면 덧붙이고, 없으면 맨 앞에 생성). compose_system 규약을 평가와 공유한다.
        """
        query = next((m.get("content", "") for m in reversed(messages)
                      if m.get("role") == "user"), "")
        if not query:
            return list(messages), []
        hits = self.retrieve(query, k)
        context = self.build_context(hits)
        if not context:
            return list(messages), hits
        new = [dict(m) for m in messages]
        sys_idx = next((i for i, m in enumerate(new) if m.get("role") == "system"), None)
        if sys_idx is None:
            new.insert(0, {"role": "system", "content": compose_system(None, context)})
        else:
            new[sys_idx]["content"] = compose_system(new[sys_idx].get("content"), context)
        return new, hits

    def stats(self) -> dict:
        cfg = get_settings().rag
        return {
            "documents": self.store.count(),
            "embedder": cfg.embedder,
            "dim": self.embedder.dim,
            "backend": cfg.backend,
            "collection": cfg.collection,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
        }
