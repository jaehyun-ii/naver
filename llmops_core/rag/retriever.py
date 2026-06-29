"""RAG 검색 — llama_index.core retriever/postprocessor + qdrant_client 임베드.

LlamaIndex 프레임워크 통째가 아닌 핵심 노드만 사용. 하이브리드 질의·테넌트 컬렉션
격리·payload 필터는 자체 코드가 보유. Qdrant는 Service(코어는 qdrant_client만).
"""

from __future__ import annotations

from dataclasses import dataclass

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError


@dataclass
class RetrievalConfig:
    tenant_id: str
    top_k: int = 5
    rerank_top_n: int = 3
    use_hybrid: bool = True  # dense + sparse


def collection_for(tenant_id: str) -> str:
    """테넌트별 컬렉션 격리 규칙."""
    s = get_settings()
    return f"{s.env}-{s.domain}-{tenant_id}"


def qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("qdrant-client", "rag") from exc
    s = get_settings().qdrant
    return QdrantClient(url=s.url, api_key=s.api_key)


class TenantRetriever:
    """테넌트 컬렉션 + payload 필터 기반 하이브리드 검색기."""

    def __init__(self, cfg: RetrievalConfig):
        self.cfg = cfg
        self.client = qdrant_client()
        self.collection = collection_for(cfg.tenant_id)

    def as_retriever(self, embed_model):
        """llama_index VectorIndexRetriever로 래핑 (리랭킹 postprocessor 별도 적용)."""
        try:
            from llama_index.core import VectorStoreIndex
            from llama_index.vector_stores.qdrant import QdrantVectorStore
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("llama-index-core", "rag") from exc

        store = QdrantVectorStore(client=self.client, collection_name=self.collection)
        index = VectorStoreIndex.from_vector_store(store, embed_model=embed_model)
        return index.as_retriever(
            similarity_top_k=self.cfg.top_k,
            sparse_top_k=self.cfg.top_k if self.cfg.use_hybrid else None,
        )

    def retrieve(self, query: str, embed_model, *, payload_filter: dict | None = None):
        """질의 → 컨텍스트 노드 리스트. payload_filter로 문서버전/테넌트 추가 필터."""
        retriever = self.as_retriever(embed_model)
        nodes = retriever.retrieve(query)
        if payload_filter:
            nodes = [
                n
                for n in nodes
                if all(n.metadata.get(k) == v for k, v in payload_filter.items())
            ]
        return nodes[: self.cfg.rerank_top_n]
