"""RAG — 런타임 서빙 경로(임베더+벡터스토어+증강) + 헤비 경로(llama_index+Qdrant 하이브리드).

- 서빙(무서비스 가능): make_embedder / RagPipeline / InMemoryVectorStore — bge-m3 또는 해시 폴백.
- 헤비(엔터프라이즈): IngestConfig / TenantRetriever — llama_index 청킹 + Qdrant 테넌트 격리.
"""

from llmops_core.rag.embedder import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    make_embedder,
)
from llmops_core.rag.ingest import IngestConfig, ensure_collection, split_documents
from llmops_core.rag.pipeline import RagPipeline, compose_system, context_block
from llmops_core.rag.retriever import RetrievalConfig, TenantRetriever, collection_for
from llmops_core.rag.store import (
    Document,
    Hit,
    InMemoryVectorStore,
    make_vector_store,
)

__all__ = [
    # 런타임 서빙 경로
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "make_embedder",
    "RagPipeline",
    "compose_system",
    "context_block",
    "Document",
    "Hit",
    "InMemoryVectorStore",
    "make_vector_store",
    # 헤비 경로(llama_index + Qdrant)
    "IngestConfig",
    "split_documents",
    "ensure_collection",
    "RetrievalConfig",
    "TenantRetriever",
    "collection_for",
]
