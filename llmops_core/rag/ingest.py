"""문서 인제스트/청킹 — llama_index.core.node_parser.SentenceSplitter 임베드.

PDF/표 파싱(Docling/unstructured)→청킹→임베딩→테넌트 컬렉션 적재. 자체 파이프라인.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmops_core.common.errors import OptionalDependencyError
from llmops_core.rag.retriever import collection_for, qdrant_client


@dataclass
class IngestConfig:
    tenant_id: str
    chunk_size: int = 512
    chunk_overlap: int = 64
    doc_version: str = "v1"  # payload 필터용 문서 버전


def split_documents(texts: list[str], cfg: IngestConfig):
    """SentenceSplitter로 청킹. 각 노드에 테넌트/문서버전 메타데이터를 부여."""
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("llama-index-core", "rag") from exc

    splitter = SentenceSplitter(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    docs = [
        Document(text=t, metadata={"tenant": cfg.tenant_id, "doc_version": cfg.doc_version})
        for t in texts
    ]
    return splitter.get_nodes_from_documents(docs)


def ensure_collection(cfg: IngestConfig, vector_size: int) -> str:
    """테넌트 컬렉션 생성(없으면). dense+sparse 하이브리드 대비 스키마."""
    from qdrant_client.models import Distance, VectorParams

    client = qdrant_client()
    name = collection_for(cfg.tenant_id)
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    return name
