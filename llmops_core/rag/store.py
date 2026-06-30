"""벡터 스토어 — 코사인 검색. 인메모리(디스크 영속) 또는 Qdrant.

임베딩은 정규화돼 들어온다고 가정 → 코사인 = 내적. 소규모는 인메모리로 충분하고,
규모가 커지면 backend=qdrant(QdrantSettings)로 교체(인터페이스 동일).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError


@dataclass
class Document:
    id: str
    text: str
    metadata: dict | None = None


@dataclass
class Hit:
    document: Document
    score: float


class InMemoryVectorStore:
    """정규화 임베딩 내적(=코사인) 검색. persist_path 지정 시 디스크에 영속."""

    def __init__(self, persist_path: str | None = None) -> None:
        self._docs: list[Document] = []
        self._vecs: list[list[float]] = []
        self._path = Path(persist_path) if persist_path else None
        if self._path and self._path.exists():
            self._load()

    def add(self, docs: list[Document], vecs: list[list[float]]) -> None:
        if len(docs) != len(vecs):
            raise ValueError("docs와 vecs 길이 불일치")
        self._docs.extend(docs)
        self._vecs.extend(vecs)
        if self._path:
            self._save()

    def search(self, query_vec: list[float], k: int = 4) -> list[Hit]:
        scored = (
            Hit(doc, sum(a * b for a, b in zip(query_vec, v)))
            for doc, v in zip(self._docs, self._vecs)
        )
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    def count(self) -> int:
        return len(self._docs)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "docs": [{"id": d.id, "text": d.text, "metadata": d.metadata} for d in self._docs],
            "vecs": self._vecs,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _load(self) -> None:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        self._docs = [Document(**d) for d in payload.get("docs", [])]
        self._vecs = payload.get("vecs", [])


class QdrantVectorStore:
    """Qdrant 백엔드(lazy). 대규모·다중인스턴스용. 인터페이스는 인메모리와 동일."""

    def __init__(self, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("qdrant-client", "rag") from exc
        cfg = get_settings().qdrant
        self._client = QdrantClient(url=cfg.url, api_key=cfg.api_key)
        self._collection = collection
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            self._client.create_collection(
                collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, docs: list[Document], vecs: list[list[float]]) -> None:  # pragma: no cover
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=i, vector=v, payload={"id": d.id, "text": d.text,
                                                 "metadata": d.metadata})
            for i, (d, v) in enumerate(zip(docs, vecs), start=self.count())
        ]
        self._client.upsert(self._collection, points)

    def search(self, query_vec: list[float], k: int = 4) -> list[Hit]:  # pragma: no cover
        res = self._client.search(self._collection, query_vector=query_vec, limit=k)
        return [
            Hit(Document(id=p.payload["id"], text=p.payload["text"],
                         metadata=p.payload.get("metadata")), float(p.score))
            for p in res
        ]

    def count(self) -> int:  # pragma: no cover
        return int(self._client.count(self._collection).count)


def make_vector_store(dim: int):
    """설정(rag.backend)에 따라 벡터스토어 생성."""
    cfg = get_settings().rag
    if cfg.backend == "qdrant":
        return QdrantVectorStore(cfg.collection, dim)
    return InMemoryVectorStore(cfg.persist_path)
