"""RAG 검색 — llama_index.core retriever/postprocessor + qdrant_client 임베드.

LlamaIndex 프레임워크 통째가 아닌 핵심 노드만 사용. 하이브리드 질의·테넌트 컬렉션
격리·payload 필터는 자체 코드가 보유. Qdrant는 Service(코어는 qdrant_client만).
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.rag.expansion import DEFAULT_BUDGET_CHARS, expand_context


@dataclass
class RetrievalConfig:
    tenant_id: str
    top_k: int = 20           # 원본 벡터 후보 수(오버페치) — 형제 near-dup을 조로 붕괴시킬 여유
    rerank_top_n: int = 3     # 최종 반환 수 = 조 단위 dedup **후** 서로 다른 조 개수
    use_hybrid: bool = True   # dense + sparse
    expand_parent: bool = True  # child 히트 → 소속 조(parent) 전체 문맥으로 확장


# ── 부모(조) 문맥 스토어 — child만 색인되므로 조 본문은 사이드카에서 조회 ──────────
_PARENT_DB_CANDIDATES = [
    os.environ.get("LLMOPS_RAG_PARENT_DB"),
    "/home/jaehyun/Dev/naver/data_chunks/rag_parents.db",
    str(Path(__file__).resolve().parents[2] / "data_chunks" / "rag_parents.db"),
]
_parent_con: sqlite3.Connection | None = None


def _parent_con_get() -> sqlite3.Connection | None:
    global _parent_con
    if _parent_con is None:
        for c in _PARENT_DB_CANDIDATES:
            if c and Path(c).exists():
                _parent_con = sqlite3.connect(f"file:{c}?mode=ro", uri=True,
                                              check_same_thread=False)
                _parent_con.row_factory = sqlite3.Row
                break
    return _parent_con


def _parent_row(chunk_id: str):
    con = _parent_con_get()
    if con is None or not chunk_id:
        return None
    return con.execute(
        "SELECT article_no, article_title, section_path, content FROM parents "
        "WHERE chunk_id=?", (chunk_id,)).fetchone()


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
        """질의 → 컨텍스트 노드 리스트. payload_filter로 문서버전/테넌트 추가 필터.

        expand_parent=True면 정밀 매칭된 child를 **소속 조 전체**로 확장하고(중복 조 제거),
        메타데이터의 section_path/조 제목을 머리말로 붙여 문맥을 완성한다.

        순서 주의: 한 조의 형제 child((가)(나)(a)(b) …)는 임베딩이 거의 같아 top-k에
        나란히 올라온다. **조 단위 dedup을 자르기보다 먼저** 수행해 형제를 조 하나로
        붕괴시킨 뒤 rerank_top_n을 세야, top-N이 서로 다른 조로 채워진다(오버페치 top_k와 결합).
        """
        retriever = self.as_retriever(embed_model)
        nodes = retriever.retrieve(query)
        if payload_filter:
            nodes = [
                n
                for n in nodes
                if all(n.metadata.get(k) == v for k, v in payload_filter.items())
            ]
        if self.cfg.expand_parent:
            nodes = self._expand_parents(nodes)   # ① 조 단위 dedup(형제 붕괴) — 자르기 前
        return nodes[: self.cfg.rerank_top_n]      # ② 그 다음 서로 다른 조 N개

    def _expand_parents(self, nodes):
        """child 노드를 소속 조(parent) 문맥으로 확장. 같은 조는 한 번만.

        확장 정책 A(예산 하강): 조 전체가 예산(4k자) 안이면 조로, 넘으면
        같은 조 형제를 Qdrant에서 로드해 항→호 단위로 하강 재조립(expansion 참조).
        """
        try:
            from llama_index.core.schema import NodeWithScore, TextNode
        except ImportError:  # pragma: no cover
            return nodes
        out, seen = [], set()
        for n in nodes:
            meta = dict(n.metadata or {})
            pid = meta.get("parent_chunk_id")
            row = _parent_row(pid) if pid else None
            if row is None:
                out.append(n)                     # 부모 없음(표/그림 등) → 원본 유지
                continue
            if pid in seen:
                continue                          # 같은 조 중복 제거
            seen.add(pid)
            head = " > ".join(p for p in (row["section_path"],
                              f"{row['article_no']} {row['article_title']}".strip()) if p)
            if len(row["content"] or "") <= DEFAULT_BUDGET_CHARS:
                body, level = row["content"], "article"
            else:                                 # 병리적 거대 조(부록 등) — 항/호로 하강
                body, level = expand_context(
                    {**meta, "content": n.get_content() if hasattr(n, "get_content") else meta.get("text", "")},
                    {"content": row["content"]},
                    self._siblings(pid),
                )
            text = f"[{head}]\n{body}" if head else body
            meta.update(expanded=level, matched_chunk_id=meta.get("chunk_id"),
                        article_no=row["article_no"], article_title=row["article_title"])
            out.append(NodeWithScore(node=TextNode(text=text, metadata=meta),
                                     score=getattr(n, "score", None)))
        return out

    def _siblings(self, parent_chunk_id: str) -> list[dict]:
        """같은 조의 child payload 로드(예산 하강 재조립용 — 거대 조에서만 호출)."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            got, _ = self.client.scroll(
                collection_name=self.collection, limit=512, with_payload=True,
                scroll_filter=Filter(must=[FieldCondition(
                    key="parent_chunk_id", match=MatchValue(value=parent_chunk_id))]),
            )
            return [p.payload for p in got if p.payload]
        except Exception:  # noqa: BLE001 — 실패 시 leaf 폴백(expand_context가 처리)
            return []
