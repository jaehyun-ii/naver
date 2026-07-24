"""RAG 서빙 제어평면 — 문서 수집·검색·증강. 런타임 검색 경로(임베더+벡터스토어).

수집(operator), 검색·증강·통계(viewer). 기본은 무서비스(인메모리+해시 임베더)로 동작하고,
설정(LLMOPS_RAG__EMBEDDER=bge-m3, BACKEND=qdrant)으로 실서비스 품질로 전환된다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.security import Principal
from llmops_core.console.security import require_perm
from llmops_core.rag import RagPipeline

router = APIRouter(prefix="/api/rag", tags=["rag"],
                   dependencies=[Depends(require_perm("read"))])

_pipeline: RagPipeline | None = None


def _rag() -> RagPipeline:
    """설정(rag.backend/embedder) 기반 서빙 파이프라인. qdrant/bge-m3 선택 시 해당
    선택 의존성이 필요하며, 없으면 503으로 명확히 안내한다(모듈 import는 항상 성공)."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RagPipeline()
        except OptionalDependencyError as exc:
            raise HTTPException(503, str(exc)) from exc
    return _pipeline


class IngestBody(BaseModel):
    texts: list[str]
    ids: list[str] | None = None
    metadata: list[dict] | None = None


class QueryBody(BaseModel):
    query: str
    k: int | None = None


@router.get("/stats")
def stats() -> dict:
    return _rag().stats()


@router.get("/collections")
def collections() -> dict:
    """벡터 DB(Qdrant) 컬렉션 인덱싱 현황 + 현재 서빙 검색 스택 설정.

    Qdrant 미가용이면 컬렉션 목록 대신 error를 담아 200으로 응답한다
    (지식베이스 화면이 dev(인메모리) 환경에서도 그대로 뜨도록).
    """
    from llmops_core.common.config import get_settings

    cfg = get_settings().rag
    out: dict = {
        "serving": {
            "backend": cfg.backend, "collection": cfg.collection,
            "embedder": cfg.embedder, "embedding_model": cfg.embedding_model,
            "query_prompt": cfg.query_prompt, "reranker_model": cfg.reranker_model,
            "top_k": cfg.top_k,
        },
        "collections": [],
    }
    try:
        from llmops_core.rag.retriever import qdrant_client

        cl = qdrant_client()
        for c in cl.get_collections().collections:
            info = cl.get_collection(c.name)
            vec = info.config.params.vectors
            dim = getattr(vec, "size", None)
            if dim is None and isinstance(vec, dict) and vec:
                dim = next(iter(vec.values())).size
            out["collections"].append({
                "name": c.name, "points": int(info.points_count or 0),
                "dim": dim, "active": c.name == cfg.collection,
            })
        out["collections"].sort(key=lambda x: (-x["active"], -x["points"]))
    except Exception as exc:  # noqa: BLE001 — 미가용 사유를 그대로 노출
        out["error"] = str(exc)
    return out


@router.post("/ingest")
def ingest(
    body: IngestBody, principal: Principal = Depends(require_perm("data:write")),
) -> dict:
    if not body.texts:
        raise HTTPException(422, "수집할 텍스트가 비었습니다")
    try:
        # qdrant 경로는 llama_index(청킹) 선택 의존성이 필요.
        n = _rag().ingest(body.texts, ids=body.ids, metadata=body.metadata)
    except OptionalDependencyError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"ingested": n, "total": _rag().store.count()}


@router.post("/query")
def query(body: QueryBody) -> dict:
    """검색만 — top-k 문서·점수 반환(생성 없음)."""
    hits = _rag().retrieve(body.query, body.k)
    return {"hits": [{"id": h.document.id, "text": h.document.text,
                      "score": round(h.score, 4), "metadata": h.document.metadata}
                     for h in hits]}


@router.post("/augment")
def augment(body: QueryBody) -> dict:
    """검색→컨텍스트 주입된 messages 반환(게이트웨이로 흘려보낼 형태)."""
    messages, hits = _rag().augment(body.query, k=body.k)
    return {"messages": messages,
            "used": [{"id": h.document.id, "score": round(h.score, 4)} for h in hits]}
