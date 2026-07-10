"""청크 JSONL → Qdrant 적재.

우리 청킹 전략대로 **child·표·그림(검색 단위)만 색인**하고 parent(조 전체)는
색인하지 않는다(문맥 확장용으로 payload의 parent_chunk_id로만 연결). 임베딩 입력은
section_path를 앞에 붙인 문맥화 텍스트(반복되는 조 번호 구분). 임베더는 rag의
것을 재사용(dev=해시 임베더 무의존, 운영=bge-m3).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

_NAMESPACE = uuid.UUID("1b671a64-40d5-491e-99b0-da01ff1f3341")

# 청커 publisher(표시명) → 선급 코드. 컬렉션 라우팅·필터 어휘는 코드를 쓴다(noksan_ax 동일).
_PUBLISHER_CODE = {
    "한국선급": "KR", "korean register": "KR",
    "classnk (nk)": "NK", "classnk": "NK", "nippon kaiji kyokai": "NK",
    "lloyd's register": "LR", "lloyd’s register": "LR",
    "dnv": "DNV",
    "bureau veritas (bv)": "BV", "bureau veritas": "BV",
    "iacs": "IACS",
    "abs": "ABS",
}


def _publisher_code(raw: str) -> str:
    return _PUBLISHER_CODE.get(str(raw or "").strip().lower(), str(raw or "").strip())


def _embed_text(row: dict) -> str:
    if row.get("chunk_type") in ("table", "figure"):
        return row.get("retrieval_text") or row.get("content") or ""
    path = " > ".join(row.get("section_path") or [])
    head = row.get("local_heading") or ""
    body = row.get("content") or ""
    # 호/목/세목 leaf는 상위 '항' lead-in이 본문에 없어 문맥을 잃는다("(가) 강재는…"만 남고
    # "재료는 다음에 적합하여야 한다:"가 소실). 항 제목을 주입해 문맥을 복원한다.
    # (중복 방지: 항 자신 child·이미 head/body에 담긴 경우 제외)
    para = row.get("paragraph_title") or ""
    add_para = (para and (row.get("item_no") or row.get("sub_item_no"))
                and para != head and not body.startswith(para))
    parts = (path, para, head, body) if add_para else (path, head, body)
    return "\n".join(p for p in parts if p)


def _payload(row: dict, order: int) -> dict:
    # table_html은 payload 제외(크기 — SQL/사이드카 전용). 좌표 3필드는 계층 질의용,
    # chunk_order는 형제 재조립 정렬 키(좌표 문자열 정렬 금지 — "10"<"2", ix<v 함정).
    return {
        "chunk_id": row.get("chunk_id"),
        "doc_id": row.get("doc_id"),
        "publisher": _publisher_code(row.get("publisher", "")),
        "document_type": row.get("document_type"),
        "chunk_type": row.get("chunk_type"),
        "chunk_order": order,
        "article_no": row.get("article_no"),
        "article_title": row.get("article_title"),
        "chapter_no": row.get("chapter_no"),
        "section_no": row.get("section_no"),
        "section_path": " > ".join(row.get("section_path") or []),
        "paragraph_no": row.get("paragraph_no") or "",
        "item_no": row.get("item_no") or "",
        "sub_item_no": row.get("sub_item_no") or "",
        "local_heading": row.get("local_heading") or "",
        "parent_chunk_id": row.get("parent_chunk_id"),
        "linked_rule_chunk_id": row.get("linked_rule_chunk_id"),
        "linked_guidance_chunk_id": row.get("linked_guidance_chunk_id"),
        "pages": row.get("pages") or [],
        "image_path": row.get("image_path") or row.get("img_path") or "",
        "text": row.get("content") or row.get("retrieval_text") or "",
    }


def vectorize_chunks(chunks_path: str | Path, collection: str, *,
                     qdrant_url: str = "http://localhost:6333", embedder=None) -> dict:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from llmops_core.rag.embedder import make_embedder

    rows = [json.loads(l) for l in Path(chunks_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    # child + table + figure — 원문 순서(chunk_order)를 정렬 키로 보존.
    # 빈 임베딩 텍스트(캡션·HTML 없는 이미지형 표 등)는 드랍(noksan_ax 가드 동일).
    units, texts = [], []
    for order, r in enumerate(rows):
        if r.get("chunk_level") != "child":
            continue
        text = _embed_text(r)
        if not text.strip():
            continue
        units.append((r, order))
        texts.append(text)
    if not units:
        return {"collection": collection, "n_vectors": 0, "indexed_of": len(rows)}

    embedder = embedder or make_embedder()
    vecs = embedder.encode(texts)

    client = QdrantClient(url=qdrant_url)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
        )
    # 재적재 안전: 이 문서(doc_id)의 기존 포인트 삭제 후 upsert(멱등)
    doc_id = units[0][0].get("doc_id")
    if doc_id:
        from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue
        client.delete(collection_name=collection, points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])))

    points = [
        PointStruct(id=str(uuid.uuid5(_NAMESPACE, r["chunk_id"])), vector=list(map(float, v)),
                    payload=_payload(r, order))
        for (r, order), v in zip(units, vecs)
    ]
    for i in range(0, len(points), 256):
        client.upsert(collection_name=collection, points=points[i:i + 256])
    return {"collection": collection, "n_vectors": len(points), "indexed_of": len(rows), "dim": embedder.dim}
