"""청크 리뷰 — 선급 문서의 원문(.md) ↔ 청킹 결과를 **조 단위로 지연 조회**.

data_chunks/review.db(SQLite)를 쿼리해, 대용량 문서(예: KR 통합본 49k청크)도
조 목록은 가볍게, 조 본문은 펼칠 때만 내려준다(정적 파일 통째 전송 대비 수십 배 빠름).

인덱스 빌드:  python scripts/build_review_db.py
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from llmops_core.console.security import require_perm

router = APIRouter(prefix="/api/review", tags=["review"],
                   dependencies=[Depends(require_perm("read"))])

# DB 위치 후보 — 컨테이너에선 코드가 /app/llmops_core, 데이터는 host repo 마운트 경로라
# 서로 다르므로 여러 후보를 시도한다(env > host repo 경로 > 코드 상대 경로).
_DB_CANDIDATES = [
    os.environ.get("LLMOPS_REVIEW_DB"),
    "/home/jaehyun/Dev/naver/data_chunks/review.db",
    str(Path(__file__).resolve().parents[3] / "data_chunks" / "review.db"),
]


def _db() -> Path:
    for c in _DB_CANDIDATES:
        if c and Path(c).exists():
            return Path(c)
    return Path(_DB_CANDIDATES[1])

SOC_ORDER = ["KR", "ClassNK", "DNV", "BV", "ABS", "IACS", "LR"]


def _con() -> sqlite3.Connection:
    db = _db()
    if not db.exists():
        raise HTTPException(503, "review.db 없음 — `python scripts/build_review_db.py` 실행 필요")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


@router.get("/docs")
def list_docs():
    """선급별 문서 목록(가벼움) — 좌측 셀렉터용."""
    con = _con()
    try:
        rows = con.execute(
            "SELECT soc,idx,title,label,family,n_chunks,n_articles FROM docs "
            "ORDER BY soc,idx").fetchall()
    finally:
        con.close()
    by_soc: dict = {}
    for r in rows:
        s = by_soc.setdefault(r["soc"], {"soc": r["soc"], "label": r["label"], "docs": []})
        s["docs"].append({"idx": r["idx"], "title": r["title"], "family": r["family"],
                          "n_chunks": r["n_chunks"], "n_articles": r["n_articles"]})
    return [by_soc[s] for s in SOC_ORDER if s in by_soc]


@router.get("/{soc}/{idx}/articles")
def list_articles(soc: str, idx: int):
    """한 문서의 조(parent) 목록 — 본문은 뺀 헤딩·경로·단위수만."""
    con = _con()
    try:
        parents = con.execute(
            "SELECT chunk_id,article_no,article_title,chapter_no,chapter_title,"
            "section_path,content_tokens FROM chunks "
            "WHERE soc=? AND idx=? AND chunk_level='parent' ORDER BY seq",
            (soc, idx)).fetchall()
        counts = dict(con.execute(
            "SELECT parent_chunk_id, COUNT(*) FROM chunks "
            "WHERE soc=? AND idx=? AND chunk_level='child' AND chunk_type='text' "
            "GROUP BY parent_chunk_id", (soc, idx)).fetchall())
    finally:
        con.close()
    if not parents:
        raise HTTPException(404, "문서 없음 또는 조 없음")
    return [{
        "chunk_id": p["chunk_id"], "article_no": p["article_no"],
        "article_title": p["article_title"],
        "chapter_no": p["chapter_no"], "chapter_title": p["chapter_title"],
        "section_path": json.loads(p["section_path"] or "[]"),
        "content_tokens": p["content_tokens"],
        "n_units": counts.get(p["chunk_id"], 0),
    } for p in parents]


@router.get("/{soc}/{idx}/article/{chunk_id}")
def article_children(soc: str, idx: int, chunk_id: str):
    """한 조의 자식(항/호/표/그림) — 펼칠 때만 요청."""
    con = _con()
    try:
        kids = con.execute(
            "SELECT chunk_id,chunk_type,paragraph_no,item_no,sub_item_no,content,"
            "table_html,table_caption,table_nrows,caption,image_path FROM chunks "
            "WHERE soc=? AND idx=? AND parent_chunk_id=? ORDER BY seq",
            (soc, idx, chunk_id)).fetchall()
    finally:
        con.close()
    return [dict(k) for k in kids]


@router.get("/{soc}/{idx}/asset")
def review_asset(soc: str, idx: int, path: str) -> FileResponse:
    """그림 원본 — figure 청크의 image_path(images/…)를 ETL 산출 디렉터리에서 서빙.

    이미지 파일은 원문 .md와 같은 디렉터리의 images/에 있다(md_path 기준 해석)."""
    con = _con()
    try:
        row = con.execute("SELECT md_path FROM docs WHERE soc=? AND idx=?", (soc, idx)).fetchone()
    finally:
        con.close()
    if row is None or not row["md_path"]:
        raise HTTPException(404, "문서 없음")
    rel = path.replace("\\", "/")
    if not rel.startswith("images/") or ".." in rel:
        raise HTTPException(422, "images/ 하위 경로만 허용됩니다")
    base = Path(row["md_path"]).resolve().parent
    f = (base / rel).resolve()
    if not str(f).startswith(str(base)) or not f.is_file():
        raise HTTPException(404, f"이미지 없음: {rel}")
    return FileResponse(f)


# 원문 블록 분리 — 프론트와 **동일 규칙**(빈 줄 OR 번호/문자 항목 시작 줄)
BLOCK_SPLIT = re.compile(
    r"\n{2,}|\n(?=\s*(?:\d{1,3}[.)]|\(\d{1,3}\)|[가-힣][.)]|\([가-힣]\)|[a-z][.)]|\([a-z]\)|[-●•▪])\s)")


def _mnorm(s: str) -> str:
    """매칭용 정규화 — 수식($$..$$)·이미지·마크다운 기호 제거 후 공백정리."""
    s = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", "", s or "", flags=re.S)
    s = re.sub(r"!\[.*?\]\([^)]*\)", "", s)
    return re.sub(r"\s+", " ", re.sub(r"[#*_>`|●•▪◦·–—]", " ", s)).strip().lower()


_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from llmops_core.rag.embedder import SentenceTransformerEmbedder
        _EMBEDDER = SentenceTransformerEmbedder("BAAI/bge-m3")
    return _EMBEDDER


@router.get("/search")
def search(q: str = "", k: int = 8, collection: str = "classification_rules"):
    """의미 검색(bge-m3) → Qdrant → **소속 조 전체로 문맥 확장**(중복 조 제거).

    임베더/Qdrant 미가용(경량 컨테이너)이면 503. 호스트 백엔드에서 동작한다.
    """
    q = (q or "").strip()
    if len(q) < 2:
        return {"hits": []}
    try:
        from qdrant_client import QdrantClient

        from llmops_core.rag.retriever import _parent_row
        emb = _embedder()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"검색 엔진 미가용(bge-m3/qdrant 필요): {e}")
    url = os.environ.get("LLMOPS_QDRANT__URL", "http://localhost:6333")
    cl = QdrantClient(url=url)
    vec = emb.encode([q])[0]
    pts = cl.query_points(collection, query=vec, limit=k, with_payload=True).points
    seen: set = set()
    hits: list = []
    for p in pts:
        pl = p.payload or {}
        pid = pl.get("parent_chunk_id")
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        row = _parent_row(pid) if pid else None
        hits.append({
            "score": round(float(p.score), 3),
            "section_path": pl.get("section_path", ""),
            "matched": (pl.get("text") or "")[:400],
            "article_no": row["article_no"] if row else "",
            "article_title": row["article_title"] if row else "",
            "article_content": (row["content"] if row else (pl.get("text") or ""))[:2500],
        })
    return {"hits": hits}


@router.get("/{soc}/{idx}/map")
def doc_map(soc: str, idx: int):
    """청크 유닛 ↔ 원문 블록 정렬 맵. 청크와 md가 **같은 원본 순서**임을 이용해
    커서 기반 순차 정렬 → 반복·짧은헤딩·수식이 낀 경우도 올바른 위치를 잡는다.
    반환: {units: {unit_id: [blockStart, blockEnd, article_id]}, n_blocks}."""
    con = _con()
    try:
        row = con.execute("SELECT md_path FROM docs WHERE soc=? AND idx=?",
                          (soc, idx)).fetchone()
        md = ""
        if row and row["md_path"] and Path(row["md_path"]).exists():
            md = Path(row["md_path"]).read_text(encoding="utf-8")
        raw = [t.strip() for t in BLOCK_SPLIT.split(md) if t and t.strip()]
        units = con.execute(
            "SELECT chunk_id, parent_chunk_id, content FROM chunks WHERE soc=? AND idx=? "
            "AND chunk_level='child' AND chunk_type='text' ORDER BY seq",
            (soc, idx)).fetchall()
    finally:
        con.close()

    # norm 블록만 concat + 각 구간 → raw 블록 인덱스
    starts: list[int] = []
    rindex: list[int] = []
    parts: list[str] = []
    pos = 0
    for ri, t in enumerate(raw):
        n = _mnorm(t)
        if not n:
            continue
        starts.append(pos)
        rindex.append(ri)
        parts.append(n)
        pos += len(n) + 1
    concat = " ".join(parts)

    import bisect

    def rblock(c: int) -> int:
        i = bisect.bisect_right(starts, c) - 1
        return rindex[i] if 0 <= i < len(rindex) else 0

    umap: dict[str, list] = {}
    cur = 0
    for u in units:
        uM = _mnorm(u["content"])
        if not uM:
            continue
        p = concat.find(uM, cur)
        if p < 0 and len(uM) >= 8:
            p = concat.find(uM[:36], cur)
        if p < 0 and len(uM) >= 8:
            p = concat.find(uM[:24])
        if p < 0:
            p = concat.find(uM[:max(5, min(20, len(uM)))])
        if p < 0:
            continue
        end = p + len(uM)
        cur = end
        umap[u["chunk_id"]] = [rblock(p), rblock(max(p, end - 1)), u["parent_chunk_id"]]
    return {"units": umap, "n_blocks": len(raw)}


@router.get("/{soc}/{idx}/md", response_class=PlainTextResponse)
def source_md(soc: str, idx: int):
    """원문 markdown(좌측 패널)."""
    con = _con()
    try:
        row = con.execute("SELECT md_path FROM docs WHERE soc=? AND idx=?",
                          (soc, idx)).fetchone()
    finally:
        con.close()
    if not row:
        raise HTTPException(404, "문서 없음")
    p = Path(row["md_path"]) if row["md_path"] else None
    return p.read_text(encoding="utf-8") if p and p.exists() else "*(원본 .md 없음)*"
