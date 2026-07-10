#!/usr/bin/env python3
"""RAG 부모 문맥 확장용 사이드카 스토어.

Qdrant엔 검색 단위(child·표·그림)만 색인되고 parent(조 전체)는 색인하지 않는다.
검색 후 문맥 확장(child → 소속 조 전체)을 위해 parent_chunk_id → 조 본문/메타를
가벼운 SQLite로 뽑아둔다. data_chunks에서 parent 청크만 읽어 생성.

    python scripts/build_parent_store.py
"""
from __future__ import annotations

import glob
import json
import sqlite3
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
CHUNKS = REPO / "data_chunks"
DB = CHUNKS / "rag_parents.db"


def main() -> int:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE parents(
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, article_no TEXT,
            article_title TEXT, section_path TEXT, content TEXT)
    """)
    n = 0
    for fn in glob.glob(str(CHUNKS / "**" / "*.jsonl"), recursive=True):
        rows = []
        for line in open(fn, encoding="utf-8"):
            c = json.loads(line)
            if c.get("chunk_level") != "parent":
                continue
            rows.append((
                c.get("chunk_id"), c.get("doc_id"), str(c.get("article_no") or ""),
                c.get("article_title") or "",
                " > ".join(c.get("section_path") or []),
                c.get("content") or ""))
        con.executemany("INSERT OR REPLACE INTO parents VALUES(?,?,?,?,?,?)", rows)
        n += len(rows)
    con.commit()
    con.close()
    print(f"✓ 부모 조 {n:,}개 → {DB} ({DB.stat().st_size / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
