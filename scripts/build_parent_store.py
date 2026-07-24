#!/usr/bin/env python3
"""RAG 부모 문맥 확장용 사이드카 스토어.

Qdrant엔 검색 단위(child·표·그림)만 색인되고 parent(조 전체)는 색인하지 않는다.
검색 후 문맥 확장(child → 소속 조 전체)을 위해 parent_chunk_id → 조 본문/메타를
가벼운 SQLite로 뽑아둔다. 청크 디렉토리에서 parent 청크만 읽어 생성.

    python scripts/build_parent_store.py [--src data_chunks]
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
CHUNKS = REPO / "data_chunks"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=CHUNKS)
    args = ap.parse_args()
    CHUNKS_DIR = args.src.resolve()
    DB = CHUNKS_DIR / "rag_parents.db"
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE parents(
            chunk_id TEXT PRIMARY KEY, doc_id TEXT, article_no TEXT,
            article_title TEXT, section_path TEXT, content TEXT)
    """)
    n = n_tab = 0
    for fn in glob.glob(str(CHUNKS_DIR / "**" / "*.jsonl"), recursive=True):
        parents, tables = [], {}
        for line in open(fn, encoding="utf-8"):
            c = json.loads(line)
            if c.get("chunk_type") == "table":
                tables[c["chunk_id"]] = c
            elif c.get("chunk_level") == "parent":
                parents.append(c)
        rows = []
        for c in parents:
            content = c.get("content") or ""
            # 링크된 표 병합 — parent 본문에는 "표 N에 따른다" 참조만 있고 값은
            # 표 청크에만 있다. 서빙 문맥에 표 값이 없으면(실측: 18만 조 전수 0건)
            # 표 기반 판정을 학습한 모델이 근거를 잃는다. 조당 12k자 상한.
            tids = list(dict.fromkeys((c.get("linked_table_chunk_ids") or [])
                                      + (c.get("linked_tables") or [])))
            blocks, total = [], 0
            for tid in tids:
                t = tables.get(tid)
                if not t:
                    continue
                cap = (t.get("content") or "").strip()
                html = (t.get("table_html") or "").strip()
                if not html:
                    continue
                block = f"{cap}\n{html}" if cap and cap not in html else html
                if total + len(block) > 12000:
                    blocks.append("(이하 표 생략 — 분량 제한)")
                    break
                blocks.append(block)
                total += len(block)
            if blocks:
                content = content.rstrip() + "\n\n[인용된 표]\n" + "\n\n".join(blocks)
                n_tab += 1
            rows.append((
                c.get("chunk_id"), c.get("doc_id"), str(c.get("article_no") or ""),
                c.get("article_title") or "",
                " > ".join(c.get("section_path") or []),
                content))
        con.executemany("INSERT OR REPLACE INTO parents VALUES(?,?,?,?,?,?)", rows)
        n += len(rows)
    con.commit()
    con.close()
    print(f"✓ 부모 조 {n:,}개(표 병합 {n_tab:,}개) → {DB} "
          f"({DB.stat().st_size / 1048576:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
