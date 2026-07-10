#!/usr/bin/env python3
"""청크 리뷰용 SQLite 인덱스 빌더.

data_chunks/<SOC>/*.jsonl 을 SQLite(data_chunks/review.db)로 적재한다. 콘솔 백엔드
(review 라우터)가 이 DB를 조 목록/조 본문 단위로 쿼리해, 대용량 문서(예: KR 통합본
49k청크)도 필요한 조각만 ms 단위로 내려준다.

문서 idx는 build_review_viewer.py와 동일 규칙(선급 내 title 정렬)으로 매긴다.
"""
from __future__ import annotations
import json
import sqlite3
import unicodedata
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
SRC = REPO / "sample_20"
CHUNKS = REPO / "data_chunks"
DB = CHUNKS / "review.db"

PREFIX = {"ABS": "abs_guide", "BV": "bv_rule", "ClassNK": "nk_rule",
          "DNV": "dnv_cg", "IACS": "iacs", "KR": "kr_rule", "LR": "lr_code"}
SOC_LABEL = {"ABS": "ABS (미국)", "BV": "BV (프랑스)", "ClassNK": "ClassNK (일본)",
             "DNV": "DNV (노르웨이)", "IACS": "IACS", "KR": "KR (대한민국)", "LR": "LR (영국)"}
SOC_ORDER = ["KR", "ClassNK", "DNV", "BV", "ABS", "IACS", "LR"]


def find_md(stem: str) -> str:
    for p in SRC.rglob(f"{stem}.md"):
        return str(p)
    return ""


def main() -> int:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript("""
        CREATE TABLE docs(
            soc TEXT, idx INTEGER, title TEXT, label TEXT, family TEXT,
            n_chunks INTEGER, n_articles INTEGER, md_path TEXT,
            PRIMARY KEY(soc, idx));
        CREATE TABLE chunks(
            soc TEXT, idx INTEGER, seq INTEGER,
            chunk_id TEXT, parent_chunk_id TEXT, chunk_level TEXT, chunk_type TEXT,
            article_no TEXT, article_title TEXT, chapter_no TEXT, chapter_title TEXT,
            section_path TEXT, paragraph_no TEXT, item_no TEXT, sub_item_no TEXT,
            content TEXT, content_tokens INTEGER,
            table_html TEXT, table_caption TEXT, table_nrows INTEGER,
            caption TEXT, image_path TEXT);
        CREATE INDEX ix_parent ON chunks(soc, idx, parent_chunk_id);
        CREATE INDEX ix_doclevel ON chunks(soc, idx, chunk_level);
    """)

    total_docs = total_chunks = 0
    for soc in SOC_ORDER:
        family = PREFIX[soc]
        # 선급 폴더의 문서들을 title(=stem) 정렬 — 뷰어 idx와 일치
        files = sorted(CHUNKS.glob(f"{soc}/*_chunks.jsonl"),
                       key=lambda p: unicodedata.normalize("NFC", p.name))
        for idx, ck in enumerate(files):
            stem = ck.name.replace("_chunks.jsonl", "")
            title = unicodedata.normalize("NFC", stem)
            rows = [json.loads(l) for l in ck.open(encoding="utf-8")]
            n_art = sum(1 for c in rows if c.get("chunk_level") == "parent")
            con.execute(
                "INSERT INTO docs VALUES(?,?,?,?,?,?,?,?)",
                (soc, idx, title, SOC_LABEL[soc], family, len(rows), n_art, find_md(stem)))
            con.executemany(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(soc, idx, seq,
                  c.get("chunk_id"), c.get("parent_chunk_id"),
                  c.get("chunk_level"), c.get("chunk_type"),
                  str(c.get("article_no") or ""), c.get("article_title") or "",
                  str(c.get("chapter_no") or ""), c.get("chapter_title") or "",
                  json.dumps(c.get("section_path") or [], ensure_ascii=False),
                  str(c.get("paragraph_no") or ""), str(c.get("item_no") or ""),
                  str(c.get("sub_item_no") or ""),
                  c.get("content") or "", c.get("content_tokens") or 0,
                  c.get("table_html") or "", c.get("table_caption") or "",
                  c.get("table_nrows") or 0,
                  c.get("caption") or "", c.get("image_path") or "")
                 for seq, c in enumerate(rows)])
            total_docs += 1
            total_chunks += len(rows)
    con.commit()
    con.close()
    print(f"✓ {total_docs} docs · {total_chunks} chunks → {DB}")
    print(f"  크기: {DB.stat().st_size / 1048576:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
