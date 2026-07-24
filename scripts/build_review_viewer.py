#!/usr/bin/env python3
"""Assemble a local side-by-side review site: source .md (left) ↔ chunks (right),
selectable by society and document, over the whole sample_20 corpus.

Copies each doc's .md + chunks.jsonl into viewer/data/<SOC>/<idx>/ (index-based
paths dodge NFD/space URL issues) and writes viewer/data/manifest.json. Serve
with:  python -m http.server -d viewer 8080
"""
from __future__ import annotations
import json, os, re, shutil, unicodedata
from collections import Counter
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
SRC = REPO / "sample_20"
CHUNKS = REPO / "data_chunks"
VIEW = REPO / "viewer"
DATA = VIEW / "data"

PREFIX = {"ABS": "abs_guide", "BV": "bv_rule", "ClassNK": "nk_rule",
          "DNV": "dnv_cg", "IACS": "iacs", "KR": "kr_rule", "LR": "lr_code"}
# 뷰어가 실제 렌더/검색에 쓰는 필드만. 레벨별로 분리해 중복 metadata를 제거:
#  - parent(조): 헤딩·경로·검색용 필드
#  - child(항/호/표/그림): 렌더에 필요한 최소 필드만(section_path·doc_title 등 제외)
KEEP_PARENT = {"chunk_id", "parent_chunk_id", "chunk_level", "chunk_type", "content",
               "content_tokens", "article_no", "article_title", "doc_title", "section_path"}
KEEP_CHILD = {"chunk_id", "parent_chunk_id", "chunk_level", "chunk_type", "content",
              "paragraph_no", "item_no", "caption", "image_path",
              "table_caption", "table_html", "table_nrows"}
SOC_LABEL = {"ABS": "ABS (미국)", "BV": "BV (프랑스)", "ClassNK": "ClassNK (일본)",
             "DNV": "DNV (노르웨이)", "IACS": "IACS", "KR": "KR (대한민국)", "LR": "LR (영국)"}


def main() -> int:
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)

    cls = sorted(p for p in SRC.rglob("*content_list.json") if not p.name.endswith("v2.json"))
    manifest: dict[str, dict] = {}
    total = 0
    for cl in cls:
        soc = unicodedata.normalize("NFC", cl.relative_to(SRC).parts[0]).split("_")[0]
        family = PREFIX.get(soc)
        if not family:
            continue
        stem = cl.name.replace("_content_list.json", "")
        title = unicodedata.normalize("NFC", stem)
        md = cl.with_name(stem + ".md")
        ck = CHUNKS / soc / f"{stem}_chunks.jsonl"
        if not ck.exists():
            continue

        man = manifest.setdefault(soc, {"label": SOC_LABEL.get(soc, soc),
                                        "family": family, "docs": []})
        idx = len(man["docs"])
        dst = DATA / soc / str(idx)
        dst.mkdir(parents=True)
        # md (may be missing → empty)
        shutil.copy2(md, dst / "source.md") if md.exists() else (dst / "source.md").write_text(
            "*(원본 .md 없음)*", encoding="utf-8")

        # 뷰어가 실제로 쓰는 필드만 남긴 슬림 청크로 저장(대용량 문서 로딩 가속).
        # parent.content(전체 조문)는 검색용으로만 쓰이므로 앞부분만 유지.
        rows = [json.loads(l) for l in ck.open(encoding="utf-8")]
        with (dst / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in rows:
                is_parent = c.get("chunk_level") == "parent"
                keep = KEEP_PARENT if is_parent else KEEP_CHILD
                slim = {k: c[k] for k in keep if k in c}
                if is_parent and slim.get("content"):
                    slim["content"] = slim["content"][:800]     # 검색 스니펫만
                fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
        ty = Counter(c["chunk_type"] for c in rows)
        n_art = sum(1 for c in rows if c["chunk_level"] == "parent")
        man["docs"].append({
            "idx": idx, "title": title, "n_chunks": len(rows), "n_articles": n_art,
            "types": dict(ty.most_common()),
        })
        total += 1

    # society order fixed; docs sorted by title
    ordered = {}
    for soc in ["KR", "ClassNK", "DNV", "BV", "ABS", "IACS", "LR"]:
        if soc in manifest:
            manifest[soc]["docs"].sort(key=lambda d: d["title"])
            # reindex after sort so idx matches folder — keep folder idx via a map
            ordered[soc] = manifest[soc]
    (DATA / "manifest.json").write_text(
        json.dumps(ordered, ensure_ascii=False), encoding="utf-8")
    print(f"✓ {total} docs → {DATA}")
    print(f"  serve:  python -m http.server -d {VIEW} 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
