#!/usr/bin/env python3
"""Audit each society's chunk JSONL against the KR-derived chunking strategy:

  1. rule/guidance separation (where the family has it)
  2. 장/절/조 (chapter/section/article) hierarchy preservation
  3. semantic-unit splitting inside articles (not one giant chunk, not over-split)
  4. tables / figures as separate structures — and *coverage* vs the source
  5. rule↔guidance cross-linking
  6. path-based unique ids (no article_no collisions)
  7. chunk_type diversity (not everything = paragraph)
  8. definitions split per term; notes/exceptions separated
  9. references populated; page coverage vs source content_list

Reads data_chunks/*.jsonl and the matching data/<name>/..._content_list.json.
"""
from __future__ import annotations
import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
CHUNKS = REPO / "data_chunks"
DATA = REPO / "data"

# chunk-file stem -> source doc dir name
def source_dir(stem: str) -> str:
    return {"1__2025": "1__2025"}.get(stem, stem.replace("_chunks", "").upper()
                                       if False else stem[:-7] if stem.endswith("_chunks") else stem)


def find_source(name: str) -> Path | None:
    # name is the chunk stem without _chunks; the data dir usually matches case-folded
    for d in DATA.iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            cands = [p for p in d.rglob("*content_list.json")
                     if "raw" not in p.parts and not p.name.endswith("v2.json")]
            if cands:
                return sorted(cands, key=lambda p: (len(p.parts), len(p.name)))[0]
    return None


GENERIC = {"paragraph"}
STRUCT = {"table", "figure"}


def audit(path: Path) -> dict:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    parents = [c for c in rows if c["chunk_level"] == "parent"]
    children = [c for c in rows if c["chunk_level"] == "child"]
    types = Counter(c["chunk_type"] for c in rows)
    ids = [c["chunk_id"] for c in rows]

    # id uniqueness + article_no collision (why path-id matters)
    dup_ids = [i for i, n in Counter(ids).items() if n > 1]
    art_by_no: dict = {}
    for p in parents:
        art_by_no.setdefault(str(p.get("article_no")), set()).add(p["chunk_id"])
    collisions = {k: len(v) for k, v in art_by_no.items() if k not in ("None", "") and len(v) > 1}

    # semantic splitting
    kids_per_parent = []
    for p in parents:
        kids_per_parent.append(len([c for c in children
                                    if c.get("parent_chunk_id") == p["chunk_id"]
                                    and c["chunk_type"] not in STRUCT]))
    big_undersplit = [p["chunk_id"] for p in parents
                      if p.get("content_tokens", 0) > 900
                      and len([c for c in children if c.get("parent_chunk_id") == p["chunk_id"]
                               and c["chunk_type"] not in STRUCT]) <= 1]
    empty_children = [c["chunk_id"] for c in children
                      if c["chunk_type"] not in STRUCT and not (c.get("content") or "").strip()]

    # type diversity
    non_struct_children = [c for c in children if c["chunk_type"] not in STRUCT]
    pct_generic = (100 * sum(1 for c in non_struct_children if c["chunk_type"] in GENERIC)
                   / max(1, len(non_struct_children)))

    # rule/guidance
    doctypes = Counter(c.get("document_type") for c in parents)
    xlink = sum(1 for p in parents if p.get("linked_guidance_chunk_id") or p.get("linked_rule_chunk_id"))

    # references
    with_ref = sum(1 for c in rows if c.get("references"))

    # hierarchy completeness
    missing_hier = sum(1 for p in parents
                       if not (p.get("chapter_no") or p.get("chapter_title")) and p.get("article_no"))
    path_depths = [len(c.get("section_path") or []) for c in parents]

    # coverage vs source
    src = find_source(path.stem[:-7] if path.stem.endswith("_chunks") else path.stem)
    cov = {}
    if src:
        items = json.loads(src.read_text(encoding="utf-8"))
        src_tables = sum(1 for b in items if b.get("type") == "table")
        src_imgs = sum(1 for b in items if b.get("type") == "image")
        src_pages = {b.get("page_idx") for b in items if b.get("page_idx") is not None}
        chunk_pages = {p for c in rows for p in (c.get("pages") or [])}
        cov = {
            "src_tables": src_tables, "tab_chunks": types.get("table", 0),
            "src_imgs": src_imgs, "fig_chunks": types.get("figure", 0),
            "src_pages": len(src_pages),
            "pages_covered": len(src_pages & chunk_pages),
        }

    return {
        "file": path.name, "n": len(rows), "parent": len(parents), "child": len(children),
        "types": dict(types.most_common()),
        "dup_ids": len(dup_ids), "collisions": collisions,
        "kids_med": statistics.median(kids_per_parent) if kids_per_parent else 0,
        "kids_max": max(kids_per_parent) if kids_per_parent else 0,
        "undersplit": big_undersplit, "empty_children": len(empty_children),
        "pct_generic": round(pct_generic, 1),
        "doctypes": dict(doctypes), "xlink": xlink,
        "with_ref_pct": round(100 * with_ref / max(1, len(rows)), 1),
        "missing_hier": missing_hier,
        "path_depth_med": statistics.median(path_depths) if path_depths else 0,
        "cov": cov,
    }


def main():
    files = sorted(CHUNKS.glob("*_chunks.jsonl"))
    results = [audit(f) for f in files]
    for r in results:
        print(f"\n■ {r['file']}  ({r['n']} chunks: {r['parent']}P/{r['child']}C)")
        print(f"   types: {r['types']}")
        print(f"   doctypes: {r['doctypes']}  xlink={r['xlink']}")
        print(f"   ids: dup={r['dup_ids']}  article_no collisions handled={len(r['collisions'])}"
              + (f" (e.g. {list(r['collisions'].items())[:3]})" if r['collisions'] else ""))
        print(f"   split: kids/parent med={r['kids_med']} max={r['kids_max']}  "
              f"undersplit(>900tok,≤1child)={len(r['undersplit'])}  empty_children={r['empty_children']}")
        print(f"   generic(paragraph)%={r['pct_generic']}  refs%={r['with_ref_pct']}  "
              f"missing_hier={r['missing_hier']}  path_depth_med={r['path_depth_med']}")
        if r["cov"]:
            c = r["cov"]
            tc = "OK" if c["tab_chunks"] >= c["src_tables"] else f"MISS {c['src_tables']-c['tab_chunks']}"
            fc = "OK" if c["fig_chunks"] >= c["src_imgs"] else f"MISS {c['src_imgs']-c['fig_chunks']}"
            pc = "OK" if c["pages_covered"] >= 0.9 * c["src_pages"] else f"{c['pages_covered']}/{c['src_pages']}"
            print(f"   coverage: tables {c['tab_chunks']}/{c['src_tables']} [{tc}]  "
                  f"figures {c['fig_chunks']}/{c['src_imgs']} [{fc}]  pages {c['pages_covered']}/{c['src_pages']} [{pc}]")


if __name__ == "__main__":
    main()
