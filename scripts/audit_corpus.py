#!/usr/bin/env python3
"""Aggregate quality audit over the chunked corpus (data_chunks/<society>/*.jsonl).

Per society: doc/chunk counts, integrity (dup ids, empty children, degenerate
parents), split health, chunk_type diversity, and the improvement signals
(exception/note/table_row). Flags the worst outlier docs for inspection.
"""
from __future__ import annotations
import json, glob, os, statistics
from collections import Counter, defaultdict
from pathlib import Path

CH = Path("/home/jaehyun/Dev/naver/data_chunks")
STRUCT = {"table", "figure", "table_row"}

def audit_doc(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    ids = [c["chunk_id"] for c in rows]
    par = [c for c in rows if c["chunk_level"] == "parent"]
    ch = [c for c in rows if c["chunk_level"] == "child"]
    dup = len(ids) - len(set(ids))
    empty = sum(1 for c in ch if c["chunk_type"] not in STRUCT and not (c.get("content") or "").strip())
    # degenerate: parent with no non-struct children AND no tables/figs
    childless = 0
    kids_by_par = defaultdict(int)
    for c in ch:
        kids_by_par[c.get("parent_chunk_id")] += 1
    for p in par:
        if kids_by_par.get(p["chunk_id"], 0) == 0:
            childless += 1
    ty = Counter(c["chunk_type"] for c in rows)
    nsc = [c for c in ch if c["chunk_type"] not in STRUCT]
    gen = 100 * sum(1 for c in nsc if c["chunk_type"] == "paragraph") / max(1, len(nsc))
    return {
        "n": len(rows), "par": len(par), "ch": len(ch), "dup": dup, "empty": empty,
        "childless": childless, "gen": gen, "ty": ty,
        "exc": ty.get("exception", 0), "note": ty.get("note", 0), "trow": ty.get("table_row", 0),
        "types_distinct": len(ty),
    }

def main():
    socs = sorted(d.name for d in CH.iterdir() if d.is_dir())
    print(f"{'선급':10}{'docs':>5}{'chunks':>9}{'parents':>8}{'dupID':>7}{'empty':>7}{'childless':>10}{'gen%':>6}{'exc':>6}{'note':>6}{'trow':>7}")
    flags = []
    grand = Counter()
    for soc in socs:
        docs = sorted(glob.glob(f"{CH}/{soc}/*.jsonl"))
        agg = Counter(); gens = []; ndoc = 0
        for d in docs:
            try:
                a = audit_doc(d)
            except Exception as e:
                flags.append(f"{soc}/{os.path.basename(d)}: AUDIT-ERR {e}"); continue
            ndoc += 1
            for k in ("n","par","ch","dup","empty","childless","exc","note","trow"):
                agg[k] += a[k]
            gens.append(a["gen"])
            # flag problem docs
            if a["dup"] > 0:
                flags.append(f"{soc}/{os.path.basename(d)[:40]}: dupID={a['dup']}")
            if a["par"] <= 1 and a["n"] > 5:
                flags.append(f"{soc}/{os.path.basename(d)[:40]}: parents={a['par']} (구조 미검출?)")
            if a["empty"] > 0:
                flags.append(f"{soc}/{os.path.basename(d)[:40]}: empty_children={a['empty']}")
        for k in agg: grand[k] += agg[k]
        grand["docs"] += ndoc
        g = statistics.mean(gens) if gens else 0
        print(f"{soc:10}{ndoc:>5}{agg['n']:>9}{agg['par']:>8}{agg['dup']:>7}{agg['empty']:>7}{agg['childless']:>10}{g:>6.0f}{agg['exc']:>6}{agg['note']:>6}{agg['trow']:>7}")
    print(f"{'─'*80}")
    print(f"{'합계':10}{grand['docs']:>5}{grand['n']:>9}{grand['par']:>8}{grand['dup']:>7}{grand['empty']:>7}{grand['childless']:>10}{'':>6}{grand['exc']:>6}{grand['note']:>6}{grand['trow']:>7}")
    print(f"\n무결성: 중복ID {grand['dup']} · 빈청크 {grand['empty']} · 자식없는parent {grand['childless']}")
    if flags:
        print(f"\n⚠ 플래그 {len(flags)}건 (상위 25):")
        for f in flags[:25]:
            print(f"    {f}")
    else:
        print("\n✓ 플래그 없음 — 전 문서 무결성 통과")

if __name__ == "__main__":
    main()
