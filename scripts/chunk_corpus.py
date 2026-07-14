#!/usr/bin/env python3
"""Chunk the sample_20 corpus (162 pre-ETL'd docs) through llmops_core.chunking.

Routes each content_list by its top-level society folder (reliable), runs the
society-specific chunker via the package, and cross-checks that the package's
content-based detect_family() agrees. Writes data_chunks/<society>/<doc>.jsonl.
"""
from __future__ import annotations
import sys, time, traceback
from collections import Counter
from pathlib import Path

REPO = Path("/home/jaehyun/Dev/naver")
sys.path.insert(0, str(REPO))
from llmops_core.chunking import chunk_document, detect_family, write_jsonl

SRC = REPO / "sample_20"
OUT = REPO / "data_chunks"

# top-level society folder ASCII prefix → chunker family
# (folder names are NFD-encoded Korean from macOS; match on the ASCII prefix)
PREFIX_FAMILY = {
    "ABS": "abs_guide",
    "BV": "bv_rule",
    "ClassNK": "nk_rule",
    "DNV": "dnv_cg",
    "IACS": "iacs",
    "KR": "kr_rule",
    "LR": "lr_code",
}
FOLDER_FAMILY = PREFIX_FAMILY  # for the summary loop


def main() -> int:
    cls = sorted(p for p in SRC.rglob("*content_list.json")
                 if not p.name.endswith("v2.json"))
    # ClassNK: Rule_Package 합본 낱권이 단행본과 같은 문서면 스킵(이중 인제스트 방지)
    stems = {p.stem for p in cls}
    dropped = [p for p in cls
               if p.name.startswith("Rule_Package") and "__pdf__" in p.stem
               and p.stem.split("__pdf__")[-1] in stems]
    cls = [p for p in cls if p not in set(dropped)]
    if dropped:
        print(f"skip {len(dropped)} Rule_Package duplicates (standalone 존재)")
    print(f"chunking {len(cls)} documents...\n")
    per_soc = Counter()
    per_soc_chunks = Counter()
    agree = disagree = fail = 0
    total_chunks = 0
    fails: list[str] = []
    disagreements: list[str] = []
    t0 = time.time()

    for cl in cls:
        soc = cl.relative_to(SRC).parts[0].split("_")[0]  # ASCII prefix key
        family = PREFIX_FAMILY.get(soc)
        if not family:
            print(f"  ? unknown society folder: {soc}"); continue
        try:
            detected = detect_family(cl)
            if detected != family:
                disagree += 1
                disagreements.append(f"{soc}/{cl.parent.name}: folder={family} vs detect={detected}")
            else:
                agree += 1
            _, chunks = chunk_document(cl, family=family)
            out = OUT / soc / f"{cl.stem.replace('_content_list','')}_chunks.jsonl"
            write_jsonl(chunks, out)
            per_soc[soc] += 1
            per_soc_chunks[soc] += len(chunks)
            total_chunks += len(chunks)
        except Exception as e:
            fail += 1
            fails.append(f"{soc}/{cl.parent.name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)

    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"완료: {sum(per_soc.values())}/{len(cls)} 문서, {total_chunks} chunks, {dt:.1f}s")
    print(f"\n선급별:")
    for soc in FOLDER_FAMILY:
        print(f"  {soc:16} {per_soc[soc]:3} docs  {per_soc_chunks[soc]:>7} chunks")
    print(f"\ndetect_family 일치: {agree}/{agree+disagree}"
          + (f"  (불일치 {disagree})" if disagree else "  (전부 일치)"))
    for d in disagreements[:15]:
        print(f"    ⚠ {d}")
    if fails:
        print(f"\n실패 {fail}건:")
        for f in fails[:20]:
            print(f"    ✗ {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
