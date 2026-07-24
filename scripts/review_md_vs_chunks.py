#!/usr/bin/env python3
"""대표 문서별로 .md(원문) 각 텍스트 줄이 청킹 결과에 실제로 담겼는지 대조.
누락(청커가 버린 원문) / 커버리지 / 구조 정렬을 처음부터 끝까지 검사한다."""
from __future__ import annotations
import glob, json, re, unicodedata, os, sys
sys.path.insert(0, "/home/jaehyun/Dev/naver")
from llmops_core.chunking import chunk_document

PICKS = {
    "ABS": ("RequirementsforLifeExtensionofFloatingProduct", "abs_guide"),
    "BV": ("590-NI_2023-03", "bv_rule"),
    "ClassNK": ("380_part_a_e_202512", "nk_rule"),
    "DNV": ("DNV-RU-INV-Pt1", "dnv_cg"),
    "IACS": ("REC_52", "iacs"),
}


def find(stem):
    return next(p for p in glob.glob("sample_20/**/*content_list.json", recursive=True)
               if "v2" not in p and unicodedata.normalize("NFC", os.path.basename(p)).startswith(stem))


def norm(s):
    return re.sub(r"\s+", "", s or "").lower()


def md_lines(md):
    raw = []
    for ln in md.split("\n"):
        t = ln.strip()
        if len(t) < 15:
            continue
        if re.search(r"\.{4,}|·{2,}", t):        # ToC dot leaders
            continue
        if t.startswith(("<", "![", "|", "#")):  # raw html/table/image/heading markup
            t = re.sub(r"^#+\s*", "", t)
            if t.startswith(("<", "![", "|")):
                continue
        raw.append(t)
    # 반복 러닝헤더/푸터(페이지마다 반복되는 줄) 제거 — 3회 이상 등장
    from collections import Counter
    freq = Counter(norm(t) for t in raw)
    return [t for t in raw if freq[norm(t)] < 3]


def main():
    soc = sys.argv[1] if len(sys.argv) > 1 else None
    for society, (stem, fam) in PICKS.items():
        if soc and society != soc:
            continue
        cl = find(stem)
        md = open(cl.replace("_content_list.json", ".md"), encoding="utf-8").read()
        _, chunks = chunk_document(cl, family=fam)
        parts = []
        for c in chunks:                      # 캡처된 모든 것: 내용 + 제목·경로(metadata) + 표
            parts += [c.get("content") or "", c.get("table_caption") or "", c.get("caption") or ""]
            parts += c.get("section_path") or []          # "1.1 Application" 형태 헤딩 포함
            if c.get("table_html"):
                parts.append(re.sub(r"<[^>]+>", " ", c["table_html"]))
        blob = norm(" ".join(parts))
        lines = md_lines(md)
        missing = []
        for ln in lines:
            key = norm(ln)[:40]
            if len(key) >= 10 and key not in blob:
                missing.append(ln)
        par = [c for c in chunks if c["chunk_level"] == "parent"]
        ch = [c for c in chunks if c["chunk_level"] == "child"]
        dup = len([c["chunk_id"] for c in chunks]) - len(set(c["chunk_id"] for c in chunks))
        print(f"\n{'='*70}\n{society}: {stem}  (family={fam})")
        print(f"  parents={len(par)} children={len(ch)} dupID={dup}")
        print(f"  md 본문 줄 {len(lines)} → 청크 미포함 {len(missing)} ({100*(1-len(missing)/max(1,len(lines))):.0f}% 커버)")
        for m in missing[:18]:
            print(f"    ✗ {m[:88]}")


if __name__ == "__main__":
    main()
