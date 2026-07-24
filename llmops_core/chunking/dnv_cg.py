#!/usr/bin/env python3
"""Domain-aware chunker for **DNV Class Guidelines** (DNV-CG-xxxx) & rules.

Reads a MinerU ``content_list.json`` and emits structured RAG chunks preserving
the DNV hierarchy — SECTION > clause (N) > sub-clause (N.M / N.M.K) — plus
TABLES, FIGURES, equations and definitions, using the "Parent-Child + 표/그림
구조체" strategy and emitting the **same JSONL schema** as the KR/ABS/LR/NK
chunkers so one viewer serves all families.

DNV specifics this handles:

  * Front matter (CLASS GUIDELINE / FOREWORD / CHANGES – CURRENT / CONTENTS) is
    skipped until the first ``SECTION n`` band; the document **Edition** (e.g.
    "Edition July 2021") is parsed from the cover text.
  * Numbering restarts inside every section — Section 2's "1 General" ≠ Section
    1's — so the current SECTION is folded into every chunk id (the DNV analogue
    of the ABS repeated-clause problem).
  * The vector-search **article unit is the deepest decimal clause** (N, N.M or
    N.M.K, all emitted as L2 headings); its enclosing top-level ``N`` is the
    section group. Bodies split into paragraph / enumerated children; a trailing
    equation block is appended to the preceding paragraph.

  parent   article-level chunk — full text + children ids (context extension)
  child    the vector-search units — one paragraph / enumerated item each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/dnv_cg_chunker.py \
        data/DNV_CG_0004/DNV_CG_0004_content_list.json \
        -o data_chunks/dnv_cg_0004_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import interleave_children, merge_figure_fragments, resolve_units, para_of_default
from ._tables import row_retrieval, table_rows


# ── structural markers ──────────────────────────────────────────────────────
RE_SECTION = re.compile(r"^SECTION\s+(\d+)\s+(\S.*)$")          # band (mostly UPPER)
RE_APPENDIX = re.compile(r"^APPENDIX\s+([A-Z0-9]+)\s+(\S.*)$")
RE_CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")           # 1 / 5.1 / 5.1.2 + title
RE_LEADER = re.compile(r"\.{4,}|·{2,}")
RE_ENUM = re.compile(r"^(?:\([a-z0-9ivx]+\)|[a-z]\)|\d+\)|[●•▪]|[–-]\s|—\s)")
# 주요 항목((a)/(1)) + 십진 조항(1.1) — dash·불릿만 제외(붕괴된 조항도 분할되게)
RE_PRIMARY = re.compile(r"^(?:\([a-z0-9ivx]{1,4}\)|[a-z]\)|\d+\))\s")
RE_CLAUSE_ITEM = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")  # 번호 뒤 공백+텍스트만(수식 제외)
RE_EDITION = re.compile(r"Edition\s+([A-Z][a-z]+\s+\d{4})")
RE_DOCID = re.compile(r"\b(DNV(?:GL)?-(?:CG|RU|RP|ST|OS|SE)-[A-Z0-9]+)\b", re.I)
# front-matter headings to ignore until the first real SECTION
FRONT_MATTER = ("class guideline", "foreword", "changes", "contents",
                "editorial corrections", "acknowledgement", "current")

# ── reference extraction ────────────────────────────────────────────────────
RE_REFS = [
    ("dnv_rule", re.compile(r"DNV(?:GL)?-(?:CG|RU|RP|ST|OS|SE|CP)-[A-Z0-9\-.]+", re.I)),
    ("dnv_internal", re.compile(r"\b(?:Sec(?:tion)?|Ch(?:apter)?|Pt|Table|Figure|App(?:endix)?)\s*\.?\s*\d+(?:\.\d+)*\b", re.I)),
    ("standard", re.compile(r"(?:IEC|IEEE|ISO(?:/IEC)?|EN|API|NEK|DIN)\s*[\w.\-:]{0,18}")),
    ("convention", re.compile(r"(?:IMO|SOLAS|MARPOL|MSC|IACS|MODU)\s*[\w.\-/()]{0,25}")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "to be documented", "should be", "acceptance criteria")
PROCEDURE = ("survey", "approval", "verification", "assessment", "analysis",
             "test", "documentation", "review", "application")
CONDITION = ("if ", "where ", "when ", "unless", "provided that", "in case")


def classify(text: str, title: str) -> str:
    low = text.lower()
    if "definition" in title.lower() or "abbreviation" in title.lower():
        return "definition"
    if low.lstrip().startswith("see ") or " refer to " in low:
        return "reference"
    if any(k in low for k in REQUIREMENT):
        return "requirement"
    if any(k in low for k in PROCEDURE):
        return "procedure"
    if any(k in low for k in CONDITION):
        return "condition"
    return "paragraph"


def extract_refs(text: str) -> list[dict]:
    out, seen = [], set()
    for rtype, rx in RE_REFS:
        for m in rx.finditer(text):
            tgt = re.sub(r"\s+", " ", m.group(0)).strip(" ,.;")
            if len(tgt) > 2 and tgt.lower() not in seen:
                seen.add(tgt.lower())
                out.append({"ref_type": rtype, "target": tgt})
    return out


def atom_refs(text: str, caption: str = "") -> list[dict]:
    """표/그림 원자용 참조 — 자기 캡션 번호(그 표 자신)는 제외."""
    cap = re.sub(r"\s+", " ", caption or "").strip()
    return [r for r in extract_refs(text) if not (cap and cap.startswith(r["target"]))]


# DNV definition lines (broadened, Rule ③):
#   "Term: definition" / "Term - definition" / "Term means/is defined as …"
RE_DEF_COLON = re.compile(r"^([A-Z][A-Za-z0-9 &/()\-,]{1,55}?)\s*[:\-–—]\s+(\S.+)$")
RE_DEF_MEANS = re.compile(
    r"^([A-Z][A-Za-z0-9 &/()\-,]{1,55}?)\s+"
    r"(?:means|is defined as|refers to|denotes|is taken to mean)\b", re.I)


def parse_definition(text: str) -> tuple[str | None, str | None]:
    t = text.strip()
    for rx in (RE_DEF_MEANS, RE_DEF_COLON):
        m = rx.match(t)
        if m and len(m.group(1)) <= 50 and " " in t[:80]:
            term = m.group(1).strip()
            acro = None
            am = re.search(r"\(([A-Z][A-Za-z0-9\-]{1,10})\)", term)
            if am:
                acro = am.group(1)
            return term, acro
    return None, None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_cover(items: list[dict]) -> dict:
    docid, edition, subtitle = "", "", ""
    for x in items[:40]:
        if x.get("type") != "text":
            continue
        t = (x.get("text") or "").strip()
        if not docid:
            m = RE_DOCID.search(t)
            if m:
                docid = m.group(1).upper()
        if not edition:
            m = RE_EDITION.search(t)
            if m:
                edition = m.group(1)
        if x.get("text_level") == 2 and not subtitle and t.lower() not in FRONT_MATTER \
                and t.upper() != "CLASS GUIDELINE" and not RE_SECTION.match(t):
            subtitle = t
    return {"doc_id": docid, "edition": edition, "subtitle": subtitle}


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        cov = parse_cover(items)
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        # doc_id는 파일 기반 고유값 — 커버 정규식은 다부 문서(-Pt1/-Pt4)의 Part를 놓쳐 충돌.
        did = re.sub(r"[^0-9A-Za-z]+", "_", stem).strip("_").upper()
        # 시리즈로 문서종류 판정: RU=Rules, CG=Class Guideline, CP=Class Programme,
        # OS/ST=Standard, RP=Recommended Practice. (커버 RE_DOCID는 CP를 못 잡음 → 파일명 우선)
        sm = re.search(r"DNV(?:GL)?[-_](RU|CG|CP|OS|ST|RP|SE)(?![A-Za-z])", did) \
            or re.search(r"DNV(?:GL)?-(RU|CG|CP|OS|ST|RP|SE)(?![A-Za-z])", cov["doc_id"] or "")
        self.doc_type = {"RU": "rule", "CG": "guideline", "CP": "programme",
                         "OS": "standard", "ST": "standard", "RP": "recommended-practice",
                         "SE": "service-specification"}.get(sm.group(1) if sm else "", "guideline")
        self.doc_meta = {
            "doc_id": did,
            "doc_title": f'{cov["doc_id"]} {cov["subtitle"]}'.strip() or stem,
            "part_no": "", "part_title": cov["subtitle"], "publisher": "DNV",
            "year": (cov["edition"].split()[-1] if cov["edition"] else "2025"),
            "edition": cov["edition"], "language": "en",
        }

    def article_id(self, is_appx: bool, band: str, clause: str | None) -> str:
        pre = "APX" if is_appx else "S"
        base = f'{self.doc_meta["doc_id"]}_{pre}{band}'
        return f"{base}_{clause.replace('.', '-')}" if clause else f"{base}_INTRO"

    def _uniq(self, cid: str) -> str:
        if cid not in self._seen_ids:
            self._seen_ids.add(cid)
            return cid
        i = ord("b")
        while f"{cid}-{chr(i)}" in self._seen_ids:
            i += 1
        cid = f"{cid}-{chr(i)}"
        self._seen_ids.add(cid)
        return cid

    def run(self) -> list[dict]:
        st = {
            "appx": False, "band": ("0", ""),        # SECTION / APPENDIX
            "grp": ("", ""),                          # top-level clause N
            "art": (None, ""),
            "pieces": [], "tables": [], "figures": [],
            "started": False,
        }

        def flush():
            self._flush(st)
            st["pieces"], st["tables"], st["figures"] = [], [], []

        for x in self.items:
            typ = x.get("type")
            page = x.get("page_idx")

            if typ in ("header", "footer", "page_number", "page_footnote"):
                continue

            raw = (x.get("text") or "").strip()

            # SECTION / APPENDIX band (real headings only, not the ToC list)
            if typ == "text" and raw and x.get("text_level") in (1, 2):
                ms, ma = RE_SECTION.match(raw), RE_APPENDIX.match(raw)
                if ms:
                    flush(); st["started"] = True; st["appx"] = False
                    st["band"] = (ms.group(1), ms.group(2).strip())
                    st["grp"], st["art"] = ("", ""), (None, ""); continue
                if ma:
                    flush(); st["started"] = True; st["appx"] = True
                    st["band"] = (ma.group(1), ma.group(2).strip())
                    st["grp"], st["art"] = ("", ""), (None, ""); continue

            if not st["started"]:
                continue      # still in front matter

            if typ == "table":
                st["tables"].append((x, len(st["pieces"])))
                continue
            if typ == "image":
                st["figures"].append((x, len(st["pieces"])))
                continue
            if typ == "equation":
                eq = (x.get("text") or "").strip()
                if eq:
                    if st["pieces"]:
                        st["pieces"][-1]["text"] += "\n" + eq
                        st["pieces"][-1]["pages"].add(page)
                    else:                          # 문단 없이 시작하는 단독 수식도 보존
                        st["pieces"].append({"text": eq, "pages": {page}, "enum": False})
                continue
            if typ == "list":
                for li in x.get("list_items", []):
                    li = (li or "").strip()
                    if li and not RE_LEADER.search(li):
                        st["pieces"].append({"text": li, "pages": {page}, "enum": True})
                continue
            if typ not in ("text", "ref_text") or not raw or RE_LEADER.search(raw):
                continue
            if raw.lower() in FRONT_MATTER:
                continue

            # clause article (N / N.M / N.M.K) at L2
            m = RE_CLAUSE.match(raw)
            if m and x.get("text_level") == 2:
                flush()
                ano, title = m.group(1), m.group(2).strip()
                st["art"] = (ano, title)
                grp = ano.split(".")[0]
                if "." not in ano:
                    st["grp"] = (grp, title)
                elif st["grp"][0] != grp:
                    st["grp"] = (grp, "")
                continue

            st["pieces"].append({"text": raw, "pages": {page},
                                 "enum": bool(RE_ENUM.match(raw))})

        flush()
        return self.out

    def _flush(self, st: dict):
        band_no, band_title = st["band"]
        grp_no, grp_title = st["grp"]
        ano, atitle = st["art"]
        pieces, tables = st["pieces"], st["tables"]
        figures = merge_figure_fragments(st["figures"])   # 조각난 복합 그림 병합
        if not (pieces or tables or figures):
            return
        is_appx = st["appx"]
        band = "APPENDIX" if is_appx else "SECTION"

        parent_id = self._uniq(self.article_id(is_appx, band_no, ano))
        path = [self.doc_meta["doc_title"], f"{band} {band_no} {band_title}".strip()]
        if grp_no and grp_no != ano:
            path.append(f"{grp_no} {grp_title}".strip())
        if ano:
            path.append(f"{ano} {atitle}".strip())

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t, _ in tables}
                       | {f.get("page_idx") for f, _ in figures})
        full_text = "\n".join(pc["text"] for pc in pieces)
        is_def = "definition" in atitle.lower() or "abbreviation" in atitle.lower()

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta, "document_type": "appendix" if is_appx else self.doc_type,
                "chapter_no": band_no, "chapter_title": band_title,      # SECTION
                "section_no": grp_no or ano, "section_title": grp_title,  # clause group
                "article_no": ano, "article_title": atitle,
                "notations": [], "section_path": path, "pages": pages,
                "source_file": self.source_file,
            }
            base.update(extra)
            return base

        child_units = [u for u in self._split_children(pieces, is_def) if u["text"].strip()]
        child_ids, table_ids, figure_ids = [], [], []
        child_chunks = []
        for k, unit in enumerate(child_units, 1):
            cid = f"{parent_id}_C{k:03d}"
            child_ids.append(cid)
            rec = meta(
                pages=sorted(unit["pages"]) if unit.get("pages") else pages,  # 유닛 실제 페이지
                chunk_id=cid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="text",
                local_heading=unit.get("heading", ""),
                paragraph_no=unit.get("para_no", ""),
                paragraph_title=unit.get("para_title", ""),
                item_no=unit.get("item_no", ""),
                sub_item_no=unit.get("sub_item_no", ""),
                content=unit["text"], summary="", topic="",
                keywords=[], entities=[],
                references=extract_refs(unit["text"]),
                previous_chunk_id=None, next_chunk_id=None,
                linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
            )
            if is_def:
                term, acro = parse_definition(unit["text"])
                if term:
                    rec["term_ko"] = term
                    rec["term_en"] = acro
                    rec["local_heading"] = term
            child_chunks.append(rec)
        for a, b in zip(child_chunks, child_chunks[1:]):
            a["next_chunk_id"] = b["chunk_id"]
            b["previous_chunk_id"] = a["chunk_id"]

        table_chunks = []
        for k, (tb, _ta) in enumerate(tables, 1):
            tid = f"{parent_id}_T{k:03d}"
            table_ids.append(tid)
            cap = " ".join(tb.get("table_caption") or [])
            body = tb.get("table_body", "")
            rows = table_rows(body)
            tpages = [tb["page_idx"]] if tb.get("page_idx") is not None else pages  # 원자 자신의 페이지(세그먼트 span 아님)
            tchunk = meta(
                pages=tpages,
                chunk_id=tid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="table",
                table_caption=cap, table_html=body,
                table_footnote=" ".join(tb.get("table_footnote") or []),
                img_path=tb.get("img_path", ""), content=cap,
                retrieval_text=(self._table_text(cap, body) + " " + " ".join(tb.get("table_footnote") or [])).strip(),
                table_nrows=len(rows), linked_table_rows=[],
                summary="", linked_article_id=parent_id,
            )
            tchunk["references"] = atom_refs(tchunk["retrieval_text"], cap)
            table_chunks.append(tchunk)
            header = rows[0] if rows else []
            if len(rows) >= 2 and len(header) >= 2:
                for ri, row in enumerate(rows[1:], 1):
                    rtext = row_retrieval(header, row)
                    if not rtext.strip():
                        continue
                    rid = f"{tid}_R{ri:03d}"
                    tchunk["linked_table_rows"].append(rid)
                    table_chunks.append(meta(
                        pages=tpages,
                        chunk_id=rid, parent_chunk_id=tid,
                        chunk_level="child", chunk_type="table_row",
                        table_caption=cap, row_index=ri, content=rtext,
                        retrieval_text=(cap + " " + rtext).strip(),
                        references=atom_refs(rtext, cap),
                        summary="", linked_article_id=parent_id, linked_table_id=tid,
                    ))

        figure_chunks = []
        for k, (fg, _fa) in enumerate(figures, 1):
            fid = f"{parent_id}_F{k:03d}"
            figure_ids.append(fid)
            cap = " ".join(fg.get("image_caption") or [])
            vis = (fg.get("content") or "").strip()   # ETL 이미지 분석(설명/OCR — MinerU image-analysis)
            figure_chunks.append(meta(
                pages=[fg["page_idx"]] if fg.get("page_idx") is not None else pages,
                chunk_id=fid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="figure",
                caption=cap, image_path=fg.get("img_path", ""),
                image_paths=fg.get("img_paths") or ([fg["img_path"]] if fg.get("img_path") else []),
                visual_summary=vis, image_kind=fg.get("sub_type") or "",
                content=cap,
                retrieval_text=" ".join(t for t in (cap, atitle, vis) if t),
                references=atom_refs(" ".join((cap, vis)), cap),
                linked_article_id=parent_id,
            ))

        parent = meta(
            chunk_id=parent_id, parent_chunk_id=None,
            chunk_level="parent", chunk_type="article",
            content=full_text, content_tokens=approx_tokens(full_text),
            summary="", topic="", has_cross_ref=False,
            references=extract_refs(full_text), children=child_ids,
            linked_tables=table_ids, linked_figures=figure_ids,
            linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
        )
        # 텍스트 자식과 그림·표 원자를 원문 등장 위치대로 병합 방출(ID 체계는 종류별 유지)
        _grps: list[list[dict]] = []
        for _c in table_chunks:
            if _c["chunk_type"] == "table":
                _grps.append([_c])
            else:
                _grps[-1].append(_c)
        _atom_groups = list(zip([a for _, a in tables], _grps)) \
            + list(zip([a for _, a in figures], [[c] for c in figure_chunks]))
        self.out.append(parent)
        self.out.extend(interleave_children(
            [(u.get("src_idx", 0), c) for u, c in zip(child_units, child_chunks)],
            _atom_groups))

    @staticmethod
    def _split_children(pieces: list[dict], is_def: bool) -> list[dict]:
        # 비-정의: 상대뎁스 리졸버(항>호>목, 혼재 대응)에 위임.
        if not is_def:
            return resolve_units(pieces, para_of_default)
        # 정의: 각 정의 항목이 개별 유닛
        units: list[dict] = []
        cur: dict | None = None
        for _si, pc in enumerate(pieces):
            t = pc["text"]
            if cur is None or parse_definition(t)[0]:
                if cur:
                    units.append(cur)
                cur = {"text": t, "heading": "", "para_no": "",
                       "para_title": "", "item_no": "",
                       "pages": set(pc.get("pages") or ()), "src_idx": _si}
            else:
                cur["text"] += "\n" + t
                cur["pages"] |= set(pc.get("pages") or ())
        if cur:
            units.append(cur)
        return units or [{"text": "", "heading": "", "para_no": "",
                          "para_title": "", "item_no": ""}]

    @staticmethod
    def _table_text(caption: str, html: str) -> str:
        cells = re.sub(r"<[^>]+>", " ", html)
        cells = cells.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")   # 부등호·기호 보존
        return re.sub(r"\s+", " ", (caption + " " + cells)).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content_list")
    ap.add_argument("-o", "--output", default="data_chunks/dnv_cg_chunks.jsonl")
    args = ap.parse_args()

    items = json.loads(Path(args.content_list).read_text(encoding="utf-8"))
    chunks = Chunker(items, Path(args.content_list).name).run()

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    lv = Counter(c["chunk_level"] for c in chunks)
    ty = Counter(c["chunk_type"] for c in chunks)
    print(f"wrote {len(chunks)} chunks -> {op}")
    print(f"  levels : {dict(lv)}")
    print(f"  types  : {dict(ty)}")


if __name__ == "__main__":
    main()
