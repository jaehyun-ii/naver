#!/usr/bin/env python3
"""Domain-aware chunker for **ABS** Guides / Guidance Notes / Requirements / Rules.

Document metadata (doc_id, title, category, year, Part) is parsed per-document
from the cover/foreword (``parse_cover``) — not hardcoded to any single guide.

Reads a MinerU ``content_list.json`` and emits structured RAG chunks that
preserve the ABS SECTION > clause > sub-clause > sub-sub-clause decimal
hierarchy (plus APPENDICES, TABLES, FIGURES, definitions and effective-date
tags), mirroring the "Parent-Child + 표/그림 구조체" strategy used for the KR
rulebook — and emitting the **same JSONL schema** so one viewer serves both.

  parent   clause-level chunk — full text + children ids (context extension)
  child    the vector-search units — one paragraph / enumerated item each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Sections restart clause numbering (Section 2's "1" ≠ Section 3's "1"), so the
current SECTION/APPENDIX is tracked from the running page header and folded into
every chunk id — the ABS analogue of the KR repeated-조 problem.

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/abs_guide_chunker.py \
        data/251-cybersafety-v2-cybersecurity-guide-jun25/251-*_content_list.json \
        -o data_chunks/abs_cyber_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import resolve_units, para_of_default

# ── structural markers ──────────────────────────────────────────────────────
# MinerU가 밴드 헤더를 자간이 벌어진 형태로 추출하는 경우가 잦다("S E C T I O N 2").
# 밴드 판별 전에 선두의 벌어진 대문자 런을 한 단어로 복원한다.
RE_SPACED_BAND = re.compile(r"^((?:[A-Z]\s+){3,}[A-Z])(.*)$")


def despace_band(t: str) -> str:
    m = RE_SPACED_BAND.match(t)
    if m:
        word = m.group(1).replace(" ", "")
        if re.match(r"^(SECTION|CHAPTER|PART|APPENDIX)", word, re.I):
            return (word + m.group(2)).strip()
    return t


# running header, e.g. "Section 2 Notation Requirements 2" / "Appendix 1 … A1"
RE_HEADER = re.compile(r"^(Section|Chapter|Appendix)\s+(\d+)\s+(.*?)\s+(?:\d+|A\d+)\s*$", re.I)
# 본문에 인라인으로 나오는 밴드 헤더(자간 복원 후): "SECTION 2 …" / "CHAPTER 2 …"
RE_BAND_INLINE = re.compile(r"^(SECTION|CHAPTER|APPENDIX)\s+(\d+)\s*(.*)$", re.I)
RE_CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")        # 2 / 2.1 / 2.2.1 + title
RE_LEADER = re.compile(r"\.{4,}|·{2,}")                       # table-of-contents dot leaders
RE_STRAY = re.compile(r"^\d{1,4}$")                           # 페이지·ToC·분리된 번호 조각(단독)
RE_EFFDATE = re.compile(r"\s*\((\d{1,2}\s+\w+\s+\d{4})\)\s*$")  # (1 June 2025)

# child-split boundaries inside a clause body
RE_ENUM = re.compile(r"^(?:\([a-z0-9ivx]+\)|[a-z]\)|\d+\)|[●•▪]|[–-]\s)")
# 주요 항목((a)/(1)) + 십진 조항(1.1) — dash·불릿만 제외(붕괴된 조항도 분할되게)
RE_PRIMARY = re.compile(r"^(?:\([a-z0-9ivx]{1,4}\)|[a-z]\)|\d+\))\s")
RE_CLAUSE_ITEM = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")  # 번호 뒤 공백+텍스트만(수식 제외)

# ── reference / notation extraction ─────────────────────────────────────────
NOTATIONS = ["CS-System", "CS-Ready", "CS-1", "CS-2", "CR-Ex"]
RE_REFS = [
    ("abs_rule", re.compile(r"ABS\s+(?:MVR|Rules?|Guide|Marine Vessel Rules)[^,.;\n]{0,40}")),
    ("abs_internal", re.compile(r"\b\d+/\d+(?:\.\d+)*\b")),          # 1/5.1 cross-ref
    ("standard", re.compile(r"(?:IEC|IEEE|ISO(?:/IEC)?|NIST(?:\s+SP)?|IMO|SOLAS)\s*[\w.\-:]{0,18}")),
    ("internal_section", re.compile(r"\b(?:Section|Appendix|Table|Figure)\s+\d+\b")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "to be submitted", "is to ", "are to ")
PROCEDURE = ("process", "survey", "review", "submittal", "application",
             "shall submit", "is to submit")
CONDITION = ("if ", "where ", "when ", "unless", "provided that")


def classify(text: str, title: str) -> str:
    low = text.lower()
    head = low.lstrip()[:12]
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
            if len(tgt) > 2 and tgt not in seen:
                seen.add(tgt)
                out.append({"ref_type": rtype, "target": tgt})
    return out


def find_notations(text: str) -> list[str]:
    return [n for n in NOTATIONS if n in text]


# ABS definition line: "Term. Definition ..."  (term may end with (ACRONYM))
RE_DEF = re.compile(r"^([A-Z][A-Za-z0-9 &/®™®()\-,\.]{1,70}?)\.\s+(\S.+)$")


def parse_definition(text: str) -> tuple[str | None, str | None]:
    m = RE_DEF.match(text.strip())
    if not m:
        return None, None
    term = m.group(1).strip()
    # drop trademark noise, pull a parenthetical acronym as the "en" slot
    acro = None
    am = re.search(r"\(([A-Z][A-Za-z0-9\-]{1,10})\)", term)
    if am:
        acro = am.group(1)
    term = re.sub(r"[®™]", "", term).strip()
    if len(term) > 60 or " " not in text[:80]:
        return None, None
    return term, acro


# ── document metadata (per-document, parsed from the cover/foreword) ─────────
# 표지 종류 접두어 → (표시 접두어, section_path 루트, 문서 카테고리)
RE_COVER_KIND = re.compile(
    r"^(GUIDANCE NOTES ON|GUIDE FOR|REQUIREMENTS FOR|ADVISORY ON|"
    r"RULES FOR BUILDING AND CLASSING|RULES FOR|SUPPLEMENT TO)\b", re.I)
RE_COVER_DATE = re.compile(
    r"\b((?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(?:19|20)\d{2})\b", re.I)
RE_COVER_PART = re.compile(r"^Part\s+([0-9A-Za-z]+)\b", re.I)
_KIND = {
    "guidance notes on": ("Guidance Notes on", "ABS Guidance Notes", "guidance-notes"),
    "guide for":         ("Guide for", "ABS Guide", "guide"),
    "requirements for":  ("Requirements for", "ABS Requirements", "requirements"),
    "advisory on":       ("Advisory on", "ABS Advisory", "advisory"),
    "rules for building and classing": ("Rules for Building and Classing", "ABS Rules", "rules"),
    "rules for":         ("Rules for", "ABS Rules", "rules"),
    "supplement to":     ("Supplement to", "ABS Supplement", "supplement"),
}
_KIND_CONNECTORS = re.compile(r"\b(Of|For|On|To|And|The|In|A|An|Using)\b")


def _titlecase(s: str) -> str:
    """ALL-CAPS 표지 제목만 Title Case로(약어·혼합대소문자는 보존)."""
    if s and s.isupper():
        s = s.title()
        s = _KIND_CONNECTORS.sub(lambda m: m.group(1).lower(), s)
        s = s[0].upper() + s[1:]
    return s


def _foreword_kind(items: list[dict]) -> str:
    """표지 템플릿("RULES FOR")이 부정확할 때 서문 자기소개 문구로 문서종류 보정."""
    for x in items[:70]:
        low = (x.get("text") or "").lower()
        if "these guidance notes" in low or "this guidance note" in low:
            return "guidance notes on"
        if low.startswith("this guide") or "this guide is" in low or "this guide references" in low:
            return "guide for"
        if "these requirements" in low or "this requirement" in low:
            return "requirements for"
        if "this advisory" in low:
            return "advisory on"
        if low.startswith("these rules") or "these rules for" in low:
            return "rules for building and classing"
    return ""


def parse_cover(items: list[dict], stem: str) -> dict:
    """표지·서문에서 문서별 메타(doc_id/title/종류/연도/Part)를 도출.

    하드코딩(사이버보안 가이드) 대신 문서마다 실제 표지를 읽는다. 문서종류는
    **제목 자체 접두어 > 서문 자기소개 문구 > 표지 템플릿 라인** 우선순위로 판정
    (일부 표지는 상단에 무의미한 "RULES FOR" 템플릿을 인쇄하기 때문).
    """
    date = title = part_no = part_title = ""
    cover_kind = ""
    cover = [x for x in items[:70]
             if x.get("type") in ("text", "footer") and (x.get("text") or "").strip()]
    for x in cover:
        t = (x.get("text") or "").strip()
        if not date:
            dm = RE_COVER_DATE.search(t)
            if dm:
                date = dm.group(1)
        if not cover_kind:
            km = RE_COVER_KIND.match(t)
            if km and x.get("text_level") not in (1, 2):     # 표지 상단 템플릿 라인
                cover_kind = km.group(1).lower()
        if not title and x.get("text_level") in (1, 2):
            tt = RE_COVER_DATE.sub("", t).strip()
            if tt.upper() not in ("ABS", "") and len(tt) > 2:
                title = tt
    kind_key = ""
    tm = RE_COVER_KIND.match(title)
    if tm:                                                    # 제목 자체가 접두어로 시작
        kind_key = tm.group(1).lower()
        title = title[tm.end():].strip(" :-")
    if not kind_key:
        kind_key = _foreword_kind(items)
    if not kind_key:
        kind_key = cover_kind
    for i, x in enumerate(cover):                             # Part(다부 문서: MVR/OR)
        pm = RE_COVER_PART.match((x.get("text") or "").strip())
        if pm and x.get("page_idx") == 0:
            part_no = pm.group(1)
            for y in cover[i + 1:i + 3]:
                yt = (y.get("text") or "").strip()
                if yt and not RE_COVER_DATE.search(yt) and yt.upper() != "ABS":
                    part_title = _titlecase(yt)
                    break
            break
    prefix, path_root, cat = _KIND.get(kind_key, ("", "ABS", "guide"))
    title = _titlecase(title)
    doc_title = f"{prefix} {title}".strip() if prefix else (title or stem)
    if part_no:
        doc_title += f" — Part {part_no}" + (f" {part_title}" if part_title else "")
    doc_id = "ABS_" + re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").upper()
    return {
        "doc_id": doc_id, "doc_title": doc_title,
        "part_no": part_no, "part_title": part_title or title or path_root,
        "publisher": "ABS", "year": (date.split()[-1] if date else ""),
        "language": "en", "doc_category": cat, "edition": date,
        "path_root": path_root,
    }


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # English ~4 chars/token


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        cov = parse_cover(items, stem)
        self.path_root = cov.pop("path_root")   # section_path 루트(청크 필드로는 스프레드 안 함)
        self.doc_meta = cov

    # -- id / context helpers ------------------------------------------------
    _BAND_PREFIX = {"appendix": "APX", "chapter": "C", "section": "S"}

    def base_id(self, bkind: str, sec_no: str) -> str:
        # doc_id를 접두로 넣어 문서 간 chunk_id 충돌 방지(벡터 point-id=uuid5(chunk_id)).
        return f"{self.doc_meta['doc_id']}_{self._BAND_PREFIX.get(bkind, 'S')}{sec_no}"

    def article_id(self, bkind, sec_no, ano) -> str:
        base = self.base_id(bkind, sec_no)
        return f"{base}_{ano.replace('.', '-')}" if ano else f"{base}_INTRO"

    def run(self) -> list[dict]:
        self._parse()
        return self.out

    def _page_sections(self) -> dict:
        """Map every page → (is_appx, section_no, title) from running headers.

        MinerU emits the running header *after* a page's body, so switching on the
        header token lags by whatever precedes it on that page. Resolving section
        by page_idx instead is order-independent. Pages with no header inherit the
        nearest known one (fill-forward; leading pages fill-backward)."""
        raw: dict[int, tuple] = {}
        for x in self.items:
            t = despace_band((x.get("text") or "").strip())
            if x.get("type") == "header":
                m = RE_HEADER.match(t)
                if m:
                    raw[x.get("page_idx")] = (m.group(1).lower(),
                                              m.group(2), m.group(3).strip())
            elif x.get("type") == "text" and x.get("text_level") and len(t) < 80:
                # 러닝헤더가 깨진 페이지 보강: 본문 인라인 밴드 헤더도 페이지맵에 반영
                bi = RE_BAND_INLINE.match(t)
                if bi:
                    raw[x.get("page_idx")] = (bi.group(1).lower(),
                                              bi.group(2), bi.group(3).strip())
        if not raw:
            return {}
        pages = [p for p in (x.get("page_idx") for x in self.items) if p is not None]
        lo, hi = min(pages), max(pages)
        first = raw[min(raw)]
        out, last = {}, first
        for p in range(lo, hi + 1):
            if p in raw:
                last = raw[p]
            out[p] = last
        return out

    def _parse(self):
        psec = self._page_sections()
        st = {
            "bkind": "section", "sec": ("1", ""),    # 밴드 종류(section/chapter/appendix)+번호
            "grp": ("", ""),                         # top-level clause group (n, title)
            "art": (None, ""), "art_eff": "",        # current clause article
            "pieces": [], "tables": [], "figures": [],
            "started": False,
        }

        def flush():
            self._flush(st)
            st["pieces"], st["tables"], st["figures"] = [], [], []
            st["art_eff"] = ""

        for x in self.items:
            typ = x.get("type")
            page = x.get("page_idx")

            if typ in ("header", "footer", "page_number"):
                continue

            # resolve 밴드(SECTION/CHAPTER/APPENDIX) by page (order-independent); switch flushes
            sec = psec.get(page)
            if sec is not None:
                bkind, sno, stitle = sec
                if (bkind, sno) != (st["bkind"], st["sec"][0]):
                    flush()
                    st["bkind"], st["sec"] = bkind, (sno, stitle)
                    st["grp"], st["art"] = ("", ""), (None, "")
                elif stitle and not st["sec"][1]:      # backfill title once known
                    st["sec"] = (sno, stitle)

            if typ == "table":
                if st["started"]:
                    st["tables"].append(x)
                continue
            if typ == "image":
                if st["started"]:
                    st["figures"].append(x)
                continue

            if typ == "list":
                for li in x.get("list_items", []):
                    li = (li or "").strip()
                    if st["started"] and li and not RE_LEADER.search(li) \
                            and not RE_STRAY.match(li):
                        st["pieces"].append({"text": li, "pages": {page}, "enum": True})
                continue

            if typ not in ("text", "ref_text"):       # ref_text=오분류된 열거항목 본문
                continue
            raw = (x.get("text") or "").strip()
            _db = despace_band(raw)
            if (RE_BAND_INLINE.match(_db) and len(_db) < 80) \
                    or _db.upper() in ("SECTION", "CHAPTER", "APPENDIX", "PART"):
                continue                              # 밴드 헤더(+번호 없는 조각)는 본문 제외
            if not raw or RE_LEADER.search(raw):      # skip empties + TOC leaders
                continue

            m = RE_CLAUSE.match(raw)
            if m and x.get("text_level") == 2:
                st["started"] = True
                flush()
                ano, title = m.group(1), m.group(2)
                dm = RE_EFFDATE.search(title)
                if dm:
                    st["art_eff"] = dm.group(1)
                    title = RE_EFFDATE.sub("", title).strip()
                st["art"] = (ano, title)
                grp_no = ano.split(".")[0]
                if "." not in ano:                     # top-level clause: it's the group too
                    st["grp"] = (grp_no, title)
                elif st["grp"][0] != grp_no:
                    st["grp"] = (grp_no, st["grp"][1] if st["grp"][0] == grp_no else "")
                continue

            if st["started"] and not RE_STRAY.match(raw):     # 단독 번호 조각 제외
                st["pieces"].append({"text": raw, "pages": {page}, "enum": False})

        flush()

    def _flush(self, st: dict):
        sec_no, sec_title = st["sec"]
        grp_no, grp_title = st["grp"]
        ano, atitle = st["art"]
        pieces, tables, figures = st["pieces"], st["tables"], st["figures"]
        if not (pieces or tables or figures):
            return
        bkind = st["bkind"]

        parent_id = self.article_id(bkind, sec_no, ano)
        # boundary pages can print a next-section clause under the prior running
        # header, colliding a clause number within a section — keep both, unique id.
        if parent_id in self._seen_ids:
            suffix = ord("b")
            while f"{parent_id}-{chr(suffix)}" in self._seen_ids:
                suffix += 1
            parent_id = f"{parent_id}-{chr(suffix)}"
        self._seen_ids.add(parent_id)
        kind = "appendix" if bkind == "appendix" else "guide"
        band = bkind.upper()
        path = [self.path_root, self.doc_meta["part_title"], f"{band} {sec_no} {sec_title}".strip()]
        if grp_no and grp_no != ano:
            path.append(f"{grp_no} {grp_title}".strip())
        if ano:
            path.append(f"{ano} {atitle}".strip())

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t in tables}
                       | {f.get("page_idx") for f in figures})
        full_text = "\n".join(pc["text"] for pc in pieces)
        notations = sorted(set(find_notations(full_text)))

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta, "document_type": kind,
                "chapter_no": sec_no, "chapter_title": sec_title,     # SECTION
                "section_no": grp_no or ano, "section_title": grp_title,  # clause group
                "article_no": ano, "article_title": atitle,
                "effective_date": st["art_eff"], "notations": notations,
                "section_path": path, "pages": pages,
                "source_file": self.source_file,
            }
            base.update(extra)
            return base

        is_def = "definition" in atitle.lower() or "abbreviation" in atitle.lower()
        child_units = [u for u in self._split_children(pieces, atitle) if u["text"].strip()]
        child_ids, table_ids, figure_ids = [], [], []
        child_chunks = []
        for k, unit in enumerate(child_units, 1):
            cid = f"{parent_id}_C{k:03d}"
            child_ids.append(cid)
            rec = meta(
                chunk_id=cid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="text",
                local_heading=unit.get("heading", ""),
                paragraph_no=unit.get("para_no", ""),        # 소속 하위조항 번호
                paragraph_title=unit.get("para_title", ""),  # 소속 하위조항 제목
                item_no=unit.get("item_no", ""),             # 자신의 항목(a)/(1)
                sub_item_no=unit.get("sub_item_no", ""),
                content=unit["text"], summary="", topic="",
                keywords=[], entities=find_notations(unit["text"]),
                references=extract_refs(unit["text"]),
                previous_chunk_id=None, next_chunk_id=None,
                linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
            )
            if is_def:
                term, acro = parse_definition(unit["text"])
                if term:
                    rec["term_ko"] = term      # reuse KR field for the display term
                    rec["term_en"] = acro
                    rec["local_heading"] = term
            child_chunks.append(rec)
        for a, b in zip(child_chunks, child_chunks[1:]):
            a["next_chunk_id"] = b["chunk_id"]
            b["previous_chunk_id"] = a["chunk_id"]

        table_chunks = []
        for k, tb in enumerate(tables, 1):
            tid = f"{parent_id}_T{k:03d}"
            table_ids.append(tid)
            cap = " ".join(tb.get("table_caption") or [])
            body = tb.get("table_body", "")
            tpages = [tb["page_idx"]] if tb.get("page_idx") is not None else pages  # 원자 자신의 페이지(세그먼트 span 아님)
            table_chunks.append(meta(
                pages=tpages,
                chunk_id=tid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="table",
                table_caption=cap, table_html=body,
                table_footnote=" ".join(tb.get("table_footnote") or []),
                img_path=tb.get("img_path", ""), content=cap,
                retrieval_text=(self._table_text(cap, body) + " " + " ".join(tb.get("table_footnote") or [])).strip(),
                summary="", linked_article_id=parent_id,
            ))

        figure_chunks = []
        for k, fg in enumerate(figures, 1):
            fid = f"{parent_id}_F{k:03d}"
            figure_ids.append(fid)
            cap = " ".join(fg.get("image_caption") or [])
            vis = (fg.get("content") or "").strip()   # ETL 이미지 분석(설명/OCR — MinerU image-analysis)
            figure_chunks.append(meta(
                pages=[fg["page_idx"]] if fg.get("page_idx") is not None else pages,
                chunk_id=fid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="figure",
                caption=cap, image_path=fg.get("img_path", ""),
                visual_summary=vis, content=cap,
                retrieval_text=" ".join(t for t in (cap, atitle, vis) if t),
                linked_article_id=parent_id,
            ))

        parent = meta(
            chunk_id=parent_id, parent_chunk_id=None,
            chunk_level="parent", chunk_type="article",
            content=full_text, content_tokens=approx_tokens(full_text),
            summary="", topic="", has_cross_ref=bool(notations),
            references=extract_refs(full_text), children=child_ids,
            linked_tables=table_ids, linked_figures=figure_ids,
            linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
        )
        self.out.append(parent)
        self.out.extend(child_chunks)
        self.out.extend(table_chunks)
        self.out.extend(figure_chunks)

    @staticmethod
    def _split_children(pieces: list[dict], title: str) -> list[dict]:
        is_def = "definition" in title.lower() or "abbreviation" in title.lower()
        # 비-정의: 상대뎁스 리졸버(항>호>목, 혼재 대응)에 위임.
        if not is_def:
            return resolve_units(pieces, para_of_default) or [
                {"text": title, "heading": "", "para_no": "",
                 "para_title": "", "item_no": ""}]
        # 정의: 각 정의 항목이 개별 유닛
        units: list[dict] = []
        cur: dict | None = None
        for pc in pieces:
            t = pc["text"]
            if cur is None or RE_DEF.match(t):
                if cur:
                    units.append(cur)
                cur = {"text": t, "heading": "", "para_no": "",
                       "para_title": "", "item_no": ""}
            else:
                cur["text"] += "\n" + t
        if cur:
            units.append(cur)
        return units or [{"text": title, "heading": "", "para_no": "",
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
    ap.add_argument("-o", "--output", default="data_chunks/abs_cyber_chunks.jsonl")
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
