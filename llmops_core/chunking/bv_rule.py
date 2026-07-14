#!/usr/bin/env python3
"""Domain-aware chunker for **Bureau Veritas (BV)** Rules & Rule Notes (NR/NI).

Reads a MinerU ``content_list.json`` and emits structured RAG chunks preserving
the BV hierarchy — Part > Chapter > Section > Article (n / [n]) > sub-article
(n.m / n.m.k) — plus TABLES, FIGURES, equations and definitions, using the
"Parent-Child + 표/그림 구조체" strategy and emitting the **same JSONL schema**
as the KR/ABS/LR/NK/DNV/IACS chunkers so one viewer serves all families.

BV specifics this handles:

  * BV structures documents as **Part / Chapter / Section**; a Rule Note (NR)
    is typically Section-only, a consolidated rule adds Part & Chapter bands.
    All three are tracked and clause numbering restarts on every Section.
  * BV traditionally prints the article number in **brackets** ("[1] General",
    "[1.1] …"); the layout model may render either "[1.1]" or a plain "1.1", so
    both forms are accepted and normalised to the same id.
  * The cover carries the **NR/NI id and revision** (e.g. "NR206 … R03 March
    2025"), parsed into the doc metadata.
  * Front matter (address, foreword) is skipped until the first Section band.

  parent   article-level chunk — full text + children ids (context extension)
  child    the vector-search units — one paragraph / enumerated item each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/bv_rule_chunker.py \
        data/BV_206NR/BV_206NR_content_list.json \
        -o data_chunks/bv_206nr_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import resolve_units, para_of_default
from ._tables import row_retrieval, table_rows


# ── structural markers ──────────────────────────────────────────────────────
# 대소문자 무시 — BV 문서마다 "Section 1"(신형)·"SECTION 1"(2013 등 구형) 혼재
RE_PART = re.compile(r"^Part\s+([A-Z0-9]+)\s+(\S.*)$", re.I)
RE_CHAPTER = re.compile(r"^Chapter\s+(\d+)\s+(\S.*)$", re.I)
RE_SECTION = re.compile(r"^Section\s+(\d+)\s+(\S.*)$", re.I)
# article: "[1.1] Title" or "1.1 Title"  (bracket optional)
RE_CLAUSE = re.compile(r"^\[?(\d+(?:\.\d+)*)\]?\s+(\S.*)$")
RE_LEADER = re.compile(r"\.{4,}|·{2,}")
RE_ENUM = re.compile(r"^(?:\([a-z0-9ivx]+\)|[a-z]\)|\d+\)|[●•▪]|[–-]\s|—\s)")
# 주요 항목((a)/(1)) + 십진 조항(1.1) — dash·불릿만 제외(붕괴된 조항도 분할되게)
RE_PRIMARY = re.compile(r"^(?:\([a-z0-9ivx]{1,4}\)|[a-z]\)|\d+\))\s")
RE_CLAUSE_ITEM = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")  # 번호 뒤 공백+텍스트만(수식 제외)
# 목차 표 감지: NR류 문서는 앞 수십 페이지 목차가 표로 추출됨(캡션이 Section/Chapter
# 제목·"Table of Content", 셀 다수가 페이지번호 꼬리). 본문 표 캡션은 "Table N …" 형식.
RE_TOC_CAP = re.compile(r"^(?:table of contents?|section\s+\d+|chapter\s+\d+|appendix\s+\d+)\b", re.I)
RE_TOC_CELL_TAG = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.I | re.S)
RE_TOC_PAGE_TAIL = re.compile(r"\d{1,4}\s*$")


def is_toc_table(caption: str, html: str) -> bool:
    if not RE_TOC_CAP.match((caption or "").strip()):
        return False
    cells = [re.sub(r"<[^>]+>", " ", c).strip() for c in RE_TOC_CELL_TAG.findall(html or "")]
    if not cells:
        return True                      # 캡션만 목차형이고 셀 없음(빈/이미지형) — 목차
    tails = sum(1 for c in cells if c and RE_TOC_PAGE_TAIL.search(c))
    return tails >= 2                    # 본문 표 캡션은 "Table N :" — 오폭 없음(실측 0)


RE_NRID = re.compile(r"\b(N[RI]\d{3,4})\b")
RE_REV = re.compile(r"\b(R\d+)\b")
RE_DATE = re.compile(r"\b([A-Z][a-z]+\s+\d{4})\b")

# ── reference extraction ────────────────────────────────────────────────────
RE_REFS = [
    ("bv_rule", re.compile(r"\bN[RI]\d{3,4}\b[^,.;\n]{0,40}")),
    ("bv_internal", re.compile(r"\b(?:Part|Ch(?:apter)?|Sec(?:tion)?|Article|Table|Fig(?:ure)?|App(?:endix)?)\s*\.?\s*[A-Z]?\d+(?:\.\d+)*\b", re.I)),
    ("standard", re.compile(r"(?:IEC|IEEE|ISO(?:/IEC)?|EN|API|ISfWEC)\s*[\w.\-:]{0,18}")),
    ("convention", re.compile(r"(?:IMO|SOLAS|MARPOL|IACS|MSC|Load Line)\s*[\w.\-/()]{0,25}")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "to be", "should be", "are to comply")
PROCEDURE = ("survey", "classification", "approval", "certification", "test",
             "assessment", "documentation", "review", "inspection", "trial")
CONDITION = ("if ", "where ", "when ", "unless", "provided that", "in case")


def classify(text: str, title: str) -> str:
    low = text.lower()
    if "definition" in title.lower() or "abbreviation" in title.lower() \
            or "symbol" in title.lower():
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


RE_DEF = re.compile(r"^([A-Z][A-Za-z0-9 &/()\-,]{1,55}?)\s*[:=]\s+(\S.+)$")


def parse_definition(text: str) -> tuple[str | None, str | None]:
    m = RE_DEF.match(text.strip())
    if m and len(m.group(1)) <= 50:
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
    nrid = rev = date = title = ""
    for x in items[:40]:
        if x.get("type") != "text":
            continue
        t = (x.get("text") or "").strip()
        if not nrid:
            m = RE_NRID.search(t)
            if m:
                nrid = m.group(1)
        if not rev:
            m = RE_REV.search(t)
            if m:
                rev = m.group(1)
        if not date:
            m = RE_DATE.search(t)
            if m and "20" in m.group(1):
                date = m.group(1)
        if not title and x.get("text_level") == 1 and t.isupper() and len(t) > 8 \
                and not RE_NRID.search(t) and "BUREAU" not in t and "RULE" not in t:
            title = t.title()
    return {"nrid": nrid, "rev": rev, "date": date, "title": title}


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        cov = parse_cover(items)
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        self.nrid = cov["nrid"] or stem
        # NR=Rule Note(규칙), NI=Guidance Note(지침) — 본문도 "this Guidance Note"로 자기선언.
        # 표지 NRID가 없으면 파일명 토큰("617-NI_2018-01" / "ni641_oct2019")으로 판정.
        toks = re.split(r"[-_ ]+", stem.upper())
        is_ni = (cov["nrid"] or "").upper().startswith("NI") \
            or any(t == "NI" or re.fullmatch(r"NI\d{3,4}", t) for t in toks)
        self.doc_type = "guidance" if is_ni else "rule"
        self.doc_meta = {
            "doc_id": (cov["nrid"] or stem).replace("-", "_").upper(),
            "doc_title": f'{cov["nrid"]} {cov["title"]}'.strip() or stem,
            "part_no": "", "part_title": cov["title"], "publisher": "Bureau Veritas (BV)",
            "year": (cov["date"].split()[-1] if cov["date"] else "2025"),
            "revision": cov["rev"], "edition": cov["date"], "language": "en",
        }

    def article_id(self, part: str, chap: str, sec: str, clause: str | None) -> str:
        base = self.doc_meta["doc_id"]
        if part:
            base += f"_P{part}"
        if chap:
            base += f"_C{chap}"
        base += f"_S{sec or '0'}"
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
        # 밴드(Part/Chapter/Section)가 전혀 없는 문서(구형 NI 등, "1. TITLE"만)는
        # 처음부터 clause를 캡처하도록 started=True로 시작
        has_band = any(
            x.get("type") == "text" and x.get("text_level") in (1, 2)
            and (RE_PART.match((x.get("text") or "").strip())
                 or RE_CHAPTER.match((x.get("text") or "").strip())
                 or RE_SECTION.match((x.get("text") or "").strip()))
            for x in self.items)
        # 목차 페이지(러닝헤더 "Table of Content") — 캡션 없는 목차 표 드랍용
        toc_pages = {
            x.get("page_idx") for x in self.items
            if x.get("type") == "header" and "table of content" in (x.get("text") or "").lower()
        }
        st = {
            "part": ("", ""), "chap": ("", ""), "sec": ("0", ""),
            "grp": ("", ""), "art": (None, ""),
            "art_pages": set(),             # 조 헤딩이 나온 페이지(빈 leaf 조 승격용)
            "heads": {},                    # 섹션 내 조상 헤딩 제목(예: "2.2" → 그룹 제목)
            "pieces": [], "tables": [], "figures": [],
            "started": not has_band,
        }

        def flush(next_ano: str | None = None):
            # MinerU가 한 문장짜리 조 본문을 L2 헤딩으로 승격하는 경우: 하위 조가
            # 따라오지 않는(leaf) 빈 조는 제목이 곧 본문이므로 제목을 본문으로 승격해
            # 유실을 막는다. (하위 조가 있는 그룹 헤딩은 heads 경로로만 보존.)
            ano, atitle = st["art"]
            if (ano and atitle and not (st["pieces"] or st["tables"] or st["figures"])
                    and not (next_ano or "").startswith(ano + ".")):
                st["pieces"].append({"text": f"{ano} {atitle}", "pages": set(st["art_pages"]), "enum": False})
            self._flush(st)
            st["pieces"], st["tables"], st["figures"] = [], [], []

        for x in self.items:
            typ = x.get("type")
            page = x.get("page_idx")

            if typ in ("header", "footer", "page_number", "page_footnote"):
                continue

            raw = (x.get("text") or "").strip()

            # bands (Part / Chapter / Section) — only real headings (L1/L2),
            # never the plain-text ToC section list (which is level-less)
            if typ == "text" and raw and x.get("text_level") in (1, 2) and not RE_LEADER.search(raw):
                mp, mc, msec = RE_PART.match(raw), RE_CHAPTER.match(raw), RE_SECTION.match(raw)
                if mp:
                    flush(); st["started"] = True
                    st["part"] = (mp.group(1), mp.group(2).strip())
                    st["chap"], st["sec"] = ("", ""), ("0", "")
                    st["grp"], st["art"] = ("", ""), (None, ""); st["heads"] = {}; continue
                if mc:
                    flush(); st["started"] = True
                    st["chap"] = (mc.group(1), mc.group(2).strip())
                    st["sec"] = ("0", ""); st["grp"], st["art"] = ("", ""), (None, ""); st["heads"] = {}; continue
                if msec:
                    # ignore the leading ToC list of sections (before real body)
                    flush(); st["started"] = True
                    st["sec"] = (msec.group(1), msec.group(2).strip())
                    st["grp"], st["art"] = ("", ""), (None, ""); st["heads"] = {}; continue

            if not st["started"]:
                continue

            if typ == "table":
                cap = " ".join(x.get("table_caption") or [])
                if not (is_toc_table(cap, x.get("table_body", ""))
                        or (not cap.strip() and page in toc_pages)):
                    st["tables"].append(x)
                continue
            if typ == "image":
                st["figures"].append(x)
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

            # article clause ([n] or n / n.m / n.m.k) at L2
            m = RE_CLAUSE.match(raw)
            if m and x.get("text_level") == 2:
                ano, title = m.group(1), m.group(2).strip()
                flush(next_ano=ano)
                st["art"] = (ano, title)
                st["art_pages"] = {page} if page is not None else set()
                st["heads"][ano] = title       # 본문 없는 중간 그룹(예: 2.2)도 하위 조 경로에 보존
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
        part_no, part_title = st["part"]
        chap_no, chap_title = st["chap"]
        sec_no, sec_title = st["sec"]
        grp_no, grp_title = st["grp"]
        ano, atitle = st["art"]
        pieces, tables, figures = st["pieces"], st["tables"], st["figures"]
        if not (pieces or tables or figures):
            return

        parent_id = self._uniq(self.article_id(part_no, chap_no, sec_no, ano))
        path = [self.doc_meta["doc_title"]]
        if part_no:
            path.append(f"Part {part_no} {part_title}".strip())
        if chap_no:
            path.append(f"Chapter {chap_no} {chap_title}".strip())
        path.append(f"Section {sec_no} {sec_title}".strip())
        heads = st.get("heads", {})
        if ano:
            segs_ = ano.split(".")
            for d in range(1, len(segs_)):     # 조상 경로 전부(예: 2.2.1 → "2 …", "2.2 …")
                pref = ".".join(segs_[:d])
                ptitle = heads.get(pref) or (grp_title if pref == grp_no else "")
                if ptitle or pref == grp_no:
                    path.append(f"{pref} {ptitle}".strip())
            path.append(f"{ano} {atitle}".strip())
        elif grp_no:
            path.append(f"{grp_no} {grp_title}".strip())

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t in tables}
                       | {f.get("page_idx") for f in figures})
        full_text = "\n".join(pc["text"] for pc in pieces)
        is_def = any(k in atitle.lower() for k in ("definition", "abbreviation", "symbol"))

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta, "document_type": self.doc_type,
                "part_seg_no": part_no, "part_seg_title": part_title,
                "chapter_no": chap_no or sec_no, "chapter_title": chap_title or sec_title,
                "section_no": grp_no or ano, "section_title": grp_title,
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
        for k, tb in enumerate(tables, 1):
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
        self.out.append(parent)
        self.out.extend(child_chunks)
        self.out.extend(table_chunks)
        self.out.extend(figure_chunks)

    @staticmethod
    def _split_children(pieces: list[dict], is_def: bool) -> list[dict]:
        # 비-정의: 상대뎁스 리졸버(항>호>목, 혼재 대응)에 위임.
        if not is_def:
            return resolve_units(pieces, para_of_default)
        # 정의: 각 정의 항목이 개별 유닛
        units: list[dict] = []
        cur: dict | None = None
        for pc in pieces:
            t = pc["text"]
            if cur is None or parse_definition(t)[0]:
                if cur:
                    units.append(cur)
                cur = {"text": t, "heading": "", "para_no": "",
                       "para_title": "", "item_no": ""}
            else:
                cur["text"] += "\n" + t
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
    ap.add_argument("-o", "--output", default="data_chunks/bv_rule_chunks.jsonl")
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
