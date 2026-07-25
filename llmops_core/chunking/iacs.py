#!/usr/bin/env python3
"""Domain-aware chunker for **IACS** publications — Recommendations (Rec. No. N)
and, by auto-detected mode, the Common Structural Rules (CSR).

Reads a MinerU ``content_list.json`` and emits structured RAG chunks preserving
the IACS numbered hierarchy — top-level clause (N) > sub-clause (N.M / N.M.K) —
plus TABLES, FIGURES and definitions, using the "Parent-Child + 표/그림 구조체"
strategy and emitting the **same JSONL schema** as the KR/ABS/LR/NK/DNV chunkers
so one viewer serves all families.

IACS Recommendation specifics this handles:

  * A recommendation opens with a section **overview list** (the six top-level
    clauses printed together) before the body restarts at clause 1; those
    overview headings carry no body and are dropped automatically (an empty
    article emits nothing).
  * The running document header — "<rec-no> (cont)" — is mis-parsed by the
    layout model as a clause; it is filtered by matching the recommendation
    number.
  * Year-only "headings" (e.g. a 1991…1998 timeline under "Actions Taken By
    IACS") are treated as body enumerations, not new articles.

  parent   clause-level chunk — full text + children ids (context extension)
  child    the vector-search units — one paragraph / enumerated item each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/iacs_chunker.py \
        data/IACS_REC_46/IACS_REC_46_content_list.json \
        -o data_chunks/iacs_rec_46_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import interleave_children, merge_figure_fragments, resolve_units, para_of_default
from ._tables import row_retrieval, table_rows


# ── structural markers ──────────────────────────────────────────────────────
# REC: 1 / 1.2 / 2.2.1   |   UR·UI·PR: G1.1 / G1.2.1 / E7.2 / S11.3 / FTP5.1
# (letter-prefixed code = the UR/UI id; optional 0–4 letter prefix on the first token)
RE_CLAUSE = re.compile(r"^([A-Z]{0,4}\d+(?:\.\d+)*)\.?\s+(\S.*)$")
# UI(통일해석) 구조: "Regulation 28.3 …" / "Paragraph 7.5.1 …" 를 조로 인식
RE_REG = re.compile(r"^(?:Regulations?|Reg\.?|Paragraphs?|Para\.?)\s+([IVXLC0-9][\w./\-]*)\s*(.*)$", re.I)
# UI는 번호 없는 명명 섹션(Interpretation/Arrangement/Approval …)을 헤딩으로 쓴다 → 조로 인식
RE_UI_SECTION = re.compile(
    r"^(?:Regulations?|Interpretation|Arrangements?|Application|Scope|Approval|"
    r"Notes?|General|Definitions?|Requirements?|Background|Introduction|Purpose|"
    r"Recommendation|Guidance|Additional\b.*|Design\b.*|Testing\b.*)\s*[:.]?\s*$", re.I)
RE_RECNO = re.compile(r"\bNo\.?\s*(\d+)\b")
RE_REV = re.compile(r"\(Rev\.?\s*\d+[^)]*\)|\((?:\d{4}|[A-Z][a-z]+\s+\d{4})\)")
RE_LEADER = re.compile(r"\.{4,}|·{2,}")
RE_ENUM = re.compile(r"^(?:\([a-z0-9ivx]+\)|[a-z]\)|\d+\)|[●•▪]|[–-]\s|—\s)")
# 주요 열거 항목(a)/(1)/(i)) + 십진 조항(1.1, 3. Title) — dash·불릿 하위항목만 제외.
# list로 붕괴된 조항 본문("1.1 …", "2.1 …")도 개별 청크로 나뉘게 한다.
RE_PRIMARY = re.compile(r"^(?:\([a-z0-9ivx]{1,4}\)|[a-z]\)|\d+\))\s")
RE_CLAUSE_ITEM = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")  # 번호 뒤 공백+텍스트만(수식 제외)
RE_CONT_NOISE = re.compile(r"^(?:No\.?\s*\d*\s*)?\(?\s*cont\.?\s*\)?$|^No\.$|^\d+$", re.I)
# 개정이력 조각("(1975)","(Rev.1","1990)","June 2000)","Feb 2021)")·문서끝 표식 노이즈
RE_DATE_NOISE = re.compile(
    r"^\(?\s*(?:Rev\.?\s*\d*|Corr\.?\s*\d*|Add\.?\s*\d*|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*(?:19|20)?\d{2}|"
    r"(?:19|20)\d{2})\s*\)?\s*$", re.I)
RE_END_DOC = re.compile(r"^End of Document\.?\s*$", re.I)
RE_YEAR = re.compile(r"^(19|20)\d{2}$")
# CSR band markers (Common Structural Rules): "Chapter N", "Section N", "Part N"
RE_CSR_PART = re.compile(r"^PART\s+(\d+)\s+(\S.*)$", re.I)
RE_CSR_CHAPTER = re.compile(r"^CHAPTER\s+(\d+)\s+(\S.*)$", re.I)
RE_CSR_SECTION = re.compile(r"^SECTION\s+(\d+)\s+(\S.*)$", re.I)

# ── reference extraction ────────────────────────────────────────────────────
RE_REFS = [
    ("iacs_rule", re.compile(r"IACS\s+(?:UR|UI|PR|Rec(?:ommendation)?\.?)\s*[A-Z]?\d*", re.I)),
    ("iacs_rule", re.compile(r"\bU[RI]\s+[A-Z]{1,2}\d+[a-z]?\b")),   # 접두 생략형 "UR W2"
    ("iacs_internal", re.compile(r"\b(?:Section|Chapter|Part|Table|Figure|Annex)\s+\d+(?:\.\d+)*\b", re.I)),
    ("standard", re.compile(r"(?:IEC|IEEE|ISO(?:/IEC)?|ASTM|EN)\s*[\w.\-:]{0,18}")),
    ("convention", re.compile(r"(?:IMO|SOLAS|MARPOL|Load Line|ILLC|MSC)\s*[\w.\-/()]{0,25}")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "should be", "recommended", "it is important")
PROCEDURE = ("loading", "unloading", "survey", "inspection", "operation",
             "procedure", "monitoring", "planning", "measurement")
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


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def detect(items: list[dict]) -> dict:
    """Recommendation number/title + CSR-mode detection from the first pages."""
    rec_no, title, is_csr = "", "", False
    for x in items[:60]:
        if x.get("type") != "text":
            continue
        t = (x.get("text") or "").strip()
        if RE_CSR_PART.match(t) or "common structural rules" in t.lower():
            is_csr = True
        if not rec_no:
            m = RE_RECNO.search(t)
            if m:
                rec_no = m.group(1)
        if not title and x.get("text_level") == 1 and len(t) > 15 \
                and not RE_RECNO.fullmatch(t):
            title = RE_REV.sub("", t)
            # 러닝헤더 조각 제거: "No. 73 (cont)", "52 (cont)", "(cont)". 코드 숫자(G1) 보존
            title = re.sub(r"\s*\bNo\.?\s*\d+\s*\(\s*cont\.?\s*\)"
                           r"|(?<![A-Za-z])\s*\d+\s*\(\s*cont\.?\s*\)|\(\s*cont\.?\s*\)",
                           " ", title, flags=re.I)
            title = re.sub(r"\s{2,}", " ", title).strip(" ,")
    return {"rec_no": rec_no, "title": title, "is_csr": is_csr}


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        info = detect(items)
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        self.rec_no = info["rec_no"] or re.sub(r"\D", "", stem) or "0"
        self.is_csr = info["is_csr"]
        self.doc_code = stem.split("_")[-1]          # "FTP6" / "E7" / "52" (러닝헤더 식별용)
        did = stem.replace("-", "_").upper()
        self.doc_meta = {
            "doc_id": did,
            "doc_title": info["title"] or f"IACS Rec. No. {self.rec_no}",
            "part_no": "", "part_title": info["title"],
            "publisher": "IACS", "year": "2024",
            # UR=Unified Requirement(강제 요건), UI=Unified Interpretation, REC=Recommendation
            "document_family": "csr" if self.is_csr else {
                "UR": "requirement", "UI": "interpretation", "REC": "recommendation",
                "PR": "procedural-requirement",
            }.get(stem.split("_")[0].upper(), "recommendation"),
            "language": "en",
        }

    def article_id(self, band: str, clause: str | None) -> str:
        pre = f"IACS_{self.doc_meta['doc_id']}"
        base = f"{pre}_{band}" if band else pre
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

    def _is_running_header(self, raw: str) -> bool:
        # 이 문서 고유 코드/번호 단독 헤더만 제거("FTP6", "No. 52", "73 (cont)").
        # 문서 코드로만 판별해 강판등급("A2","D36") 등 실제 내용은 건드리지 않는다.
        cont = r"(?:\s*\(\s*cont\.?\s*\)?)?"
        for key in (re.escape(self.doc_code), re.escape(self.rec_no)):
            if re.fullmatch(rf"(?:No\.?\s*)?{key}{cont}", raw.strip(), re.I):
                return True
        return False

    def _clean_piece(self, raw: str) -> str:
        # 본문에 삽입된 러닝헤더 조각 제거: "No. 73 (cont)", "52 (cont)", 단독 "(cont)"
        raw = re.sub(r"\s*\bNo\.?\s*\d+\s*\(\s*cont\.?\s*\)", " ", raw, flags=re.I)
        raw = re.sub(r"(?<![A-Za-z])\s*\d+\s*\(\s*cont\.?\s*\)", " ", raw, flags=re.I)
        raw = re.sub(r"\(\s*cont\.?\s*\)", " ", raw, flags=re.I)
        return re.sub(r"[ \t]{2,}", " ", raw).strip()

    def _is_noise_heading(self, num: str, title: str) -> bool:
        # running "<code> (cont)" header (REC number or UR/UI code), bare year, or
        # a bare code with no title (e.g. "G1" / "FTP5" repeated as a page header)
        if not title or title.startswith("("):     # 빈 제목·"(cont)"·"(June 2010)" 등
            return True
        if RE_YEAR.match(num) and not title:
            return True
        return False

    def run(self) -> list[dict]:
        st = {
            "band": ("", ""),                 # CSR Chapter/Section (empty for Rec)
            "grp": ("", ""),                  # top-level clause N
            "art": (None, ""),
            "pieces": [], "tables": [], "figures": [],
        }

        def flush():
            self._flush(st)
            st["pieces"], st["tables"], st["figures"] = [], [], []

        for x in self.items:
            typ = x.get("type")
            page = x.get("page_idx")

            if typ in ("header", "footer", "page_number", "page_footnote", "aside_text"):
                continue
            if typ == "table":
                st["tables"].append((x, len(st["pieces"])))
                continue
            if typ in ("image", "chart"):
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
            if typ not in ("text", "ref_text"):       # ref_text=오분류된 열거항목 본문
                continue

            raw = (x.get("text") or "").strip()
            if not raw or RE_LEADER.search(raw):
                continue

            # CSR band headers (Chapter/Section) reset clause numbering
            if self.is_csr:
                m = RE_CSR_CHAPTER.match(raw) or RE_CSR_SECTION.match(raw)
                if m:
                    flush()
                    st["band"] = (m.group(1), m.group(2).strip())
                    st["grp"], st["art"] = ("", ""), (None, "")
                    continue

            m = RE_REG.match(raw)                    # UI: Regulation/Paragraph = 조
            if m and x.get("text_level") == 2:
                flush()
                st["art"] = (m.group(1), raw[:80].strip())
                st["grp"] = ("", "")
                continue

            m = RE_CLAUSE.match(raw)
            if m and x.get("text_level") == 2:
                num, title = m.group(1), m.group(2).strip()
                if self._is_noise_heading(num, title):
                    continue
                if RE_YEAR.match(num):          # year timeline → body, not a clause
                    st["pieces"].append({"text": raw, "pages": {page}, "enum": True})
                    continue
                flush()
                st["art"] = (num, title)
                grp = num.split(".")[0]
                if "." not in num:
                    st["grp"] = (grp, title)
                elif st["grp"][0] != grp:
                    st["grp"] = (grp, "")
                continue

            if x.get("text_level") == 2 and RE_UI_SECTION.match(raw) and len(raw) < 60:
                flush()                              # UI 명명 섹션 = 조
                slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-")[:28]
                st["art"] = (slug, raw.strip().rstrip(":.")[:70])
                st["grp"] = ("", "")
                continue

            if (RE_CONT_NOISE.match(raw) or RE_DATE_NOISE.match(raw)
                    or RE_END_DOC.match(raw) or self._is_running_header(raw)):
                continue                            # 러닝헤더·개정이력·문서끝·문서코드 노이즈
            raw = self._clean_piece(raw)            # 본문 삽입 러닝헤더 조각 제거
            if not raw:
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

        parent_id = self._uniq(self.article_id(f"S{band_no}" if band_no else "", ano))
        path = [self.doc_meta["doc_title"]]
        if band_no:
            path.append(f"Section {band_no} {band_title}".strip())
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
                **self.doc_meta, "document_type": self.doc_meta["document_family"],
                "chapter_no": band_no, "chapter_title": band_title,
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
                pages=sorted(unit["pages"]) if unit.get("pages") else pages,  # 유닛 실제 페이지
                chunk_id=cid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type="text",
                local_heading=unit.get("heading", ""),
                paragraph_no=unit.get("para_no", ""),        # 소속 하위조항 번호
                paragraph_title=unit.get("para_title", ""),  # 소속 하위조항 제목
                item_no=unit.get("item_no", ""),             # 자신의 항목(a)/(1)
                sub_item_no=unit.get("sub_item_no", ""),
                content=unit["text"], summary="", topic="",
                keywords=[], entities=[],
                references=extract_refs(unit["text"]),
                previous_chunk_id=None, next_chunk_id=None,
                linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
            )
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
        # 상대뎁스 리졸버(항>호>목, 혼재 대응). "Note:"·각주는 리졸버가 공통 처리.
        return resolve_units(pieces, para_of_default)

    @staticmethod
    def _table_text(caption: str, html: str) -> str:
        cells = re.sub(r"<[^>]+>", " ", html)
        cells = cells.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")   # 부등호·기호 보존
        return re.sub(r"\s+", " ", (caption + " " + cells)).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content_list")
    ap.add_argument("-o", "--output", default="data_chunks/iacs_chunks.jsonl")
    args = ap.parse_args()

    items = json.loads(Path(args.content_list).read_text(encoding="utf-8"))
    chunker = Chunker(items, Path(args.content_list).name)
    chunks = chunker.run()

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    lv = Counter(c["chunk_level"] for c in chunks)
    ty = Counter(c["chunk_type"] for c in chunks)
    print(f"wrote {len(chunks)} chunks -> {op}  (mode={'CSR' if chunker.is_csr else 'Recommendation'})")
    print(f"  levels : {dict(lv)}")
    print(f"  types  : {dict(ty)}")


if __name__ == "__main__":
    main()
