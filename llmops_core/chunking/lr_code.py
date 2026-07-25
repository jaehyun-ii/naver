#!/usr/bin/env python3
"""Domain-aware chunker for Lloyd's Register **Codes** (LR-CO-xxx).

Reads a MinerU ``content_list.json`` and emits structured RAG chunks that
preserve the LR Code hierarchy — Chapter > Section > clause (N.M) >
sub-clause (N.M.K) — plus TABLES, FIGURES, terms/abbreviations definitions,
mirroring the "Parent-Child + 표/그림 구조체" strategy of the KR/ABS chunkers
and emitting the **same JSONL schema** so one viewer serves all families.

LR specifics this handles:

  * The **Chapter** is carried in the running document title (L1), e.g.
    "… - Chapter 1 General - Section 1 Introduction"; each exported PDF is one
    chapter fragment. It is parsed from the title, not from body headings.
  * The vector-search **article unit is the clause N.M** (L2 heading), whose
    children are the "N.M.K …" sub-clause paragraphs that follow in the body.
  * In "Terms and definitions" / "Abbreviations" clauses the sub-clause number
    is often promoted to an L2 heading (e.g. "1.11.2 CLAME"); these are folded
    back as definition children of the enclosing clause, not new articles.
  * A definitions-only extract with no headings at all (e.g. LR-CO-003 Grey
    Boat Code) still yields useful chunks: its tables plus paragraph children
    under a single synthetic clause.

  parent   clause-level chunk — full text + children ids (context extension)
  child    the vector-search units — one sub-clause / definition each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/lr_code_chunker.py \
        data/LR_CO_001/LR_CO_001_content_list.json \
        -o data_chunks/lr_co_001_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import interleave_children, merge_figure_fragments, para_of_default, resolve_units
from ._tables import row_retrieval, table_rows


# ── structural markers ──────────────────────────────────────────────────────
# document title carries code id + chapter + (optional) section
# 제목이 code + (Part) + (Chapter/Volume) + (Section) 경로를 담는다
# — LR-CO Code, LR-RU/FR Rules 공통. " - "로 세그먼트 분리 후 밴드별 파싱.
RE_TITLE_CODE = re.compile(r"\b(LR-(?:CO|RU|FR)-\d+)\b")
RE_TITLE_YEAR = re.compile(r"\b([A-Z][a-z]+\s+\d{4})\b")
_TITLE_BANDS = [
    ("part", re.compile(r"^Part\s+([A-Z0-9]+)\s*(.*)$", re.I)),
    ("chapter", re.compile(r"^Chapter\s+([A-Z0-9]+)\s*(.*)$", re.I)),
    ("volume", re.compile(r"^Volume\s+([A-Z0-9]+)\s*(.*)$", re.I)),
    ("section", re.compile(r"^Section\s+([A-Z0-9]+)\s*(.*)$", re.I)),
]
RE_SECTION = re.compile(r"^Section\s+(\d+)\s+(\S.*)$", re.I)     # band inside a chapter
RE_CLAUSE = re.compile(r"^(\d+\.\d+)\s+(\S.*)$")                  # 1.2 Certification  (article)
RE_SUBCLAUSE = re.compile(r"^(\d+\.\d+\.\d+)\s+(\S.*)$")          # 1.1.6 …            (child)
RE_LEADER = re.compile(r"\.{4,}|·{2,}")                           # ToC dot leaders
RE_ENUM = re.compile(r"^(?:\([a-z0-9ivx]+\)|[a-z]\)|\d+\)|[●•▪]|[–-]\s)")
# Regs4ships/OneOcean 재배포 워터마크 — text 타입으로 본문에 섞여(단독 줄 또는 조 본문 끝줄)
# 들어와 임베딩 노이즈가 된다. 아이템 통째가 아니라 **해당 줄만** 제거한다.
RE_WATERMARK = re.compile(
    r"Taken from Regs4ships|Redistributed on behalf of the publisher"
    r"|\bRegs4ships\b|r4s\.oneocean\.com", re.I)


def strip_watermark(text: str) -> str:
    if not text:
        return text
    lines = [ln for ln in text.split("\n") if not RE_WATERMARK.search(ln)]
    return "\n".join(lines).strip()

# ── reference / notation extraction ─────────────────────────────────────────
RE_REFS = [
    ("lr_rule", re.compile(r"(?:LR|Lloyd's Register)\s+(?:Rules?|Code|ShipRight|Naval)[^,.;\n]{0,45}")),
    ("lr_internal", re.compile(r"\b(?:Ch(?:apter)?|Sec(?:tion)?|Table|Figure)\s+\d+(?:\.\d+)*\b", re.I)),
    ("standard", re.compile(r"(?:IEC|IEEE|ISO(?:/IEC)?|EN|BS|API|DIN|ASTM)\s*[\w.\-:]{0,18}")),
    ("convention", re.compile(r"(?:IMO|SOLAS|MARPOL|MODU|COLREG|IACS)\s*[\w.\-/]{0,25}")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "to be submitted", "is to ", "are to ", "should be")
PROCEDURE = ("survey", "certification", "classification procedure", "application",
             "submitted", "approval", "inspection", "testing")
CONDITION = ("if ", "where ", "when ", "unless", "provided that", "in the event")


def classify(text: str, title: str) -> str:
    low = text.lower()
    tl = title.lower()
    if "definition" in tl or "abbreviation" in tl or "terms and" in tl:
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


# ── definition parsing (Terms/Abbreviations clauses) ────────────────────────
# "CLAME  Competent Lifting Appliance …"  or  "Term. Definition"
RE_DEF_ABBR = re.compile(r"^([A-Z][A-Za-z0-9&/\-]{1,12})\s+([A-Z].{3,})$")
RE_DEF_TERM = re.compile(r"^([A-Z][A-Za-z0-9 &/()\-,]{1,60}?)[\.:]\s+(\S.+)$")


def parse_definition(text: str) -> tuple[str | None, str | None]:
    t = text.strip()
    # strip a leading sub-clause number ("1.11.2 CLAME …")
    t = re.sub(r"^\d+(?:\.\d+)+\s+", "", t)
    m = RE_DEF_ABBR.match(t)
    if m and m.group(1).isupper():
        return m.group(1), None
    m = RE_DEF_TERM.match(t)
    if m and len(m.group(1)) <= 55:
        return m.group(1).strip(), None
    return None, None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # English ≈ 4 chars/token


# ── document metadata (parsed from title) ───────────────────────────────────
def parse_title(items: list[dict]) -> dict:
    title = ""
    for x in items:
        if x.get("type") == "text" and x.get("text_level") == 1:
            title = (x.get("text") or "").strip()
            break
    cm = RE_TITLE_CODE.search(title)
    ym = RE_TITLE_YEAR.search(title)
    code, year = (cm.group(1) if cm else ""), (ym.group(1) if ym else "")
    segs = [s.strip() for s in title.split(" - ")]
    name = re.sub(r"^LR-(?:CO|RU|FR)-\d+\s*", "", segs[0] if segs else title)
    name = re.sub(r",\s*[A-Z][a-z]+\s+\d{4}\s*$", "", name).strip(" ,")
    part_no = part_title = chap_no = chap_title = sec_no = sec_title = ""
    for s in segs[1:]:
        for kind, rx in _TITLE_BANDS:
            mm = rx.match(s)
            if not mm:
                continue
            num, ttl = mm.group(1), mm.group(2).strip()
            if kind == "part":
                part_no, part_title = num, ttl
            elif kind in ("chapter", "volume") and not chap_no:
                chap_no, chap_title = num, ttl
            elif kind == "section":
                sec_no, sec_title = num, ttl
            break
    return {
        "code": code,
        "doc_title": (name or title).strip(" ,"),
        "year": year,
        "part_no": part_no, "part_title": part_title,
        "chapter_no": chap_no or "0",
        "chapter_title": chap_title,
        "sec_hint": (sec_no, sec_title),
        "title": title,
    }


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        ti = parse_title(items)
        self.tinfo = ti
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        # 코드 접두로 문서종류 판정: RU=Rules & Regulations, CO=Code, FR=Framework
        pm = re.match(r"LR[-_](RU|CO|FR)", (ti["code"] or stem).upper())
        self.doc_type = {"RU": "rule", "CO": "code",
                         "FR": "framework"}.get(pm.group(1) if pm else "", "code")
        self.doc_meta = {
            "doc_id": (ti["code"] or stem).replace("-", "_").upper(),
            "doc_title": f'{ti["code"]} {ti["doc_title"]}'.strip(),
            "part_no": ti["part_no"], "part_title": ti["part_title"] or ti["doc_title"],
            "publisher": "Lloyd's Register", "year": ti["year"], "language": "en",
        }

    # -- id helpers ----------------------------------------------------------
    def article_id(self, sec_no: str, clause: str | None) -> str:
        base = f'{self.doc_meta["doc_id"]}_C{self.tinfo["chapter_no"]}_S{sec_no or "0"}'
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
        # 일부 LR-RU 문서는 clause(N.M)가 heading이 아니라 본문(text_level None)으로 온다.
        # L2 clause heading이 하나도 없으면 완화 모드로, 섹션번호와 major가 일치하는
        # 십진-선두 본문도 조로 인식(오탐 방지: major==현재 섹션번호 요구).
        self._relaxed = not any(
            x.get("text_level") == 2
            and RE_CLAUSE.match((x.get("text") or "").strip())
            and not RE_SUBCLAUSE.match((x.get("text") or "").strip())
            for x in self.items
        )
        st = {
            "sec": self.tinfo["sec_hint"] if self.tinfo["sec_hint"][0] else ("1", ""),
            "clause": (None, ""),                     # current article (N.M)
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
            if typ == "equation":                 # 수식은 직전 문단에 병합(단독이면 새 조각)
                eq = strip_watermark((x.get("text") or "").strip())
                if eq:
                    if st["pieces"]:
                        st["pieces"][-1]["text"] += "\n" + eq
                        st["pieces"][-1]["pages"].add(page)
                    else:
                        st["pieces"].append({"text": eq, "pages": {page}, "enum": False})
                continue
            if typ == "list":
                for li in x.get("list_items", []):
                    li = strip_watermark((li or "").strip())
                    if li and not RE_LEADER.search(li):
                        st["pieces"].append({"text": li, "pages": {page}, "enum": True})
                continue
            if typ not in ("text", "ref_text"):       # ref_text=오분류된 열거항목 본문
                continue

            raw = strip_watermark((x.get("text") or "").strip())
            if not raw or RE_LEADER.search(raw) or x.get("text_level") == 1:
                continue
            lvl = x.get("text_level")

            # Section band (starts a new section; clause numbering is chapter-global anyway)
            m = RE_SECTION.match(raw)
            if m and lvl == 2:
                flush()
                st["sec"], st["clause"] = (m.group(1), m.group(2).strip()), (None, "")
                continue

            # clause N.M as an article heading (L2). N.M.K promoted to L2 stays a child.
            m = RE_CLAUSE.match(raw)
            if m and lvl == 2 and not RE_SUBCLAUSE.match(raw):
                flush()
                st["clause"] = (m.group(1), m.group(2).strip())
                continue

            # 완화 모드: 본문형 clause(레벨 없음, N.M로 시작, major==섹션번호). 본문 보존.
            if (m and lvl != 2 and self._relaxed and not RE_SUBCLAUSE.match(raw)
                    and m.group(1).split(".")[0] == st["sec"][0]):
                flush()
                first_line = m.group(2).split("\n", 1)[0].strip()
                st["clause"] = (m.group(1), first_line[:60])
                st["pieces"].append({"text": raw, "pages": {page}, "enum": False})
                continue

            # everything else is body — a sub-clause paragraph or continuation
            sub = bool(RE_SUBCLAUSE.match(raw))
            st["pieces"].append({"text": raw, "pages": {page}, "enum": sub})

        flush()
        return self.out

    # -- emit parent + children + tables + figures ---------------------------
    def _flush(self, st: dict):
        sec_no, sec_title = st["sec"]
        clause_no, clause_title = st["clause"]
        pieces, tables = st["pieces"], st["tables"]
        figures = merge_figure_fragments(st["figures"])   # 조각난 복합 그림 병합
        if not (pieces or tables or figures):
            return

        parent_id = self._uniq(self.article_id(sec_no, clause_no))
        ti = self.tinfo
        path = [self.doc_meta["doc_title"]]
        if ti["part_no"]:
            path.append(f'Part {ti["part_no"]} {ti["part_title"]}'.strip())
        if ti["chapter_no"] != "0" or ti["chapter_title"]:
            path.append(f'Chapter {ti["chapter_no"]} {ti["chapter_title"]}'.strip())
        path.append(f"Section {sec_no} {sec_title}".strip())
        if clause_no:
            path.append(f"{clause_no} {clause_title}".strip())

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t, _ in tables}
                       | {f.get("page_idx") for f, _ in figures})
        full_text = "\n".join(pc["text"] for pc in pieces)
        is_def = any(k in clause_title.lower() for k in ("definition", "abbreviation", "terms and"))

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta, "document_type": self.doc_type,
                "chapter_no": ti["chapter_no"], "chapter_title": ti["chapter_title"],
                "section_no": sec_no, "section_title": sec_title,
                "article_no": clause_no, "article_title": clause_title,
                "effective_date": ti["year"], "notations": [],
                "section_path": path, "pages": pages, "source_file": self.source_file,
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
                    rec["term_ko"] = term      # display-term slot (shared schema)
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
                retrieval_text=" ".join(t for t in (cap, clause_title, vis) if t),
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
        # 비-정의: 상대뎁스 리졸버에 위임 — 하위조항 N.M.K=항 > a. b.=호 > i. ii.=목
        # (para_of_default가 N.M.K를 항으로, 마침표형 열거 a./i.는 등장순서로 호/목 분리).
        if not is_def:
            return resolve_units(pieces, para_of_default)
        # 정의: 각 정의 항목이 개별 유닛
        units: list[dict] = []
        cur: dict | None = None

        def blank(t, pgs=None, si=0):
            return {"text": t, "heading": "", "para_no": "", "para_title": "",
                    "item_no": "", "pages": set(pgs or ()), "src_idx": si}

        for _si, pc in enumerate(pieces):
            t = pc["text"]
            if cur is None or parse_definition(t)[0]:
                if cur:
                    units.append(cur)
                cur = blank(t, pc.get("pages"), _si)
            else:
                cur["text"] += "\n" + t
                cur["pages"] |= set(pc.get("pages") or ())
        if cur:
            units.append(cur)
        return units or [blank(pieces[0]["text"] if pieces else "",
                               pieces[0].get("pages") if pieces else None)]

    @staticmethod
    def _table_text(caption: str, html: str) -> str:
        cells = re.sub(r"<[^>]+>", " ", html)
        cells = cells.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")   # 부등호·기호 보존
        return re.sub(r"\s+", " ", (caption + " " + cells)).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content_list")
    ap.add_argument("-o", "--output", default="data_chunks/lr_code_chunks.jsonl")
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
