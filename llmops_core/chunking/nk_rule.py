#!/usr/bin/env python3
"""Domain-aware chunker for **ClassNK (Nippon Kaiji Kyokai)** rules & regulations.

Reads a MinerU ``content_list.json`` and emits structured RAG chunks preserving
the ClassNK hierarchy — Regulations/Guidance segment > Chapter > clause (N.M) >
sub-clause (N.M.K) > enumerated items — plus TABLES, FIGURES and definitions,
using the "Parent-Child + 표/그림 구조체 + 규정-지침 참조" strategy and emitting
the **same JSONL schema** as the KR/ABS/LR chunkers so one viewer serves all.

ClassNK specifics this handles:

  * Like the KR rulebook, ClassNK documents pair **REGULATIONS/RULES** with a
    parallel **GUIDANCE** section (and optional **Appendix**); the running L1
    band header switches the segment. Regulation and guidance articles that
    share the same (Chapter, N.M[.K]) path are cross-linked.
  * Numbering is decimal: **Chapter N** (band) > **N.M** clause (titled L2) >
    **N.M.K** sub-clause (titled L2). The finest titled heading is the article
    (vector-context) unit; its enclosing **N.M** is the section group.
  * Article bodies enumerate with bare "1 …", "2 …" or "(1) …" items — each is
    split into its own child 의미 단위.
  * A trailing "*" on a title is ClassNK's "see guidance" marker; it is stripped
    from the title and recorded as ``guidance_marked``.

  parent   article-level chunk — full text + children ids (context extension)
  child    the vector-search units — one enumerated item / paragraph each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Stdlib only. ``summary`` / ``topic`` and figure ``ocr_text`` stay empty for the
existing ``enrich_chunks.py`` pass.

    python scripts/nk_rule_chunker.py \
        data/NK_registry/NK_registry_content_list.json \
        -o data_chunks/nk_registry_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ._hierarchy import resolve_units, para_of_default
from ._tables import row_retrieval, table_rows


# ── structural markers ──────────────────────────────────────────────────────
RE_CHAPTER = re.compile(r"^Chapter\s+(\d+)\s+(\S.*)$", re.I)
RE_ARTICLE = re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s+(\S.*)$")   # 2.1 / 2.1.3  + title
# 지침부 문자 접두 번호(Part K Guidance: "K1 GENERAL" / "K1.1.1 Application").
# 숫자부를 취해 규칙부(Chapter 1 / 1.1.1)와 같은 좌표계로 맞춘다 — cross-link 키 일치.
RE_CHAPTER_LTR = re.compile(r"^([A-Z]{1,2})(\d+[A-Z]?)\s+(\S.*)$")
RE_ARTICLE_LTR = re.compile(r"^([A-Z]{1,2})(\d+[A-Z]?\.\d+(?:\.\d+)?)\s+(\S.*)$")
RE_APPENDIX = re.compile(r"^Appendix\b\s*(\S.*)$", re.I)
RE_LEADER = re.compile(r"\.{4,}|·{2,}")                       # ToC dot leaders
RE_ENUM_ITEM = re.compile(r"^(?:\(\d{1,2}\)|\d{1,2})\s+\S")    # "1 …" / "(1) …"
RE_PARA_NUM = re.compile(r"^\d{1,2}\s+\S")                     # 항: 맨앞 번호 "1 …" (상위)
RE_ITEM_PAREN = re.compile(r"^\(\d{1,2}\)\s+\S")               # 호: "(1) …" (항 아래 중첩)
RE_CLAUSE_ITEM = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")  # 붕괴 십진 조항(번호 뒤 공백+텍스트, 수식 제외)
RE_ENUM_ALPHA = re.compile(r"^(?:\([a-z]\)|[a-z]\))\s+\S")

# segment (band) L1 headers
RE_SEG_RULE = re.compile(r"^(?:REGULATIONS|RULES)\s+FOR\b", re.I)
RE_SEG_GUIDE = re.compile(r"^GUIDANCE\s+FOR\b", re.I)
RE_SEG_APPX = re.compile(r"^Appendix\b", re.I)

# ── reference extraction ────────────────────────────────────────────────────
RE_REFS = [
    ("nk_rule", re.compile(r"(?:ClassNK|Nippon Kaiji|NK)\s+(?:Rules?|Guidance|Regulations?)[^,.;\n]{0,40}")),
    ("nk_internal", re.compile(r"\b(?:Chapter|Part|Annex|Appendix|Table|Fig(?:ure)?)\s+\d+(?:\.\d+)*\b", re.I)),
    # 규칙↔지침 상호인용("5.2 of the Rules" / "K1.1.1 of the Guidance") 및 문자 Part
    ("nk_cross", re.compile(
        r"\b[A-Z]{0,2}\d+(?:\.\d+){1,3}(?:\([0-9a-z]+\))*(?:\s*(?:and|to)\s*"
        r"[A-Z]{0,2}\d+(?:\.\d+){1,3})?\s*(?:above|below)?\s+of\s+the\s+(?:Rules|Guidance)\b", re.I)),
    ("nk_internal", re.compile(r"\bPart\s+[A-Z]{1,2}\b")),
    ("standard", re.compile(r"(?:IACS|IEC|ISO(?:/IEC)?|IEEE|ITU)\s*[\w.\-:]{0,18}")),
    ("convention", re.compile(r"(?:IMO|SOLAS|MARPOL|MLC|STCW|ILO|Load Line)\s*[\w.\-/]{0,25}")),
]

REQUIREMENT = ("is to be", "are to be", "shall", "must ", "is required",
               "to be submitted", "is to ", "are to ", "will be")
PROCEDURE = ("survey", "registration", "classification", "application", "certificate",
             "inspection", "approval", "audit", "renewal", "re-issuance")
CONDITION = ("if ", "where ", "when ", "unless", "provided that", "in case")


def strip_star(title: str) -> tuple[str, bool]:
    marked = title.rstrip().endswith("*")
    return title.rstrip().rstrip("*").rstrip(), marked


def classify(text: str, title: str) -> str:
    low = text.lower()
    tl = title.lower()
    if "definition" in tl:
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


# ClassNK definition lines (broadened, Rule ③):
#   '"term" means …' / 'Term  means …' / 'Term is defined as …' / 'Term: definition'
#   '(1) "term" means …' (enumerated) / 'Term shall mean …'
RE_DEF_MEANS = re.compile(
    r'^(?:\(\d+\)\s*)?[“"]?([A-Z][A-Za-z0-9 &/()\-,]{1,60}?)[”"]?\s+'
    r'(?:means|shall mean|is defined as|refers to|is taken to mean)\b', re.I)
RE_DEF_COLON = re.compile(r'^[“"]?([A-Z][A-Za-z0-9 &/()\-]{1,45}?)[”"]?\s*[:：]\s+(\S.+)$')


def parse_definition(text: str) -> tuple[str | None, str | None]:
    t = text.strip()
    m = RE_DEF_MEANS.match(t)
    if m and len(m.group(1)) <= 55:
        return m.group(1).strip(), None
    m = RE_DEF_COLON.match(t)
    if m and len(m.group(1)) <= 45 and " " in m.group(2)[:80]:
        term = m.group(1).strip()
        acro = None
        am = re.search(r"\(([A-Z][A-Za-z0-9\-]{1,10})\)", term)
        if am:
            acro = am.group(1)
        return term, acro
    return None, None


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


SEG_PREFIX = {"rule": "REG", "guidance": "GUIDE", "appendix": "APPX"}
SEG_LABEL = {"rule": "Regulations", "guidance": "Guidance", "appendix": "Appendix"}


def doc_title(items: list[dict]) -> str:
    for x in items:
        if x.get("type") == "text" and x.get("text_level") == 1:
            t = (x.get("text") or "").strip()
            if not RE_CHAPTER.match(t):
                return t
    return ""


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        self._seen_ids: set[str] = set()
        # (chapter, article_no) -> parent id, per segment, for cross-linking
        self.article_index: dict[str, dict[tuple, str]] = {"rule": {}, "guidance": {}}
        title = doc_title(items)
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        self.doc_meta = {
            "doc_id": stem.replace("-", "_").upper(),
            "doc_title": title or stem,
            "part_no": "", "part_title": title, "publisher": "ClassNK (NK)",
            "year": "2025", "language": "en",
        }

    def article_id(self, seg: str, chap: str, art: str | None) -> str:
        base = f"{self.doc_meta['doc_id']}_{SEG_PREFIX[seg]}_C{chap}"   # doc_id 접두로 문서 간 충돌 방지
        return f"{base}_{art.replace('.', '-')}" if art else f"{base}_INTRO"

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
            "seg": "rule", "chap": ("0", ""),
            "sec": ("", ""),                         # N.M group (no + title)
            "art": (None, ""), "marked": False,
            "pieces": [], "tables": [], "figures": [],
        }

        def flush():
            self._flush(st)
            st["pieces"], st["tables"], st["figures"] = [], [], []
            st["marked"] = False

        # 표지·Contents(목차)는 첫 밴드 헤더 앞에 온다 — 밴드가 있으면 거기서 시작해
        # 목차 조각이 기본 seg(rule)로 새는 것을 막는다(예: 지침 전용 720 typeapproval).
        start = 0
        for i, x in enumerate(self.items):
            if x.get("type") == "text" and x.get("text_level") == 1:
                t = (x.get("text") or "").strip()
                if RE_SEG_RULE.match(t) or RE_SEG_GUIDE.match(t):
                    start = i
                    break

        for x in self.items[start:]:
            typ = x.get("type")
            page = x.get("page_idx")

            if typ in ("header", "footer", "page_number", "page_footnote"):
                continue
            if typ == "table":
                st["tables"].append(x)
                continue
            if typ == "image":
                st["figures"].append(x)
                continue
            if typ == "equation":                 # 수식은 직전 문단에 병합(단독이면 새 조각)
                eq = (x.get("text") or "").strip()
                if eq:
                    if st["pieces"]:
                        st["pieces"][-1]["text"] += "\n" + eq
                        st["pieces"][-1]["pages"].add(page)
                    else:
                        st["pieces"].append({"text": eq, "pages": {page}, "enum": False})
                continue
            if typ == "list":
                for li in x.get("list_items", []):
                    li = (li or "").strip()
                    if li and not RE_LEADER.search(li):
                        st["pieces"].append({"text": li, "pages": {page}, "enum": True})
                continue
            if typ not in ("text", "ref_text"):   # ref_text=오분류된 열거항목 본문
                continue

            raw = (x.get("text") or "").strip()
            if not raw or RE_LEADER.search(raw):
                continue
            lvl = x.get("text_level")

            # segment band (L1): REGULATIONS/RULES, GUIDANCE, Appendix
            if lvl == 1:
                if RE_SEG_GUIDE.match(raw):
                    flush(); st["seg"] = "guidance"; st["chap"] = ("0", "")
                    st["sec"], st["art"] = ("", ""), (None, ""); continue
                if RE_SEG_RULE.match(raw):
                    flush(); st["seg"] = "rule"; st["chap"] = ("0", "")
                    st["sec"], st["art"] = ("", ""), (None, ""); continue
                if RE_SEG_APPX.match(raw):
                    flush(); st["seg"] = "appendix"; st["chap"] = ("0", "")
                    st["sec"], st["art"] = ("", ""), (None, ""); continue
                # otherwise an L1 chapter/title echo — fall through to chapter check

            # chapter
            m = RE_CHAPTER.match(raw)
            if m and lvl in (1, 2):
                flush()
                st["chap"] = (m.group(1), m.group(2).strip())
                st["sec"], st["art"] = ("", ""), (None, "")
                continue

            # article: N.M or N.M.K titled heading (L2) — 문자 접두형(K1.1.1)도 동일 취급
            m = RE_ARTICLE.match(raw)
            lm = None if m else RE_ARTICLE_LTR.match(raw)
            if (m or lm) and lvl == 2:
                flush()
                ano = m.group(1) if m else lm.group(2)
                title, marked = strip_star((m.group(2) if m else lm.group(3)).strip())
                st["art"] = (ano, title); st["marked"] = marked
                grp = ".".join(ano.split(".")[:2])
                if ano.count(".") == 1:        # this heading *is* the N.M group
                    st["sec"] = (grp, title)
                elif st["sec"][0] != grp:      # entered a new N.M group via a sub-clause
                    st["sec"] = (grp, "")
                continue

            # 문자 접두 장 헤딩("K1 GENERAL") — 지침부에서 Chapter 헤딩 역할
            lm = RE_CHAPTER_LTR.match(raw)
            if lm and lvl in (1, 2):
                flush()
                st["chap"] = (lm.group(2), lm.group(3).strip())
                st["sec"], st["art"] = ("", ""), (None, "")
                continue

            # body — enumerated item or paragraph
            enum = bool(RE_ENUM_ITEM.match(raw) or RE_ENUM_ALPHA.match(raw))
            st["pieces"].append({"text": raw, "pages": {page}, "enum": enum})

        flush()
        self._cross_link()
        return self.out

    def _flush(self, st: dict):
        seg = st["seg"]
        chap_no, chap_title = st["chap"]
        sec_no, sec_title = st["sec"]
        art_no, art_title = st["art"]
        pieces, tables, figures = st["pieces"], st["tables"], st["figures"]
        if not (pieces or tables or figures):
            return

        parent_id = self._uniq(self.article_id(seg, chap_no, art_no))
        if seg in self.article_index and art_no:
            self.article_index[seg].setdefault((chap_no, art_no), parent_id)

        path = [SEG_LABEL[seg], self.doc_meta["doc_title"]]
        if chap_no != "0" or chap_title:
            path.append(f"Chapter {chap_no} {chap_title}".strip())
        if sec_no and sec_no != art_no:
            path.append(f"{sec_no} {sec_title}".strip())
        if art_no:
            path.append(f"{art_no} {art_title}".strip())

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t in tables}
                       | {f.get("page_idx") for f in figures})
        full_text = "\n".join(pc["text"] for pc in pieces)
        is_def = "definition" in art_title.lower()

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta, "document_type": seg,
                "chapter_no": chap_no, "chapter_title": chap_title,
                "section_no": sec_no or art_no, "section_title": sec_title,
                "article_no": art_no, "article_title": art_title,
                "guidance_marked": st["marked"], "notations": [],
                "section_path": path, "pages": pages, "source_file": self.source_file,
            }
            base.update(extra)
            return base

        child_units = [u for u in self._split_children(pieces, is_def) if u["text"].strip()]
        # note/exception 분리(Rule 4) — 정의 조항은 통째로 유지

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
                term, _ = parse_definition(unit["text"])
                if term:
                    rec["term_ko"] = term
                    rec["term_en"] = None
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
                retrieval_text=" ".join(t for t in (cap, art_title, vis) if t),
                references=atom_refs(" ".join((cap, vis)), cap),
                linked_article_id=parent_id,
            ))

        parent = meta(
            chunk_id=parent_id, parent_chunk_id=None,
            chunk_level="parent", chunk_type="article",
            content=full_text, content_tokens=approx_tokens(full_text),
            summary="", topic="", has_cross_ref=st["marked"],
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

    def _cross_link(self):
        rule_ix, guide_ix = self.article_index["rule"], self.article_index["guidance"]
        by_id = {c["chunk_id"]: c for c in self.out if c["chunk_level"] == "parent"}
        for key, rid in rule_ix.items():
            gid = guide_ix.get(key)
            if gid and rid in by_id and gid in by_id:
                by_id[rid]["linked_guidance_chunk_id"] = gid
                by_id[gid]["linked_rule_chunk_id"] = rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content_list")
    ap.add_argument("-o", "--output", default="data_chunks/nk_rule_chunks.jsonl")
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
    linked = sum(1 for c in chunks if c["chunk_level"] == "parent"
                 and (c.get("linked_guidance_chunk_id") or c.get("linked_rule_chunk_id")))
    print(f"wrote {len(chunks)} chunks -> {op}")
    print(f"  levels : {dict(lv)}")
    print(f"  types  : {dict(ty)}")
    print(f"  cross-linked articles: {linked}")


if __name__ == "__main__":
    main()
