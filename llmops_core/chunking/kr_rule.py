#!/usr/bin/env python3
"""Domain-aware chunker for 선급 및 강선규칙 제1편 (KR classification rulebook).

Reads a MinerU ``content_list.json`` and emits structured RAG chunks that
preserve the 규칙 / 지침 / 부록 division and the 편 > 장 > 절 > 조 > 항 > 호 > 목
hierarchy, following the "계층 Parent-Child + 표/그림 구조체 + 규칙-지침 참조"
strategy.

Emitted JSONL records (one per line), distinguished by ``chunk_level`` /
``chunk_type``:

  parent   article-level chunk — full 조 text + children ids (문맥 확장용)
  child    the vector-search units — one 의미 단위 (항/호/정의) each
  table    table_body(HTML) + retrieval_text, linked_article_id
  figure   image path + caption + retrieval_text, linked_article_id

Rule and guidance articles sharing the same 장/절/조 path are cross-linked via
``linked_rule_chunk_id`` / ``linked_guidance_chunk_id`` (reinforced by the
【지침 참조】 / 【규칙 참조】 in-text markers).

Stdlib only. ``summary`` / ``topic`` (LLM) and figure ``ocr_text`` are left
empty for the existing ``enrich_chunks.py`` pass to fill.

    python scripts/kr_rule_chunker.py \
        data/1__2025/1__2025/1편_2025_content_list.json \
        -o data_chunks/kr_rule_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from html.parser import HTMLParser
from pathlib import Path

# ── table row-level splitting (전략 Rule 5: 표 행 단위 검색) ─────────────────
class _RowParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def table_rows(html: str) -> list[list[str]]:
    p = _RowParser()
    try:
        p.feed(html or "")
    except Exception:
        return []
    return [r for r in p.rows if any(c.strip() for c in r)]


def row_retrieval(header: list[str], row: list[str]) -> str:
    if header and len(header) == len(row):
        return " | ".join(f"{h}: {c}".strip(" :") for h, c in zip(header, row) if c.strip())
    return " ".join(c for c in row if c.strip())




# ── structural markers ──────────────────────────────────────────────────────
RE_CHAPTER = re.compile(r"^제\s*(\d+)\s*장\s*(.*)$")
RE_SECTION = re.compile(r"^제\s*(\d+(?:-\d+)?)\s*절\s*(.*)$")
RE_ARTICLE = re.compile(r"^(\d{3,4})\.\s*(\S.*)$")          # 101. 용어의 정의
RE_APPENDIX = re.compile(r"^부록\s*([\d\-]+)\s+(\S.*)$")     # 부록 1-17 …
RE_PARA = re.compile(r"^(\d{1,2})\.\s+\S")                   # 항  1. 2.
# 항 변형 "2-1. 산적화물선…"(정의 항목 등 N-M 넘버링) — 두 숫자 모두 1~2자리 제한으로
# 날짜(2025-12.)·조번호(101-1.) 배제, 뒤 첫 글자 한글/영문/괄호 가드로 수치 나열 배제.
RE_PARA_DASH = re.compile(r"^(\d{1,2}-\d{1,2})\.\s+[가-힣A-Za-z(]")
RE_ITEM = re.compile(r"^\((\d{1,2})\)\s*\S")                # 호  (1) (2)
# 호 변형 "1) 고객은…"(비고·특수 열거) — 줄 시작+공백 필수라 "비고 1)을" 같은
# 문장 내 참조는 자동 배제, 뒤 첫 글자 가드로 수치·수식 배제.
RE_ITEM_NP = re.compile(r"^(\d{1,2})\)\s+[가-힣A-Za-z(]")
RE_SUBITEM = re.compile(r"^\(([가나다라마바사아자차카타파하])\)\s*\S")  # 목 (가)(나)…(하) — (주)/(예)/(단) 등 제외
# 세목 (a)(b)… 단일문자 + 로마숫자 (i)(ii)…(x) (다중문자) — 공용 리졸버(paren_roman)와 일관.
# 로마형을 먼저 두어 (ii)(iii)(iv) 등이 (i)로 잘리지 않게 한다.
RE_SUBITEM2 = re.compile(r"^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-z])\)\s*\S")

# noise: dot-leaders / trailing page numbers (table-of-contents), 개정 block
RE_LEADER = re.compile(r"·{2,}|…|\.{4,}")
RE_AMEND = re.compile(r"^\s*[\-–—]\s|^적용일자|^개정사항|^차\s*례$|^【")

REF_MARKER_GUIDE = "【지침 참조】"
REF_MARKER_RULE = "【규칙 참조】"

# ── reference extraction ────────────────────────────────────────────────────
RE_REFS = [
    ("external_convention", re.compile(r"SOLAS[^,。\n]{0,40}")),
    ("external_convention", re.compile(r"MARPOL[^,。\n]{0,40}")),
    ("external_rule", re.compile(r"IACS\s*(?:UR|UI|PR|Rec\.?)?[\s\w\.\-]{0,20}")),
    ("internal_appendix", re.compile(r"부록\s*[\d\-]+(?:\-\d+)?")),
]

# ── chunk_type heuristics ───────────────────────────────────────────────────
OBLIGATION = ("하여야 한다", "해야 한다", "받아야 한다", "제출하여야", "따른다",
              "포함한다", "제공하여야", "확인하여야", "갖추어야", "비치하여야")
PROCEDURE = ("신청", "재검사", "재등록", "절차", "검사신청")
CONDITION = ("경우", "조건으로", " 때 ", "때에는")


def classify(text: str, article_title: str) -> str:
    head = text.lstrip()[:20]
    if "정의" in article_title or "이라 함은" in text or "이라 한다" in text:
        return "definition"
    if any(k in text for k in OBLIGATION):
        return "requirement"
    if any(k in text for k in PROCEDURE):
        return "procedure"
    if any(k in text for k in CONDITION):
        return "condition"
    return "paragraph"


def extract_refs(text: str) -> list[dict]:
    out, seen = [], set()
    for ref_type, rx in RE_REFS:
        for m in rx.finditer(text):
            tgt = re.sub(r"\s+", " ", m.group(0)).strip(" ,.")
            if tgt and tgt not in seen:
                seen.add(tgt)
                out.append({"ref_type": ref_type, "target": tgt})
    return out


def is_noise(text: str) -> bool:
    return bool(RE_LEADER.search(text)) or bool(RE_AMEND.match(text))


# ── term extraction for definitions (101. 용어의 정의) ───────────────────────
RE_DEF = re.compile(r"^\d+(?:-\d+)?\.\s*([^()\n]+?)\s*(?:\(([^)]*)\))?\s*(?:이|라)?라?\s*함은")  # "2-1. 산적화물선…" 변형 포함


def parse_definition(text: str) -> tuple[str | None, str | None]:
    m = RE_DEF.match(text.strip())
    if not m:
        return None, None
    term_ko = m.group(1).strip()
    term_en = (m.group(2) or "").strip() or None
    if term_en and not re.search(r"[A-Za-z]", term_en):
        term_en = None
    return term_ko or None, term_en


# ── document model ──────────────────────────────────────────────────────────
SEG_PREFIX = {"rule": "RULE", "guidance": "GUIDE", "appendix": "APPX"}
SEG_LABEL = {"rule": "규칙", "guidance": "지침", "appendix": "부록"}

# 편 anchors: "제2편 재료 및 용접" / "규칙 제 2 편"
RE_PART_TITLED = re.compile(r"^제\s*(\d+)\s*편\s+(\S.*)$")
RE_PART_ANCHOR = re.compile(r"^(?:규칙|지침)\s*제\s*(\d+)\s*편")


def detect_part(items: list[dict]) -> tuple[str, str]:
    """Return (part_no, part_title) from the 편 heading, defaulting to 1편."""
    part_no, part_title = "", ""
    for x in items[:400]:
        if x.get("type") != "text":
            continue
        t = (x.get("text") or "").strip()
        m = RE_PART_TITLED.match(t)
        if m:
            part_no = part_no or m.group(1)
            # keep the first, longest clean title (drop dot-leaders / page nums)
            cand = re.sub(r"[·.\s]*\d*\s*$", "", m.group(2)).strip(" “”\"")
            if cand and (not part_title or len(cand) < len(part_title)):
                part_title = cand
        elif not part_no:
            a = RE_PART_ANCHOR.match(t)
            if a:
                part_no = a.group(1)
    return part_no or "1", part_title or ""


def approx_tokens(text: str) -> int:
    # cheap proxy: Korean ~1 token/2 chars is close enough for size gates
    return max(1, len(text) // 2)


class Chunker:
    def __init__(self, items: list[dict], source_file: str):
        self.items = items
        self.source_file = source_file
        self.out: list[dict] = []
        # (chap, sec, art) -> parent chunk_id, per segment, for cross-linking
        self.article_index: dict[str, dict[tuple, str]] = {"rule": {}, "guidance": {}}
        self._seen_ids: set[str] = set()
        stem = re.sub(r"_content_list$", "", Path(source_file).stem)
        self.part_no, self.part_title = detect_part(items)
        if self.part_no == "1":                     # 편 번호 미검출 시 파일명에서 보강
            fm = re.match(r"^(\d+)편", stem)
            if fm:
                self.part_no = fm.group(1)
        self.decimal = self._is_decimal(items)      # CSR류(공통구조규칙) 십진구조?
        # doc_id는 파일(=문서)마다 고유해야 함 — part 번호만으론 Part-1 문서끼리 충돌.
        slug = re.sub(r"[^0-9A-Za-z가-힣]+", "_", stem).strip("_") or f"PART{self.part_no}"
        self.doc_meta = {
            "doc_id": f"KR_{slug}",
            "doc_title": "선급 및 강선규칙 / 선급 및 강선규칙 적용지침",
            "part_no": self.part_no,
            "part_title": self.part_title,
            "publisher": "한국선급",
            "year": "2025",
            "language": "ko",
        }

    # -- id helpers ----------------------------------------------------------
    def base_id(self, seg: str, chap: str, sec: str | None) -> str:
        did = self.doc_meta["doc_id"]               # 문서 간 chunk_id 충돌 방지 접두
        if seg == "appendix":
            base = f"{did}_APPX_P{self.part_no}_{chap}"   # chap holds the 부록 number, e.g. 1-17
        else:
            base = f"{did}_{SEG_PREFIX[seg]}_P{self.part_no}_C{chap}"
        if sec:
            base += f"_S{sec}"
        return base

    def article_id(self, seg, chap, sec, art) -> str:
        base = self.base_id(seg, chap, sec)
        return f"{base}_A{art}" if art else f"{base}_INTRO"

    def _uniq(self, cid: str) -> str:
        """같은 (편,장,절,조) 경로가 문서 내 반복될 때 고유 접미사로 분리."""
        if cid not in self._seen_ids:
            self._seen_ids.add(cid)
            return cid
        i = ord("b")
        while f"{cid}-{chr(i)}" in self._seen_ids:
            i += 1
        cid = f"{cid}-{chr(i)}"
        self._seen_ids.add(cid)
        return cid

    # -- main pass -----------------------------------------------------------
    def run(self) -> list[dict]:
        if self.decimal:                    # CSR류: 제N장 > 십진 clause
            return self._run_decimal()
        segments = self._segment()
        for seg, lo, hi in segments:
            self._parse_segment(seg, lo, hi)
        self._cross_link()
        return self.out

    @staticmethod
    def _is_decimal(items: list[dict]) -> bool:
        """CSR류 판별: 조(101.)가 거의 없고 십진 clause(1.1/1.1.1)가 많은 구조."""
        n_jo = n_dec = 0
        for x in items:
            if x.get("type") != "text":
                continue
            t = (x.get("text") or "").strip()
            if RE_ARTICLE.match(t):
                n_jo += 1
            elif re.match(r"^\d+\.\d+", t):
                n_dec += 1
        return n_jo < 3 and n_dec > max(10, n_jo * 3)

    def _csr_page_map(self) -> dict:
        """CSR은 장/절/부를 러닝헤더에 둔다("11 편 1 장 1 절" / "1 부 1 장 1 절").
        페이지 → (부, 장, 절) 맵을 만든다(ABS 방식). 편/부를 포함한 헤더만 신뢰."""
        raw: dict[int, tuple] = {}
        for x in self.items:
            typ = x.get("type")
            t = (x.get("text") or "").strip()
            # 러닝헤더: header/footer, 또는 text로 오분류된 짧은 "N편/부 M장 K절" 블록
            if typ in ("header", "footer"):
                if not re.search(r"\d+\s*(?:편|부)", t):
                    continue
            elif typ == "text" and len(t) < 16 and re.match(r"^\d+\s*(?:편|부)\b", t):
                pass
            else:
                continue
            mc = re.search(r"(\d+)\s*장", t)
            if not mc:
                continue
            mb = re.search(r"(\d+)\s*부", t)
            ms = re.search(r"(\d+)\s*절", t)
            raw[x.get("page_idx")] = (mb.group(1) if mb else "", mc.group(1),
                                      ms.group(1) if ms else "")
        if not raw:
            return {}
        pages = [p for p in (x.get("page_idx") for x in self.items) if p is not None]
        out, last = {}, raw[min(raw)]
        for p in range(min(pages), max(pages) + 1):
            if p in raw:
                last = raw[p]
            out[p] = last
        return out

    def _run_decimal(self) -> list[dict]:
        """CSR 십진구조: 장/절/부는 러닝헤더(page map), clause(1/1.1/1.1.1)는 본문 L2.
        장마다 clause가 재시작하므로 (부.)장.절 을 id에 넣어 충돌을 막는다."""
        RE_DEC = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*(\S.*)?$")   # 1. / 1.1 / 1.1.1 (+제목)
        pmap = self._csr_page_map()
        st = {"chap": ("0", ""), "sec": (None, ""), "art": (None, ""),
              "pieces": [], "tables": [], "figures": [], "ref_flag": False}
        seg = "rule"
        cur = None

        def flush():
            self._flush_article(seg, st)
            st["pieces"], st["tables"], st["figures"] = [], [], []
            st["ref_flag"] = False

        for x in self.items:
            typ = x.get("type")
            page = x.get("page_idx")
            if typ in ("header", "footer", "page_number", "page_footnote"):
                continue
            if typ == "table":
                st["tables"].append(x); continue
            if typ == "image":
                st["figures"].append(x); continue
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
                    if li and not is_noise(li):
                        st["pieces"].append({"text": li, "pages": {page}, "sub": False})
                continue
            if typ not in ("text", "ref_text"):   # ref_text=오분류된 조항/본문(예: 【참조】 마커 페이지)
                continue
            raw = (x.get("text") or "").strip()
            if not raw or is_noise(raw):        # ToC 점선·머리말 등 제거
                continue
            lvl = x.get("text_level")

            key = pmap.get(page)                # 러닝헤더 기반 장/절 전환
            if key and key != cur:
                flush(); cur = key
                b, c, s = key
                st["chap"] = (f"{b}.{c}" if b else c, "")
                st["sec"] = (s or None, "")
                st["art"] = (None, "")

            m = RE_DEC.match(raw)               # 십진 clause = article
            if m and lvl == 2:
                flush(); st["art"] = (m.group(1), (m.group(2) or "").strip()); continue
            st["pieces"].append({"text": raw, "pages": {page}, "sub": False})   # 본문

        flush()
        self._cross_link()
        return self.out

    def _segment(self) -> list[tuple[str, int, int]]:
        """Locate rule / guidance / appendix body ranges, skipping front matter."""
        rule_anchor = guide_anchor = appx_anchor = None
        for i, x in enumerate(self.items):
            if x.get("type") != "text":
                continue
            t = (x.get("text") or "").strip()
            if rule_anchor is None and re.match(r"^규칙\s*제\s*\d+\s*편", t):
                rule_anchor = i
            elif guide_anchor is None and re.match(r"^지침\s*제\s*\d+\s*편", t):
                guide_anchor = i
        # body starts at the first *clean* article heading after each anchor
        rule_body = self._first_article(rule_anchor or 0, guide_anchor or len(self.items))
        guide_body = self._first_article(guide_anchor or 0, len(self.items))
        # appendix body = first level-2 "부록 N-M" heading after the guidance body
        for i in range(guide_body or 0, len(self.items)):
            x = self.items[i]
            if (x.get("type") == "text" and x.get("text_level") == 2
                    and RE_APPENDIX.match((x.get("text") or "").strip())):
                appx_anchor = i
                break
        segs = []
        end_guide = appx_anchor if appx_anchor is not None else len(self.items)
        # rule body stops at the guidance *anchor* so the guidance front matter /
        # table-of-contents (which sits between anchor and body) belongs to neither.
        rule_end = guide_anchor if guide_anchor is not None else end_guide
        # 세그먼트는 anchor 직후부터 시작해 서문(머리말·적용범위)을 INTRO 조로 보존한다.
        # 목차는 파스 단계의 dot-leader/차례 노이즈 필터가 제거하므로 새지 않는다.
        # (anchor 미검출 문서는 기존처럼 첫 조에서 시작.)
        rule_start = (rule_anchor + 1) if rule_anchor is not None else rule_body
        guide_start = (guide_anchor + 1) if guide_anchor is not None else guide_body
        if rule_body is not None:
            segs.append(("rule", rule_start, rule_end))
        if guide_body is not None:
            segs.append(("guidance", guide_start, end_guide))
        if appx_anchor is not None:
            segs.append(("appendix", appx_anchor, len(self.items)))
        return segs

    def _first_article(self, lo: int, hi: int) -> int | None:
        for i in range(lo, hi):
            x = self.items[i]
            if x.get("type") != "text" or x.get("text_level") != 2:
                continue
            t = (x.get("text") or "").strip()
            if is_noise(t):
                continue
            if RE_ARTICLE.match(t):
                # rewind to the nearest clean 장 heading so context is captured
                for j in range(i, max(lo, i - 12), -1):
                    tj = (self.items[j].get("text") or "").strip()
                    if RE_CHAPTER.match(tj) and not is_noise(tj):
                        return j
                return i
        return None

    # -- per-segment parser --------------------------------------------------
    def _parse_segment(self, seg: str, lo: int, hi: int):
        st = {
            "chap": ("0", ""), "sec": (None, ""), "art": (None, ""),
            "pieces": [], "tables": [], "figures": [],
            "ref_flag": False,
        }

        def flush():
            self._flush_article(seg, st)
            st["pieces"], st["tables"], st["figures"] = [], [], []
            st["ref_flag"] = False

        for i in range(lo, hi):
            x = self.items[i]
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
                    if li and not is_noise(li):
                        st["pieces"].append({"text": li, "pages": {page}, "sub": False})
                continue

            if typ not in ("text", "ref_text"):   # ref_text=오분류된 조항/본문(예: 【참조】 마커 페이지)
                continue

            raw = (x.get("text") or "").strip()
            if not raw or is_noise(raw):
                continue
            lvl = x.get("text_level")

            # appendix heading
            m = RE_APPENDIX.match(raw)
            if seg == "appendix" and m and lvl == 2:
                flush()
                st["chap"] = (m.group(1), m.group(2).strip())
                st["sec"], st["art"] = (None, ""), (None, "")
                continue
            # chapter
            m = RE_CHAPTER.match(raw)
            if m and seg != "appendix":
                flush()
                st["chap"] = (m.group(1), m.group(2).strip())
                st["sec"], st["art"] = (None, ""), (None, "")
                continue
            # section
            m = RE_SECTION.match(raw)
            if m:
                flush()
                st["sec"] = (m.group(1), m.group(2).strip())
                st["art"] = (None, "")
                continue
            # article — L2 헤딩, 또는 【규칙/지침 참조】 마커가 있는 국내조(참조 스텁은
            # MinerU가 text_level을 안 매기는 경우가 잦아 레벨 무관 검출)
            m = RE_ARTICLE.match(raw)
            has_ref = bool(m) and (REF_MARKER_GUIDE in raw or REF_MARKER_RULE in raw)
            if m and (lvl == 2 or has_ref):
                flush()
                title = m.group(2)
                if REF_MARKER_GUIDE in title or REF_MARKER_RULE in title:
                    st["ref_flag"] = True
                    title = title.replace(REF_MARKER_GUIDE, "").replace(
                        REF_MARKER_RULE, "").strip()
                st["art"] = (m.group(1), title)
                continue

            # sub-heading inside an article (e.g. "1. 케이블 수밀 관통부 검사")
            sub = lvl == 2 and bool(RE_PARA.match(raw))
            st["pieces"].append({"text": raw, "pages": {page}, "sub": sub})

        flush()

    # -- flush one article into parent + child + table + figure chunks -------
    def _flush_article(self, seg: str, st: dict):
        chap_no, chap_title = st["chap"]
        sec_no, sec_title = st["sec"]
        art_no, art_title = st["art"]
        pieces, tables, figures = st["pieces"], st["tables"], st["figures"]
        if not (pieces or tables or figures):
            return

        parent_id = self._uniq(self.article_id(seg, chap_no, sec_no, art_no))
        if seg in self.article_index and art_no:
            self.article_index[seg].setdefault((chap_no, sec_no, art_no), parent_id)

        path = [SEG_LABEL[seg], f"제{self.part_no}편 {self.part_title}".strip()]
        if seg == "appendix":
            path.append(f"부록 {chap_no} {chap_title}".strip())
        elif chap_title:
            path.append(f"제{chap_no}장 {chap_title}")
        if sec_no:
            path.append(f"제{sec_no}절 {sec_title}".strip())
        if art_no:
            path.append(f"{art_no}. {art_title}")

        pages = sorted({p for pc in pieces for p in pc["pages"]}
                       | {t.get("page_idx") for t in tables}
                       | {f.get("page_idx") for f in figures})

        def meta(**extra) -> dict:
            base = {
                **self.doc_meta,
                "document_type": seg,
                "chapter_no": chap_no, "chapter_title": chap_title,
                "section_no": sec_no, "section_title": sec_title,
                "article_no": art_no, "article_title": art_title,
                "section_path": path,
                "pages": pages,
                "source_file": self.source_file,
            }
            base.update(extra)
            return base

        # -- parent ----------------------------------------------------------
        full_text = "\n".join(pc["text"] for pc in pieces)
        child_units = self._split_children(pieces, art_title)
        child_ids, table_ids, figure_ids = [], [], []

        # -- children --------------------------------------------------------
        is_def = "정의" in art_title

        child_chunks = []
        for k, unit in enumerate(child_units, 1):
            cid = f"{parent_id}_C{k:03d}"
            child_ids.append(cid)
            ctype = "text"
            rec = meta(
                chunk_id=cid, parent_chunk_id=parent_id,
                chunk_level="child", chunk_type=ctype,
                local_heading=unit.get("heading", ""),
                paragraph_no=unit.get("para_no", ""),      # 소속 항 번호
                paragraph_title=unit.get("para_title", ""),  # 소속 항 제목
                item_no=unit.get("item_no", ""),           # 자신의 호 번호((n))
                sub_item_no=unit.get("sub_item_no", ""),
                content=unit["text"],
                summary="", topic="",
                keywords=[], entities=[],
                references=extract_refs(unit["text"]),
                previous_chunk_id=None, next_chunk_id=None,
                linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
            )
            if is_def:
                term_ko, term_en = parse_definition(unit["text"])
                if term_ko:
                    rec["term_ko"] = term_ko
                    rec["term_en"] = term_en
                    rec["local_heading"] = term_ko
            child_chunks.append(rec)
        for a, b in zip(child_chunks, child_chunks[1:]):
            a["next_chunk_id"] = b["chunk_id"]
            b["previous_chunk_id"] = a["chunk_id"]

        # -- tables (표 전체 + 행 단위, 전략 Rule 5) --------------------------
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
                table_caption=cap,
                table_html=body,
                table_footnote=" ".join(tb.get("table_footnote") or []),
                img_path=tb.get("img_path", ""),
                content=cap,
                retrieval_text=(self._table_text(cap, body) + " " + " ".join(tb.get("table_footnote") or [])).strip(),
                table_nrows=len(rows),
                summary="", linked_article_id=parent_id, linked_table_rows=[],
            )
            table_chunks.append(tchunk)
            # 행 단위 검색 청크 — 헤더+데이터 2행 이상, 2열 이상일 때만
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
                        table_caption=cap, row_index=ri,
                        content=rtext,
                        retrieval_text=(cap + " " + rtext).strip(),
                        summary="", linked_article_id=parent_id, linked_table_id=tid,
                    ))

        # -- figures ---------------------------------------------------------
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
                caption=cap,
                image_path=fg.get("img_path", ""),
                ocr_text="", visual_summary="",
                content=cap,
                retrieval_text=" ".join(t for t in (cap, art_title, vis) if t),
                linked_article_id=parent_id,
            ))

        parent = meta(
            chunk_id=parent_id, parent_chunk_id=None,
            chunk_level="parent", chunk_type="article",
            content=full_text,
            content_tokens=approx_tokens(full_text),
            summary="", topic="",
            has_cross_ref=st["ref_flag"],
            references=extract_refs(full_text),
            children=child_ids,
            linked_tables=table_ids, linked_figures=figure_ids,
            linked_rule_chunk_id=None, linked_guidance_chunk_id=None,
        )
        self.out.append(parent)
        self.out.extend(child_chunks)
        self.out.extend(table_chunks)
        self.out.extend(figure_chunks)

    # -- split article body into child 의미 단위 -----------------------------
    @staticmethod
    def _split_children(pieces: list[dict], art_title: str) -> list[dict]:
        # 항(項)·호(號)(n)·목(가)·세목(a) 모두에서 개별 청크로 나눈다.
        # 목/세목은 소속 호 번호(item_no)를 상속하고, 자신의 번호는 sub_item_no에 기록한다
        # (목="가", 세목="가.a"). 대시·이어지는 본문은 상위 유닛에 붙는다.
        units: list[dict] = []
        cur: dict | None = None
        para_no, para_title = "", ""
        cur_item = cur_sub = ""                                # 현재 호·목 번호(하위 상속)

        for pc in pieces:
            # 블록 내부의 목/세목·호도 잡도록 줄 단위로 처리한다.
            # 호"(n)"·목"(가)"·세목"(a)"는 모호하지 않아 모든 줄에서 분리하되,
            # 항"n."은 **블록 첫 줄에서만** 인정한다(비고의 "1." "2." 오분류 방지).
            for li, t in enumerate(pc["text"].split("\n")):
                t = t.rstrip()
                if not t:
                    continue
                first = li == 0
                m_ho = RE_ITEM.match(t) or RE_ITEM_NP.match(t)   # 호 (n) / 변형 n)
                m_mok = RE_SUBITEM.match(t)
                m_semok = RE_SUBITEM2.match(t)
                # 항 n.은 블록 첫 줄만(비고 오분류 방지), 변형 n-m.은 가드가 강해 모든 줄 허용
                m_para = (RE_PARA.match(t) if first else None) or RE_PARA_DASH.match(t)
                if (first and pc.get("sub")) or m_para:       # 새 항
                    if cur:
                        units.append(cur)
                    para_no = m_para.group(1) if m_para else ""
                    para_title = re.sub(r"^\d{1,2}(?:-\d{1,2})?\.\s*", "", t).strip()[:60]
                    cur_item = cur_sub = ""
                    cur = {"text": t, "heading": para_title, "para_no": para_no,
                           "para_title": para_title, "item_no": "", "sub_item_no": ""}
                elif m_ho:                                    # 새 호 → 개별 유닛
                    if cur:
                        units.append(cur)
                    cur_item, cur_sub = m_ho.group(1), ""
                    cur = {"text": t, "heading": para_title, "para_no": para_no,
                           "para_title": para_title, "item_no": cur_item, "sub_item_no": ""}
                elif m_mok:                                   # 새 목 → 개별 유닛
                    if cur:
                        units.append(cur)
                    cur_sub = m_mok.group(1)
                    cur = {"text": t, "heading": para_title, "para_no": para_no,
                           "para_title": para_title, "item_no": cur_item, "sub_item_no": cur_sub}
                elif m_semok:                                 # 새 세목 → 개별 유닛
                    if cur:
                        units.append(cur)
                    sino = f"{cur_sub}.{m_semok.group(1)}" if cur_sub else m_semok.group(1)
                    cur = {"text": t, "heading": para_title, "para_no": para_no,
                           "para_title": para_title, "item_no": cur_item, "sub_item_no": sino}
                else:                                         # 대시/비고/이어지는 본문
                    if cur is None:
                        cur = {"text": t, "heading": "", "para_no": para_no,
                               "para_title": para_title, "item_no": "", "sub_item_no": ""}
                    else:
                        cur["text"] += "\n" + t
        if cur:
            units.append(cur)
        return units or [{"text": art_title, "heading": "", "para_no": "",
                          "para_title": "", "item_no": "", "sub_item_no": ""}]

    @staticmethod
    def _table_text(caption: str, html: str) -> str:
        cells = re.sub(r"<[^>]+>", " ", html)
        cells = cells.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")   # 부등호·기호 보존
        return re.sub(r"\s+", " ", (caption + " " + cells)).strip()

    # -- cross-link rule <-> guidance articles by 장/절/조 -------------------
    def _cross_link(self):
        rule_ix = self.article_index["rule"]
        guide_ix = self.article_index["guidance"]
        by_id = {c["chunk_id"]: c for c in self.out if c["chunk_level"] == "parent"}
        # exact (장, 절, 조) match only — article numbers repeat across chapters,
        # so matching on 조 alone would produce unsound many-to-one links.
        for key, rid in rule_ix.items():
            gid = guide_ix.get(key)
            if gid:
                by_id[rid]["linked_guidance_chunk_id"] = gid
                by_id[gid]["linked_rule_chunk_id"] = rid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("content_list", help="MinerU content_list.json")
    ap.add_argument("-o", "--output", default="data_chunks/kr_rule_chunks.jsonl")
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
