"""상대 뎁스 열거 리졸버 (전 선급 공용).

선급/문서마다 열거 마커의 뎁스가 다르고 혼재한다(STRUCTURE_ANALYSIS.md 참조).
마커→고정뎁스가 아니라 **등장 순서**로 뎁스를 정한다:

  - 한 항(para) 범위에서 **처음 나온 열거 스타일 = 호(item)** → 개별 유닛으로 분리
  - 그 아래 나오는 **다른 스타일 = 목/세목** → 상위 호에 포함(분리 안 함)
  - 같은 스타일이 다시 나오면 형제 호
  - dash·불릿·이어지는 본문 = 언제나 상위에 포함

이렇게 하면 `bare-1 (1)(2) bare-2`(항>호)나 `1) a)b) 2) a)b)`(호>목) 모두 올바른
뎁스로 청킹된다.
"""
from __future__ import annotations
import re

# 열거 스타일 분류 — 로마자를 알파벳보다 먼저 검사
_STYLES = [
    ("paren_num",   re.compile(r"^\(\d{1,2}\)\s")),                        # (1)
    ("paren_roman", re.compile(r"^\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s", re.I)),  # (i)
    ("paren_alpha", re.compile(r"^\([a-z]\)\s")),                         # (a)
    ("paren_kr",    re.compile(r"^\([가-힣]\)\s")),                        # (가)
    ("num_paren",   re.compile(r"^\d{1,2}\)\s")),                         # 1)
    ("roman_paren", re.compile(r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s", re.I)),   # i)
    ("alpha_paren", re.compile(r"^[a-z]\)\s")),                           # a)
    # 마침표형(LR 등): "a. 산문…" / "i. 산문…" — 로마를 알파벳보다 먼저 검사.
    # 가드: 마커 뒤에 letter가 와야 인정 → 변수정의("m. : 계수", "l. = …")·페이지("p. 45")
    #       같은 오분리를 배제(단일 문자 마침표는 수식 변수 기호로도 흔히 쓰임).
    ("roman_dot",   re.compile(r"^(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s+[A-Za-z]")),  # i. Word
    ("alpha_dot",   re.compile(r"^[a-z]\.\s+[A-Za-z]")),                 # a. Word
]
_RE_DASH = re.compile(r"^[-–•·]\s")

# 항(para) 판별: 십진 하위조항(N.M~) 또는 맨앞 번호("1 …"). 수식은 번호 뒤 공백 요구로 배제.
RE_DECIMAL = re.compile(r"^(?:\d+\.\d+(?:\.\d+)*\s+\S|\d+\.\s+[A-Z])")
RE_BARE_NUM = re.compile(r"^\d{1,2}\s+[A-Za-z가-힣]")


def para_of_default(t: str):
    """기본 항 판별 — 십진 하위조항 or 맨앞 번호면 para_no, 아니면 None."""
    if RE_DECIMAL.match(t) or RE_BARE_NUM.match(t):
        return re.split(r"\s", t.split("\n", 1)[0], 1)[0].rstrip(".")
    return None


def enum_style(t: str) -> str | None:
    """열거 마커 스타일 이름(호/목 판별용). dash·본문은 None."""
    for name, rx in _STYLES:
        if rx.match(t):
            return name
    return None


def _first_token(t: str) -> str:
    return re.split(r"\s", t.split("\n", 1)[0], 1)[0]


def _blank(t: str, para_no: str = "", para_title: str = "") -> dict:
    return {"text": t, "heading": "", "para_no": para_no,
            "para_title": para_title, "item_no": "", "sub_item_no": ""}


_RE_NOTE = re.compile(r"^Notes?\s*[:.]", re.I)      # "Note:" / "Notes." 각주 헤더


def resolve_units(pieces: list[dict], para_of) -> list[dict]:
    """조 안의 pieces를 항>호>목 유닛으로 분해.

    para_of(text) -> para_no(str) 이면 그 조각이 새 **항**(십진 하위조항/맨앞번호 등)을
    시작함을 뜻하고, None이면 항이 아님. 항 판별은 선급별로 주입한다.

    "Note:" 이후의 맨앞-번호("1 …")는 문단(항)이 아니라 **각주(호)** 로 취급한다
    (footnote 번호가 문단 번호로 오분류되는 것을 방지).
    """
    units: list[dict] = []
    cur: dict | None = None
    para_no = para_title = ""
    item_style: str | None = None      # 현재 항에서 '호'로 확정된 스타일
    sub_style: str | None = None       # 그 아래 '목'으로 확정된 스타일
    cur_item = cur_sub = ""            # 현재 호·목 번호(하위가 상속)
    note_mode = False                  # Note: 범위 안이면 맨앞-번호=각주(호)
    pending_note: str | None = None    # 단독 "Note:" 라벨 — 다음 유닛에 접어 넣기 위해 보류

    def note_text(t: str) -> str:      # 보류된 "Note:" 라벨을 새 유닛 맨앞에 병합
        nonlocal pending_note
        if pending_note:
            t = pending_note + "\n" + t
            pending_note = None
        return t

    for pc in pieces:
        t = pc["text"]
        if _RE_NOTE.match(t):                      # Note: → 각주 헤더(단독 유닛 만들지 않음)
            if cur:
                units.append(cur)
                cur = None
            para_no, para_title = "Note", "Note"
            item_style = sub_style = None
            cur_item = cur_sub = ""
            note_mode = True
            pending_note = t                       # 뒤따르는 각주/본문 유닛에 접어 넣음
            continue

        pno = None if (note_mode and RE_BARE_NUM.match(t)) else para_of(t)
        if pno is not None:                        # 새 항
            if cur:
                units.append(cur)
            para_no = pno
            para_title = re.sub(r"^\S+\.?\s*", "", t.split("\n", 1)[0]).strip()[:70]
            item_style = sub_style = None
            cur_item = cur_sub = ""
            note_mode = False
            cur = {"text": note_text(t), "heading": para_no, "para_no": para_no,
                   "para_title": para_title, "item_no": "", "sub_item_no": ""}
            continue

        if note_mode and RE_BARE_NUM.match(t):     # 각주 번호("1 …") = 호
            if cur:
                units.append(cur)
            tok = _first_token(t)
            cur_item, cur_sub = tok, ""
            cur = {"text": note_text(t), "heading": tok, "para_no": "Note",
                   "para_title": "Note", "item_no": tok, "sub_item_no": ""}
            continue

        st = enum_style(t)
        if st is not None:
            tok = _first_token(t)
            if item_style is None:
                item_style = st                    # 이 항의 첫 열거 스타일 = 호
            if st == item_style:                   # 형제 호 → 새 유닛
                if cur:
                    units.append(cur)
                cur_item, cur_sub, sub_style = tok, "", None
                cur = {"text": note_text(t), "heading": tok, "para_no": para_no,
                       "para_title": para_title, "item_no": tok, "sub_item_no": ""}
                continue
            if sub_style is None or st == sub_style:   # 목 → 개별 유닛
                sub_style = st
                if cur:
                    units.append(cur)
                cur_sub = tok
                cur = {"text": note_text(t), "heading": tok, "para_no": para_no,
                       "para_title": para_title, "item_no": cur_item, "sub_item_no": tok}
                continue
            # 그보다 더 깊은 스타일 = 세목 → "목.세목" 경로
            if cur:
                units.append(cur)
            cur = {"text": note_text(t), "heading": tok, "para_no": para_no, "para_title": para_title,
                   "item_no": cur_item, "sub_item_no": f"{cur_sub}.{tok}" if cur_sub else tok}
            continue

        if cur is None:                            # dash·이어지는 본문
            cur = _blank(note_text(t), para_no, para_title)
        else:
            cur["text"] += "\n" + t
    if cur:
        units.append(cur)
    elif pending_note:                             # "Note:" 뒤 내용 없이 끝난 경우만 단독 보존
        units.append(_blank(pending_note, "Note", "Note"))
    return units
