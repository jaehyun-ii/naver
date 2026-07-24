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


def _blank(t: str, para_no: str = "", para_title: str = "",
           pages: set | None = None, src_idx: int = 0) -> dict:
    return {"text": t, "heading": "", "para_no": para_no,
            "para_title": para_title, "item_no": "", "sub_item_no": "",
            "pages": set(pages or ()), "src_idx": src_idx}


_RE_NOTE = re.compile(r"^Notes?\s*[:.]", re.I)      # "Note:" / "Notes." 각주 헤더


def interleave_children(text_children: list[tuple[int, dict]],
                        atom_groups: list[tuple[int, list[dict]]]) -> list[dict]:
    """텍스트 자식과 그림·표 원자를 원문 등장 순서로 병합.

    text_children: (유닛 시작 piece 인덱스, 청크). atom_groups: (앵커, [원자, 행…]).
    앵커 p = 원자가 등장한 시점까지 쌓인 piece 수 — piece p-1과 p 사이 위치를 뜻하므로
    같은 인덱스에서는 원자(0)가 그 인덱스에서 시작하는 텍스트(1)보다 앞선다.
    """
    merged = (
        [(pos, 1, seq, [c]) for seq, (pos, c) in enumerate(text_children)]
        + [(pos, 0, seq, grp) for seq, (pos, grp) in enumerate(atom_groups)]
    )
    merged.sort(key=lambda m: (m[0], m[1], m[2]))
    return [c for _, _, _, grp in merged for c in grp]


# 그림/표 번호 토큰("그림 1.2.3" / "Fig. 3" / "Table 5") — 조각 병합의 경계 신호
_RE_FIG_NO = re.compile(r"(그림|표|figure|fig\.?|table)\s*\.?\s*[A-Za-z]?\d", re.I)


def merge_figure_fragments(figures: list[tuple[dict, int]]) -> list[tuple[dict, int]]:
    """레이아웃 모델이 조각낸 복합 그림(패널+라벨)을 원자 1개로 병합.

    같은 (페이지, 앵커)에서 연속 감지된 figure 묶음 중 **그림 번호 캡션이 정확히
    1개**일 때만 병합한다(0개=판단 불가, 2개 이상=서로 다른 그림 → 그대로 둠).
    앵커가 같다는 것은 사이에 본문 조각이 없었다는 뜻이라 오병합을 막는다.
    병합 결과: 캡션은 원문 순서 결합("5% 20% … 그림 1.2.3 피팅강도"),
    img_paths 는 조각 경로 리스트, 분석 텍스트는 개행 결합.
    """
    groups: list[list[tuple[dict, int]]] = []
    for fg, anchor in figures:
        key = (fg.get("page_idx"), anchor)
        if groups and (groups[-1][0][0].get("page_idx"), groups[-1][0][1]) == key:
            groups[-1].append((fg, anchor))
        else:
            groups.append([(fg, anchor)])
    out: list[tuple[dict, int]] = []
    for grp in groups:
        caps = [" ".join(fg.get("image_caption") or []) for fg, _ in grp]
        numbered = [i for i, c in enumerate(caps) if _RE_FIG_NO.search(c)]
        # 번호 토큰 총수로 판정 — 한 캡션에 번호 2개("Figure 5.4 Figure 5.5")면
        # 파서가 나란한 두 그림의 캡션을 합쳐둔 것이므로 병합하지 않는다.
        n_tokens = sum(len(list(_RE_FIG_NO.finditer(c))) for c in caps)
        if len(grp) < 2 or len(numbered) != 1 or n_tokens != 1:
            out.extend(grp)
            continue
        # 결합 캡션 — 캡션 **요소** 단위로 완전 동일한 것은 1회만(순서 유지,
        # 무손실 dedupe). 조각마다 같은 각주("All dimensions in mm …")를 반복
        # 부착하는 파서 패턴을 정리한다. 길이 기반 정리는 하지 않는다:
        # 긴 조각 캡션 대부분이 패널 설명("(b) slit type…") 실데이터라 자르면 손실.
        uniq_caps = list(dict.fromkeys(
            el.strip() for fg, _ in grp for el in (fg.get("image_caption") or []) if el.strip()))
        merged = {
            "type": "image",
            "page_idx": grp[0][0].get("page_idx"),
            "image_caption": [" ".join(uniq_caps)],
            "image_footnote": [fn for fg, _ in grp for fn in (fg.get("image_footnote") or [])],
            "img_path": grp[numbered[0]][0].get("img_path", ""),   # 대표=번호 캡션 조각
            "img_paths": [fg.get("img_path", "") for fg, _ in grp if fg.get("img_path")],
            "content": "\n".join((fg.get("content") or "").strip()
                                 for fg, _ in grp if (fg.get("content") or "").strip()),
            "sub_type": next((fg.get("sub_type") for fg, _ in grp if fg.get("sub_type")), None),
        }
        out.append((merged, grp[0][1]))
    return out


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
    pending_pages: set = set()         # 보류된 라벨의 페이지(병합 시 함께 이월)

    def note_text(t: str, pgs: set) -> str:   # 보류된 "Note:" 라벨을 새 유닛 맨앞에 병합
        nonlocal pending_note, pending_pages
        if pending_note:
            t = pending_note + "\n" + t
            pgs |= pending_pages
            pending_note, pending_pages = None, set()
        return t

    for i, pc in enumerate(pieces):
        t = pc["text"]
        pgs = set(pc.get("pages") or ())
        if _RE_NOTE.match(t):                      # Note: → 각주 헤더(단독 유닛 만들지 않음)
            if cur:
                units.append(cur)
                cur = None
            para_no, para_title = "Note", "Note"
            item_style = sub_style = None
            cur_item = cur_sub = ""
            note_mode = True
            pending_note = t                       # 뒤따르는 각주/본문 유닛에 접어 넣음
            pending_pages = pgs
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
            cur = {"text": note_text(t, pgs), "heading": para_no, "para_no": para_no,
                   "para_title": para_title, "item_no": "", "sub_item_no": "",
                   "pages": pgs, "src_idx": i}
            continue

        if note_mode and RE_BARE_NUM.match(t):     # 각주 번호("1 …") = 호
            if cur:
                units.append(cur)
            tok = _first_token(t)
            cur_item, cur_sub = tok, ""
            cur = {"text": note_text(t, pgs), "heading": tok, "para_no": "Note",
                   "para_title": "Note", "item_no": tok, "sub_item_no": "",
                   "pages": pgs, "src_idx": i}
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
                cur = {"text": note_text(t, pgs), "heading": tok, "para_no": para_no,
                       "para_title": para_title, "item_no": tok, "sub_item_no": "",
                       "pages": pgs, "src_idx": i}
                continue
            if sub_style is None or st == sub_style:   # 목 → 개별 유닛
                sub_style = st
                if cur:
                    units.append(cur)
                cur_sub = tok
                cur = {"text": note_text(t, pgs), "heading": tok, "para_no": para_no,
                       "para_title": para_title, "item_no": cur_item, "sub_item_no": tok,
                       "pages": pgs, "src_idx": i}
                continue
            # 그보다 더 깊은 스타일 = 세목 → "목.세목" 경로
            if cur:
                units.append(cur)
            cur = {"text": note_text(t, pgs), "heading": tok, "para_no": para_no, "para_title": para_title,
                   "item_no": cur_item, "sub_item_no": f"{cur_sub}.{tok}" if cur_sub else tok,
                   "pages": pgs, "src_idx": i}
            continue

        if cur is None:                            # dash·이어지는 본문
            cur = _blank(note_text(t, pgs), para_no, para_title, pgs, i)
        else:
            cur["text"] += "\n" + t
            cur["pages"] |= pgs
    if cur:
        units.append(cur)
    elif pending_note:                             # "Note:" 뒤 내용 없이 끝난 경우만 단독 보존
        units.append(_blank(pending_note, "Note", "Note", pending_pages, len(pieces)))
    return units
