"""표 행 단위 분해 공용 유틸 (전략 Rule 5: 표 행 단위 검색).

rowspan/colspan을 그리드로 전개한다 — 병합 값을 점유 칸 전체에 복제해
행 청크가 (1) 헤더-값 정렬, (2) 병합 셀 값을 잃지 않게 한다.
전개 후 모든 행은 같은 폭이므로 row_retrieval 은 항상 "헤더: 값" 포맷을 탄다.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser


class _GridParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells: dict[tuple[int, int], str] = {}
        self._r = -1
        self._c = 0
        self._cell: list[str] | None = None
        self._cell_pos: tuple[int, int, int, int] | None = None  # r, c, rowspan, colspan

    @property
    def n_rows(self) -> int:
        return self._r + 1

    @staticmethod
    def _span(attrs, name: str) -> int:
        for k, v in attrs:
            if k == name:
                try:
                    n = int(str(v).strip())
                except (TypeError, ValueError):
                    return 1
                # rowspan="0"(HTML: 섹션 끝까지)·음수·과대값은 1/상한으로 방어
                return max(1, min(n, 512))
        return 1

    def _flush_cell(self):
        # </td> 누락(<td>a<td>b) 대비 — 새 셀/행 시작 시에도 호출
        if self._cell is None:
            return
        r, c, rs, cs = self._cell_pos
        text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
        for dr in range(rs):
            for dc in range(cs):
                self.cells.setdefault((r + dr, c + dc), text)
        self._c = c + cs
        self._cell = self._cell_pos = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._flush_cell()
            self._r += 1
            self._c = 0
        elif tag in ("td", "th") and self._r >= 0:
            self._flush_cell()
            while (self._r, self._c) in self.cells:  # 위 행 rowspan이 점유한 칸 건너뜀
                self._c += 1
            self._cell = []
            self._cell_pos = (self._r, self._c,
                              self._span(attrs, "rowspan"), self._span(attrs, "colspan"))

    def handle_endtag(self, tag):
        if tag in ("td", "th", "tr", "table"):
            self._flush_cell()

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def table_rows(html: str) -> list[list[str]]:
    """<table> HTML → 병합 전개된 균일 폭 2-D 행 리스트(공백 행 제거)."""
    p = _GridParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:
        return []
    if p.n_rows <= 0 or not p.cells:
        return []
    width = max(c for _, c in p.cells) + 1
    rows = [[p.cells.get((r, c), "") for c in range(width)]
            for r in range(p.n_rows)]
    return [r for r in rows if any(c.strip() for c in r)]


def row_retrieval(header: list[str], row: list[str]) -> str:
    vals = [c.strip() for c in row if c.strip()]
    if vals and len(set(vals)) == 1 and len(vals) == len(row):
        return vals[0]  # 전폭 colspan 구분자 행("Seagoing operations" 등) — 값 한 번만
    if header and len(header) == len(row):
        seen: list[str] = []
        for h, c in zip(header, row):
            if not c.strip() or h.strip() == c.strip():
                continue  # 빈 셀·자기참조 쌍(2단 헤더의 rowspan 헤더 셀) 제외
            part = f"{h}: {c}".strip(" :")
            if part and part not in seen:  # colspan 복제로 생긴 동일 쌍 축약
                seen.append(part)
        return " | ".join(seen)
    return " ".join(c for c in row if c.strip())
