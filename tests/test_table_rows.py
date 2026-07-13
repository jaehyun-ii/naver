"""_tables 그리드 전개 검증 — rowspan/colspan 병합 표의 행 청크 무결성."""

from llmops_core.chunking._tables import row_retrieval, table_rows


def test_plain_table():
    html = ("<table><tr><td>이름</td><td>값</td></tr>"
            "<tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>")
    rows = table_rows(html)
    assert rows == [["이름", "값"], ["A", "1"], ["B", "2"]]
    assert row_retrieval(rows[0], rows[1]) == "이름: A | 값: 1"


def test_rowspan_expands_down():
    # 선급 표 전형: 첫 열이 구분값으로 세로 병합
    html = ("<table>"
            "<tr><td>구분</td><td>항목</td><td>기준</td></tr>"
            "<tr><td rowspan='2'>강재</td><td>인장</td><td>400</td></tr>"
            "<tr><td>항복</td><td>235</td></tr>"
            "</table>")
    rows = table_rows(html)
    assert rows[1] == ["강재", "인장", "400"]
    assert rows[2] == ["강재", "항복", "235"]          # 병합 값 복제
    assert row_retrieval(rows[0], rows[2]) == "구분: 강재 | 항목: 항복 | 기준: 235"


def test_colspan_expands_right():
    html = ("<table>"
            "<tr><td>구분</td><td colspan='2'>치수</td></tr>"
            "<tr><td>A</td><td>10</td><td>20</td></tr>"
            "</table>")
    rows = table_rows(html)
    assert rows[0] == ["구분", "치수", "치수"]
    assert rows[1] == ["A", "10", "20"]
    # 헤더 폭이 맞으니 header:value 포맷, 중복 쌍 없음
    assert row_retrieval(rows[0], rows[1]) == "구분: A | 치수: 10 | 치수: 20"


def test_colspan_duplicate_pair_collapsed():
    html = ("<table>"
            "<tr><td>구분</td><td>비고</td><td>비고</td></tr>"
            "<tr><td>A</td><td colspan='2'>공통</td></tr>"
            "</table>")
    rows = table_rows(html)
    assert rows[1] == ["A", "공통", "공통"]
    assert row_retrieval(rows[0], rows[1]) == "구분: A | 비고: 공통"


def test_rowspan_and_colspan_combined():
    html = ("<table>"
            "<tr><td rowspan='2' colspan='2'>X</td><td>c</td></tr>"
            "<tr><td>d</td></tr>"
            "</table>")
    rows = table_rows(html)
    assert rows == [["X", "X", "c"], ["X", "X", "d"]]


def test_unclosed_td_and_junk_span():
    html = ("<table><tr><td rowspan='abc'>a<td rowspan='0'>b</tr>"
            "<tr><td>c<td>d</tr></table>")
    rows = table_rows(html)
    assert rows == [["a", "b"], ["c", "d"]]


def test_malformed_html_returns_empty():
    assert table_rows(None) == []
    assert table_rows("") == []
    assert table_rows("<p>no table</p>") == []


def test_header_mismatch_falls_back_to_join():
    assert row_retrieval(["h1", "h2"], ["a", "b", "c"]) == "a b c"


def test_full_width_divider_row():
    # 전폭 colspan 구분자 행은 값 한 번만
    html = ("<table>"
            "<tr><td>Operation</td><td>Load type</td></tr>"
            "<tr><td colspan='2'>Seagoing operations</td></tr>"
            "<tr><td>Transit</td><td>S+D</td></tr>"
            "</table>")
    rows = table_rows(html)
    assert row_retrieval(rows[0], rows[1]) == "Seagoing operations"
    assert row_retrieval(rows[0], rows[2]) == "Operation: Transit | Load type: S+D"


def test_self_referential_pair_dropped():
    # 2단 헤더: rowspan 헤더 셀이 하위 행에 복제돼 "h: h" 자기참조가 되는 경우
    assert row_retrieval(["구분", "기준"], ["구분", "235"]) == "기준: 235"
