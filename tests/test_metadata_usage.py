"""메타데이터 적재적소 수정(2026-07-21) 검증 — LLM·Qdrant 불필요(순수 단위).

감사에서 잡은 5개 갭의 회귀 테스트:
①표 병합 expansion(예산 인지형 동반) ②학습 풀-골든 표 대칭(_attach_tables)
③규칙↔지침 미러 결정적 제외(RagContext) ④router 참조 해소(장접두·지침·named·
references 보강·접두 번호 탈취 방지) ⑤서빙 프롬프트 학습 형식 정렬.
"""
from __future__ import annotations

import types

import pytest

from llmops_core.rag.expansion import (_split_tables, _with_tables,
                                       expand_context)
from llmops_core.rag.pipeline import context_block
from llmops_core.rag.store import Document, Hit
from scripts.aireg_kr import build_training as bt
from scripts.aireg_kr import router as rt

TAB = "\n\n[인용된 표]\n"


# ── ① expansion: 예산 인지형 표 동반 ────────────────────────────────────
def test_split_tables_roundtrip():
    body, tables = _split_tables("본문A" + TAB + "표행1\n표행2")
    assert body == "본문A" and tables.startswith("표행1")
    assert _split_tables("표 없음") == ("표 없음", "")


def test_with_tables_budget():
    # 여유 예산 → 전체 동반
    out = _with_tables("본문", "표" * 300, 4000)
    assert "[인용된 표]" in out and "생략" not in out
    # 부족 예산 → 절단 동반 + 표식
    out = _with_tables("본" * 3000, "표" * 3000, 4000)
    assert "생략" in out and len(out) <= 4100
    # 200자 미만 여유 → 표 생략(무의미한 파편 방지)
    assert "[인용된 표]" not in _with_tables("본" * 3900, "표" * 500, 4000)


def test_expand_article_keeps_truncated_tables_over_budget():
    """표 병합으로 조가 예산을 넘겨도 본문 + 절단 표를 유지해야 한다(종전: 표 전량 유실)."""
    parent = {"content": "본문. " * 500 + TAB.lstrip("\n") + "표값 " * 800}
    parent["content"] = "본문. " * 500 + TAB + "표값 " * 800
    text, level = expand_context({"content": "leaf", "metadata": {}}, parent, [],
                                 budget_chars=4000)
    assert level == "article"
    assert "[인용된 표]" in text and "생략" in text
    assert len(text) <= 4100


def test_expand_descent_carries_tables():
    """항 하강 시에도 남는 예산에 표를 동반한다."""
    kids = [{"content": f"경로줄\n{i}항 본문", "metadata": {
        "chunk_type": "text", "paragraph_no": "1", "chunk_order": i}}
        for i in range(3)]
    leaf = {"content": "leaf", "metadata": {"paragraph_no": "1", "item_no": ""}}
    parent = {"content": "본문. " * 2000 + TAB + "표값A 표값B"}
    text, level = expand_context(leaf, parent, kids, budget_chars=4000)
    assert level == "paragraph"
    assert "표값A" in text


# ── ② 학습 풀 표 대칭: _attach_tables 경로 ──────────────────────────────
def test_attach_tables_merges_linked():
    from scripts.aireg_kr.common import _attach_tables
    p = {"content": "표 1.1에 따른다.", "linked_tables": ["T1"]}
    _attach_tables([p], {"T1": {"content": "표 1.1 제목",
                                "table_html": "<table><tr><td>10</td></tr></table>"}})
    assert "[인용된 표]" in p["content"] and "<table" in p["content"]


# ── ③ RagContext: 미러 결정적 제외 ─────────────────────────────────────
def _fake_retrieval(hits, equiv):
    m = types.SimpleNamespace()
    m.search_parents = lambda q, k=20, **kw: hits[:k]
    m.equivalent_parents = lambda text, gid, k=40: set(equiv) | {gid}
    m.QDRANT_URL, m.COLLECTION = "fake", "fake"
    return m


def test_ragcontext_excludes_linked_mirror(monkeypatch):
    pool = [
        {"publisher": "KR", "chunk_id": "R_GOLD", "content": "골든",
         "linked_guidance_chunk_id": "G_MIRROR"},
        {"publisher": "KR", "chunk_id": "G_MIRROR", "content": "지침 미러"},
        {"publisher": "KR", "chunk_id": "R_OTHER", "content": "무관 조"},
    ]
    ctx = bt.RagContext.__new__(bt.RagContext)  # __init__의 실제 retrieval 임포트 우회
    ctx.idx = bt.build_doc_index(pool)
    ctx._eq = {}
    # 검색이 미러를 1순위로 돌려줘도(임베딩 임계 아래 가정 — equiv에 없음)
    ctx.rt = _fake_retrieval(
        hits=[{"parent_chunk_id": "G_MIRROR", "score": 0.79, "publisher": "KR"},
              {"parent_chunk_id": "R_OTHER", "score": 0.5, "publisher": "KR"},
              {"parent_chunk_id": "R_GOLD", "score": 0.4, "publisher": "KR"}],
        equiv=[])
    rule = {"chunk_id": "R_GOLD", "publisher": "KR", "article_text": "골든"}
    docs, golden_hit = ctx.retrieve("질의", rule, n=2)
    ids = [d["chunk_id"] for d in docs]
    assert "G_MIRROR" not in ids  # 링크 메타로 결정적 제외
    assert "R_GOLD" not in ids


# ── ④ router 참조 해소 ─────────────────────────────────────────────────
def _mk(cid, ch, no, content, doc_type="rule", src="1편_2025_chunks"):
    return {"chunk_id": cid, "chapter_no": ch, "article_no": no,
            "content": content, "document_type": doc_type,
            "section_path": ["규칙"], "_source_file": src}


def test_ch_ref_resolves_and_bare_prefix_guard():
    ref = _mk("R_C4_301", "4", "301", "지지점 사이의 거리는 5 m 이상이어야 한다.")
    by_ch = {("4", "301"): [ref]}
    r = _mk("R_SELF", "2", "101", "간격은 4장 301.의 규정에 따른다.")
    aux = rt.find_crossref(r, {}, by_ch)
    assert aux and aux["ref_chunk_id"] == "R_C4_301"
    # 접두 번호 탈취 방지: "지침 4편 9장 201."의 201을 bare 매처가 집으면 안 된다
    r2 = _mk("R_SELF2", "2", "101", "이 요건은 지침 4편 9장 201.의 규정에도 적합하여야 한다.")
    by_no = {"201": [_mk("R_C2_201", "2", "201", "연차검사 시기.")]}
    assert rt.find_crossref(r2, by_no, {}) is None


def test_guide_prefix_resolves_to_guide_segment():
    guide = _mk("G_C1_801", "1", "801", "1. 추가검사의 요구 …", doc_type="guidance")
    r = _mk("R_SELF", "2", "401", '"검사원이 필요하다고 인정하는 경우"라 함은 지침 1장 801.의 1항에 해당되는 경우.')
    aux = rt.find_crossref(r, {}, {}, {("1", "801"): [guide]})
    assert aux and aux["ref_chunk_id"] == "G_C1_801"
    assert aux["ref_doc_type"] == "guidance"
    # 합성어 문서명(OO지침·OO규칙)은 같은 문서 조인 금지 유지
    r2 = _mk("R_SELF2", "2", "401", "저인화점연료선박규칙 1장 801.에 따른다.")
    assert rt.find_crossref(r2, {}, {("1", "801"): [_mk("R_C1_801", "1", "801", "x")]},
                            {}) is None


def test_meta_refs_augment_regex_miss():
    """개행 분절 등으로 본문 정규식이 못 잡아도 청커 references로 해소된다."""
    ref = _mk("R_C3_111", "3", "111", "만재흘수는 다음과 같다.")
    r = _mk("R_SELF", "2", "101", "타판의 재료계수는 3편 1 장 1 1 1 . 의 규정에 따른다.")
    r["references"] = [{"ref_type": "internal_article", "target": "3장 111"}]
    aux = rt.find_crossref(r, {}, {("3", "111"): [ref]})
    assert aux and aux["ref_chunk_id"] == "R_C3_111"


def test_xdoc_numbered_and_guide():
    rule = _mk("R_P5_C6_201", "6", "201", "공기관의 요건.", src="5편_2025_chunks")
    guide = _mk("G_P4_C2_102", "2", "102", "노출갑판의 위치.", doc_type="guidance",
                src="4편_2025_chunks")
    xdoc = {("5", "6"): [rule]}
    gx = {("4", "2"): [guide]}
    r = _mk("R_SELF", "7", "104", "코퍼댐에는 5편 6장 201.에 따른 공기관을 설치한다.",
            src="10편_2025_chunks")
    aux = rt.find_xdoc_crossref(r, xdoc, gx)
    assert aux and aux["ref_chunk_id"] == "R_P5_C6_201" and aux["xdoc"]
    r2 = _mk("R_SELF2", "1", "104", "상세는 지침 4편 2장 102.에 따른다.",
             src="14편_2025_chunks")
    aux2 = rt.find_xdoc_crossref(r2, xdoc, gx)
    assert aux2 and aux2["ref_chunk_id"] == "G_P4_C2_102"
    assert aux2["ref_doc_type"] == "guidance"


def test_nameddoc_unique_and_ambiguous():
    idx = {"저인화점연료선박규칙2025chunks": {("6", "301"): [
        _mk("N_C6_301", "6", "301", "위험구역의 통풍.")]}}
    r = _mk("R_SELF", "1", "101", "통풍은 저인화점연료선박규칙 6장 301.에 따른다.")
    aux = rt.find_nameddoc_crossref(r, idx)
    assert aux and aux["ref_chunk_id"] == "N_C6_301"
    # 다중 문서 매칭 → 모호 스킵
    idx2 = dict(idx); idx2["다른저인화점연료선박규칙x"] = idx["저인화점연료선박규칙2025chunks"]
    assert rt.find_nameddoc_crossref(r, idx2) is None


def test_permissive_clause_excluded_from_unit_convert():
    art = ("길이방향의 이음은 맞대기이음 양면용접으로 하여야 한다. 다만, 판두께가 "
           "9 mm 이하일 때에는 필릿용접 겹이음으로 하여도 좋다.")
    assert rt.find_unit_threshold({"content": art}) is None  # 허용 예외 수치 배제
    art2 = "동체의 판두께는 12 mm 이상이어야 한다."
    got = rt.find_unit_threshold({"content": art2})
    assert got and got["threshold"] == 12.0 and got["quantity"] == "판두께"
    # 관형형 "…할 수 있는 장치"는 허용 술어가 아니다(과차단 방지)
    art3 = "배기관은 갑판 위 2.4 m 이상의 높이까지 유도하고 화염의 방출을 방지할 수 있는 장치를 갖추어야 한다."
    assert rt.find_unit_threshold({"content": art3}) is not None


def test_foreign_refs_gate():
    from scripts.aireg_kr.build_suite_qa import foreign_refs
    row = {"question": "질문", "gold_answer": "이는 304.4항의 두께 계산에 사용된다. 표 3.14.4 참조.",
           "evidence": [{"article_text": "계수는 표 3.14.3에 정하는 값. 4. 두께는 다음과 같다."}]}
    refs = foreign_refs(row)
    assert "304.4항" in "".join(refs) and any("3.14.4" in x for x in refs)
    row2 = {"question": "질문", "gold_answer": "표 3.14.3에 따라 값은 0.4.",
            "evidence": [{"article_text": "계수는 표 3.14.3에 정하는 값."}]}
    assert foreign_refs(row2) == []


# ── ⑤ 서빙 프롬프트 학습 형식 정렬 ─────────────────────────────────────
def test_build_context_matches_training_format():
    from llmops_core.rag.pipeline import RagPipeline
    hits = [Hit(Document(id="1", text="본문A", metadata={"section_path": ["규칙", "제1편"]}), 0.9),
            Hit(Document(id="2", text="본문B", metadata={}), 0.8)]
    ctx = RagPipeline.build_context(RagPipeline.__new__(RagPipeline), hits)
    assert ctx.startswith("[문서 1] 규칙 > 제1편\n본문A")
    assert "\n\n[문서 2]\n본문B" in ctx


def test_context_block_header_aligned():
    assert "[검색된 규정 조항]" in context_block("X")


# ── ⑥ 표 셀 대조 정규화 (전문가 검수 7차 오탐 대응) ─────────────────────
def test_cell_in_entities_latex_composite():
    from scripts.aireg_kr.build_suite_qa import _cell_in, norm_table
    body = norm_table("10년 &lt; 선령 ≤ 15년 | $37 \\text{ m}^{2}$ | RLCA -40°C 평균 흡수에너지 27J 이상")
    assert _cell_in("10년 < 선령 ≤ 15년", body)      # HTML 엔티티
    assert _cell_in("37 m²", body)                    # LaTeX 래퍼·위첨자
    assert _cell_in("RLCA -40°C 평균 흡수에너지 27J 이상", body)  # 복합 셀
    assert not _cell_in("존재하지 않는 계수 9.99 MPa 특수값", body)  # 날조 거부


# ── ⑦ 결론 게이트 부정 범위·가상 표준번호 (전문가 검수 9차 대응) ────────
def test_conclusion_gate_negated_negation_not_flagged():
    from scripts.aireg_kr.build_suite_qa import judgment_conclusion_issues
    # 재생성 #12 실측: "미적용 사유가 아니라"는 적용 결론을 지지 — 오탐 금지
    row = {"expected_judgment": "applicable",
           "gold_answer": "형식시험을 실시하지 않은 것은 미적용 사유가 아니라 "
                          "적용되는 요건을 위반한 것이다. 따라서 적용된다."}
    assert judgment_conclusion_issues(row) == []
    # 진짜 혼합 결론·반대 결론은 계속 검출
    mix = {"expected_judgment": "not_applicable",
           "gold_answer": "제1항은 적용되지만 제2항은 적용되지 않는다."}
    assert any("모두 포함" in i for i in judgment_conclusion_issues(mix))
    opp = {"expected_judgment": "not_applicable",
           "gold_answer": "본 시나리오에는 조항이 적용된다."}
    assert any("반대" in i for i in judgment_conclusion_issues(opp))


def test_ungrounded_standard_reference_gate():
    from scripts.aireg_kr.build_suite_qa import ungrounded_placeholder_standards
    # 재생성 #14 실측: 원문에 없는 'ISO 0000' 창작 — 한글 조사가 붙어도 검출
    ev_none = [{"quote": "", "article_text": "우리 선급이 인정하는 국제표준에 따른다."}]
    assert ungrounded_placeholder_standards(
        {"question": "ISO 0000을 적용한 시험", "gold_answer": "",
         "evidence": ev_none}) == ["ISO 0000"]
    assert ungrounded_placeholder_standards(
        {"question": "", "gold_answer": "ASME-0000에 따른다",
         "evidence": ev_none}) == ["ASME 0000"]
    # 원문에 실존하면 허용(원문 충실성) — 표기 변형(ISO-0000)도 동일 비교
    ev_has = [{"quote": "", "article_text": "ISO 0000에 따른다."}]
    assert ungrounded_placeholder_standards(
        {"question": "ISO-0000에 따른 설계인가?",
         "gold_answer": "원문에 따라 ISO 0000을 적용한다.", "evidence": ev_has}) == []
    # 실존 표준번호는 애초에 대상 아님
    assert ungrounded_placeholder_standards(
        {"question": "IEC 60092 및 ISO 6802", "gold_answer": "KS V ISO 9097",
         "evidence": ev_none}) == []


# ── ⑧ 기호 오적용 방지 (Zb/Zs 실측 대응) ────────────────────────────────
def test_foreign_symbols_gate():
    from scripts.aireg_kr.build_suite_qa import foreign_symbols
    # 원문 기호 유실 조항에서 질문·답변이 기호를 창작 — 검출(질문 순환 접지 차단)
    row = {"question": "Zs 값이 Zw 값보다 큰 경우의 판정은?",
           "gold_answer": "Zs 기준으로 면재측 순단면계수를 적용한다.",
           "evidence": [{"article_text": "…경우에는 는 면재측의 순단면계수로 한다."}]}
    assert set(foreign_symbols(row)) == {"Zs", "Zw"}
    # 원문 실재 기호(LaTeX 표기 포함)는 통과, 단위·원소는 미검출
    ok = {"question": "Kix 조건", "gold_answer": "Kix 값 적용, 5.0 Pa, CuNi 재질",
          "evidence": [{"article_text": "$K_{ix}$에 따른다"}]}
    assert foreign_symbols(ok) == []


def test_symbol_and_side_pairs():
    from scripts.aireg_kr.verify_suite import symbol_pairs, side_pairs
    art = "$Z_{b}$는 부착판측, $Z_{s}$는 면재측의 순단면계수로 한다."
    assert symbol_pairs(art, "Zs 기준으로 판정") == [("Zb", "Zs")]
    assert symbol_pairs(art, "Zb와 Zs 모두 대조") == []
    assert side_pairs("면재측 또는 부착판측의 값", "부착판측 값") == [("면재측", "부착판측")]


# ── ⑨ 학습 투입 질문 자연화 (구조 라벨 제거 — 채팅 붙여넣기 서빙 정합) ───
def test_naturalize_question_strips_labels():
    from scripts.aireg_kr.style_augment import naturalize_question
    q = ("판정하라.\n\n[발췌: OO호 설계 설명서 발췌]\n지름은 0.2 m로 설계되었다.")
    n = naturalize_question(q, "id1")
    assert "[발췌" not in n and n.endswith("지름은 0.2 m로 설계되었다.")
    q2 = "적용되는가?\n[대상 조항] 규칙 / 1604.\n\n[상황]\n전기추진설비 보유."
    n2 = naturalize_question(q2, "id2")
    assert "[상황]" not in n2 and "[대상 조항]" in n2 and "전기추진설비" in n2
    assert naturalize_question("블록 없음?", "id3") == "블록 없음?"


# ── ⑩ 검증 사유 분류(EVIDENCE/GENERATION) — 조기 REVIEW 오분류 대응 ─────
def test_issue_class_generation_vs_evidence():
    from scripts.aireg_kr.build_suite_qa import _issue_class
    # 답변 결함 서술('계산 누락'·'무관한 수치'·'사용함')은 재생성 대상
    gen = ["용어 '강갑판'이 원문 '강력갑판'과 일치하지 않음",
           "원문·질문에 존재하지 않는 수치를 사용함", "계산이 누락됨"]
    assert _issue_class(gen) == "GENERATION"
    assert _issue_class(["원문에 없는 수치를 사용함"]) == "GENERATION"
    # 소스 한계 전용 표현만 EVIDENCE(조기 REVIEW)
    assert _issue_class(["표의 값이 유실되어 도출될 수 없음"]) == "EVIDENCE"
    assert _issue_class(["근거(105.) 부재"]) == "EVIDENCE"


# ── ⑪ crossref 참조 내용 미전개 게이트 (40건 배치 A103 실측) ────────────
def test_xref_unexpanded_gate():
    from scripts.aireg_kr.build_suite_qa import xref_unexpanded
    main = {"article_text": "웰의 수직거리는 h/2 이상. 다만 이를 만족하지 못하면 3편 7장 101. 3항의 요건을 만족하여야 한다."}
    ref = {"article_text": "선박의 구조, 형상 및 용도 등으로 인하여 이중저 구조를 생략하고자 할 경우 우리 선급의 승인을 받아 생략할 수 있다."}
    # 번호만 반복 — 검출
    bad = {"track": "crossref", "evidence": [main, ref],
           "gold_answer": "해당 웰은 3편 7장 101. 3항의 요건을 만족하여야 한다."}
    assert xref_unexpanded(bad) is True
    # 참조 내용 전개 — 통과
    good = {"track": "crossref", "evidence": [main, ref],
            "gold_answer": "3편 7장 101. 3항에 따라 이중저 생략에 준하여 우리 선급의 승인을 받아야 한다. 구조·형상·용도를 고려한 승인이 전제된다."}
    assert xref_unexpanded(good) is False
    # 번호 언급 없는 서술형 답 — 비대상
    desc = {"track": "crossref", "evidence": [main, ref],
            "gold_answer": "선급의 승인을 받아 생략할 수 있다."}
    assert xref_unexpanded(desc) is False


# ── ⑫ 재량 단서 조건부 판정 (스모크 v2 코퍼댐 실측) ─────────────────────
def test_relief_clause_conditional_answer():
    from scripts.aireg_kr.build_suite_qa import _static_reasons, _find_relief
    sent = "간격은 600 mm 이상. 다만 인화점이 60°C를 넘는 선박은 적절히 참작하여도 좋다."
    assert _find_relief({"content": sent}, {"rule_sentence": sent})
    bad = {"track": "unit_convert", "expected_judgment": "non_compliant",
           "metadata": {}, "evidence": [{"quote": sent}],
           "gold_answer": "기준 미충족(non_compliant)."}
    assert "relief_clause_uncertain" in _static_reasons(bad)  # 단정 → 검수행
    ok = dict(bad, gold_answer="기준 미충족. 다만 단서('참작하여도 좋다')에 따라 "
                               "선급 참작 대상일 수 있어 확인 필요.")
    assert "relief_clause_uncertain" not in _static_reasons(ok)  # 조건부 → 통과


# ── ⑬ 검수 사이드카 표준화 (외부 검토 반영) ─────────────────────────────
def test_issue_codes_mapping():
    from scripts.aireg_kr.verify_suite import issue_codes
    real = ["1. 전제 일치 위반", "기준 3 위반: 용어", "6: 완전성", "2: 유일 도출"]
    assert issue_codes(real) == ["PREMISE_MISMATCH", "TERM_MISMATCH",
                                 "INCOMPLETE_ANSWER", "NOT_DERIVABLE_FROM_SOURCE"]
    assert issue_codes(["번호 없는 자유문장"]) == []


def test_quality_tier_on_finalize():
    from scripts.aireg_kr.build_suite_qa import _finalize
    r = {"metadata": {}}
    _finalize(r, "ACCEPT", [])
    assert r["quality"] == "silver"
    r2 = {"metadata": {"expert_reviewed": True}}
    _finalize(r2, "ACCEPT", [])
    assert r2["quality"] == "gold"


# ── ⑭ 검수 10차: 문서명 복구·전 트랙 범위 게이트·근거 추적성 ────────────
def test_doc_name_and_scope_gate():
    from scripts.aireg_kr.build_suite_qa import doc_name, _static_reasons
    assert doc_name({"_source_file": "대형요트 지침_2014_chunks"}) == "대형요트 지침(2014)"
    assert doc_name({"_source_file": "7편_2025_chunks"}) == "선급 및 강선규칙 7편(2025)"
    row = {"track": "hierarchy",
           "metadata": {"doc_subject": "대형요트", "premise_check": {"x": 1}},
           "question": "일반 화물선의 방화문에 2항이 적용되는가?"}
    assert "doc_scope_mismatch" in _static_reasons(row)
    ok = dict(row, question="대형요트 '오션드림호'의 방화문에 2항이 적용되는가?")
    assert "doc_scope_mismatch" not in _static_reasons(ok)


def test_evidence_meta_traceability():
    from scripts.aireg_kr.build_suite_qa import evidence_meta, provision_id
    assert provision_id("KR_10편_2025_RULE_P10_C7_S1_A103") == "KR:P10:C7:S1:A103"
    m = evidence_meta("600 mm 이상", "간격은 600 mm 이상이어야 한다.", "KR_x_RULE_P1_A1")
    assert m["char_start"] == 4 and m["char_end"] == 13
    assert m["normalized_quote"] == "600mm이상"


# ── ⑮ 검수 11차: 범위 게이트 자기 무력화 해소·스팬 역매핑·참조 정밀화 ────
def test_scope_gate_ignores_anchor_line():
    from scripts.aireg_kr.build_suite_qa import _static_reasons, _subject_aliases
    # 별칭은 코퍼스 도출 — 파일명 괄호 약어·'이하 X라 한다' 선언·접미 부분어
    al = _subject_aliases("부유식", "부유식 생산구조물 지침_2023_chunks",
                          "부유식 생산구조물(이하 구 조물이라 한다)이라 함은 …")
    assert "구조물" in al  # 공백 오염("구 조물")도 정규화 흡수
    row = {"track": "hierarchy",
           "metadata": {"doc_subject": "부유식", "premise_check": {"x": 1},
                        "doc_subject_aliases": sorted(al)},
           "question": ("해양경비함에 2항이 적용되는가?\n"
                        "[대상 조항] 부유식 생산구조물 지침(2023) · 규칙 / 302.\n\n"
                        "[상황]\n해양경비함이 소화장치를 설치했다."),
           "evidence": []}
    assert "doc_scope_mismatch" in _static_reasons(row)  # 앵커 문서명은 불인정
    ok = dict(row, question=row["question"].replace("해양경비함", "해당 구조물"))
    assert "doc_scope_mismatch" not in _static_reasons(ok)  # 자기 지칭 선언 허용
    osv = _subject_aliases("해상작업지원선", "해상작업지원선(OSV) 지침_2024_chunks", "")
    assert "OSV" in osv and "작업지원선" in osv  # 파일명 약어·접미 부분어


def test_ws_span_and_unresolved_gate():
    from scripts.aireg_kr.build_suite_qa import _ws_span, _static_reasons
    art = "간격은  600 mm\n이상이어야 한다."
    s, e = _ws_span("600 mm 이상", art)
    assert s == art.find("600") and art[s:e].replace(" ", "").replace("\n", "") == "600mm이상"
    assert _ws_span("없는 문구", art) == (-1, -1)
    bad = {"track": "spec", "metadata": {}, "question": "q",
           "evidence": [{"quote": "없는 문구", "char_start": -1}]}
    assert "evidence_span_unresolved" in _static_reasons(bad)


def test_def_ref_chapter_priority_and_skip():
    from scripts.aireg_kr.router import find_def_link
    terms = {"갑탱크": {"chunk_id": "DEF1", "chapter_no": "3",
                       "section_path": ["규칙", "401."],
                       "content": "1. 갑탱크라 함은 402.에 따른 탱크를 말한다.\n"
                                  "2. 을탱크라 함은 일반 탱크를 말한다."},
             "을탱크": {"chunk_id": "DEF1", "chapter_no": "3",
                       "section_path": ["규칙", "401."],
                       "content": "1. 갑탱크라 함은 402.에 따른 탱크를 말한다.\n"
                                  "2. 을탱크라 함은 일반 탱크를 말한다."}}
    r = {"chunk_id": "MAIN", "content": "갑탱크와 을탱크에는 밸브를 설치한다."}
    by_no = {"402": [{"chunk_id": "A402_C9", "chapter_no": "9", "content": "x",
                      "section_path": ["규칙"]},
                     {"chunk_id": "A402_C3", "chapter_no": "3", "content": "y",
                      "section_path": ["규칙"]}]}
    aux = find_def_link(r, terms, by_no)
    assert aux["def_ref_chunk_id"] == "A402_C3"  # 같은 장 우선
    aux2 = find_def_link(r, terms, {"402": []})
    assert aux2["term"] == "을탱크"  # 미해소 참조 용어는 스킵
