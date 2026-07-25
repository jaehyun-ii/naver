"""스위트 QA 통합 생성 — 조당 1건, 조 특성 라우팅(router.py).

모든 QA 트랙을 한 프로세스에서 생성한다. 라우터가 조 특성에 맞는 트랙을 조당
하나 고르고, 각 행은 해당 트랙의 파일에 기록된다(기존 스위트 파일 규약 유지):

  spec          스펙 조회형        → spec_qa.jsonl
  applicability 적용성 판단형      → applicability_qa.jsonl
  crossref      상호참조 해석형    → crossref_qa.jsonl
  def_link      정의 결합형        → def_link_qa.jsonl
  precedence    일반·특별 우선형   → precedence_qa.jsonl
  unit_convert  단위 환산 판정형   → unit_convert_qa.jsonl   (라벨은 프로그램 계산)
  table_lookup  표 참조형          → table_lookup_qa.jsonl
  hierarchy     조항호목 구조형    → hierarchy_qa.jsonl
  (+ --n-compare) 선급 비교형      → compare_qa.jsonl (build_crossref_qa 재사용)

정적 검증: 모든 인용(quote)은 원문 부분문자열 검증, 불일치 → needs_review.
unit_convert는 발췌 속 수치를 재파싱해 라벨을 재계산 — 불일치 계획은 폐기한다.
rule_cards.jsonl이 있으면 spec/applicability 프롬프트에 카드를 주입한다(없어도 동작).

    python -m scripts.aireg_kr.build_suite_qa --route-only   # 배정 분포만(LLM 없음)
    python -m scripts.aireg_kr.build_suite_qa --limit 50     # 생성
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

from . import prompts
from .common import (OUT_DIR, append_jsonl, chat_json, gen_meta, load_done,
                     load_jsonl, log, nfc, pmap, rule_uid)
from .condition_evaluator import convert_value, parse_number_with_unit
from .router import UNIT_DISPLAY, route_corpus

ROUTE_OUT = OUT_DIR / "suite_routing.jsonl"

# 트랙 → (출력 파일, task_type 표기)
TRACK_FILES = {
    "spec": ("spec_qa.jsonl", "스펙 조회형"),
    "applicability": ("applicability_qa.jsonl", "적용성 판단형"),
    "crossref": ("crossref_qa.jsonl", "상호참조 해석형"),
    "def_link": ("def_link_qa.jsonl", "정의 결합형"),
    "precedence": ("precedence_qa.jsonl", "일반·특별 우선형"),
    "formula_calc": ("formula_calc_qa.jsonl", "수식 계산형"),
    "figure_qa": ("figure_qa.jsonl", "그림 근거형"),
    "table_lookup": ("table_lookup_qa.jsonl", "표 참조형"),
    "hierarchy": ("hierarchy_qa.jsonl", "조항호목 구조형"),
}

# rule_cards.jsonl 역조인(chunk_id → card) — main에서 채운다. 없어도 동작.
CARDS: dict[str, dict] = {}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def quote_in(quote: str, text: str) -> bool:
    """인용이 원문에서 왔는지 검증 — 정확 일치 → 접두 일치 → 분절 일치 순.

    분절 모드(전문가 검수 2026-07-21 오탐 대응): 생략부호("…"/"...")로 발췌하거나
    수식 음역이 섞인 인용은 verbatim 실패 → 문장·생략부호 단위 조각으로 나눠
    15자 창이 원문에 실재하는 조각이 과반이면 인정한다(날조 인용은 15자 연속
    일치가 나오지 않는다)."""
    q, t = norm(quote), norm(text)
    if not q:
        return False
    if q in t or (len(q) >= 40 and q[:40] in t):
        return True
    frags = [f for f in re.split(r"\.{3}|…|(?<=다\.)", q) if len(f) >= 8]
    if len(frags) < 2:
        return False
    hit = sum(1 for f in frags
              if any(f[i:i + 15] in t for i in range(0, max(1, len(f) - 14), 5)))
    return hit >= 1 and hit / len(frags) >= 0.5


def sp(rule: dict) -> str:
    return " / ".join(rule.get("section_path") or [])


def stable(rid: str) -> int:
    return int(hashlib.sha256(rid.encode()).hexdigest(), 16)


def card_json(rule: dict) -> str:
    card = CARDS.get(rule["chunk_id"])
    return json.dumps(card, ensure_ascii=False) if card else "{}"


def risk_block(rule: dict) -> str:
    """Rule Card의 혼동 위험(risk_of_misinterpretation)·모호성(ambiguity)을
    생성 프롬프트에 경고로 주입 — 유사 용어 치환류 오류의 사전 예방."""
    card = CARDS.get(rule["chunk_id"]) or {}
    notes = ((card.get("risk_of_misinterpretation") or [])
             + (card.get("ambiguity") or []))
    if not notes:
        return ""
    return ("\n\n[혼동 주의 — 생성 시 반드시 반영]\n"
            + "\n".join(f"- {n}" for n in notes[:5])
            + "\n위 지점의 용어·조건·대상을 서로 바꿔 쓰지 마십시오. "
            "원문이 어느 용어에 대해 규정한 내용인지 정확히 유지하십시오.")


# ── 시나리오 유사어 치환 정적 검사 (생성 직후, 검증 LLM 이전의 무료 게이트) ──
# 실측 실패 사례: 원문 "폐위된 구역"을 시나리오가 "밀폐된 제어실"로 바꿔 써서
# 원문만으로 적용 여부가 도출되지 않는 문제가 생성됨(hy3 REJECT).
_PARA_TOKEN = re.compile(r"([가-힣]{2,4})(?:된|한|되는|하는)\s*[가-힣]")
_ART_TOKEN = re.compile(r"[가-힣]{2,7}")
_PARA_GENERIC = {"설치", "탑재", "승인", "구성", "운항", "사용", "적용", "위치",
                 "연결", "부착", "설계", "제작", "요구", "충족", "만족", "발생",
                 "발견", "확인", "유지", "제외", "포함", "진행", "관련", "인접",
                 "해당", "구비", "장착", "적재", "예정", "완료", "제공"}


def paraphrase_suspects(scenario: str, article: str) -> list[tuple[str, str]]:
    """시나리오의 수식어("X된/X한 N")가 원문에 없고 원문에 근사어가 있으면 의심.

    근사어: 길이 ±1, 글자 집합을 (len-1)자 이상 공유 — 밀폐↔폐위된 류를 겨냥한
    보수적 휴리스틱. 과검출은 재생성 1회 비용에 그친다."""
    sus = []
    art_tokens = set(_ART_TOKEN.findall(article))
    for m in _PARA_TOKEN.finditer(scenario or ""):
        x = m.group(1)
        if x in _PARA_GENERIC or x in article:
            continue
        for u in art_tokens:
            if (abs(len(u) - len(x)) <= 1 and u not in (scenario or "")
                    and len(set(x) & set(u)) >= max(1, len(x) - 1)):
                sus.append((x, u))
                break
    return sus


def chat_scenario(prompt: str, article: str, extract, max_tokens: int,
                  tries: int = 3, essential=None) -> tuple[dict, list]:
    """시나리오 생성 + 유사어 정적 검사 — 위반 시 지적을 덧붙여 재생성.

    잘린 출력의 부분 파싱(선두 필드 premise_check만 회수돼 scenario/answer가
    빈 obj)은 essential 검사로 걸러 재시도한다 — 사고 분량 변동으로 간헐 발생.
    재생성이 퇴화하면 마지막 완전 출력을 반환해 ERROR(전량 유실)를 면한다."""
    warn, best, best_sus = "", {}, []
    fails = 0
    for _ in range(tries):
        # essential 실패(잘림)마다 한도 상향 — 재작성 지시가 붙는 라운드는 사고
        # 분량이 늘어 같은 한도로는 계속 잘린다(A107 ERROR 연쇄 실측)
        obj = chat_json(prompt + warn, max_tokens=max_tokens + 800 * min(fails, 2))
        ok = essential(obj) if essential else bool(str(extract(obj) or "").strip())
        if not ok:
            fails += 1
            continue  # 잘림 부분 파싱·빈 출력 — warn 유지, 한도 올려 재시도
        sus = paraphrase_suspects(extract(obj), article)
        if not sus:
            return obj, []
        best, best_sus = obj, sus
        x, u = sus[0]
        warn = (f"\n\n[재작성 지시] 직전 생성의 시나리오에 원문에 없는 유사어 "
                f"'{x}'이(가) 사용되었습니다. 원문의 용어(예: '{u}')를 그대로 "
                "사용해 전체를 다시 작성하십시오.")
    return best, best_sus


def doc_name(rule: dict) -> str:
    """실제 문서명 복구 — 청크 doc_title은 전 문서 동일(통합 표제, 검수 10차)이라
    _source_file에서 파생한다: '대형요트 지침_2014_chunks' → '대형요트 지침(2014)',
    '7편_2025_chunks' → '선급 및 강선규칙 7편(2025)'."""
    stem = nfc(rule.get("_source_file") or "").replace("_chunks", "")
    m = re.match(r"(.+?)_(\d{4})$", stem)
    name, year = (m.group(1), m.group(2)) if m else (stem, "")
    if re.match(r"\d+편", name):
        name = f"선급 및 강선규칙 {name}"
    return f"{name}({year})" if year else name


# 조 앵커 원칙 — 항·호·목·표 참조는 조가 특정돼야 의미가 생긴다. 질문 단독으로
# 대상 조항이 식별되게 경로를 명시한다(RAFT 밖 사용·검색 정합·검수 가독성).
_Q_ANCHORED = re.compile(  # 이미 조 식별자가 있는 질문
    r"「|\[대상 조항\]|\d{3,4}\.(?!\d)|\d+\.\d+(?:\.\d+)+|\d+\s*조")
_BARE_REF = re.compile(  # 조 없이 부유하는 참조
    r"\d+\s*항|\(\d+\)\s*호|\([가-힣]\)\s*목|표\s*\d|그림\s*\d")


def anchor_question(q: str, rule: dict, always: bool = False) -> str:
    """질문에 조 앵커([대상 조항] 경로) 명시. 이미 식별자가 있으면 그대로.

    always=True: 조항 스코프 트랙(hierarchy·precedence·table·unit) — 질문이
    조항 내부를 전제하므로 무조건. False: 주제형 트랙(spec·crossref·def_link) —
    부유 참조(항·호·목·표·그림)가 있을 때만."""
    if not q or _Q_ANCHORED.search(q):
        return q
    if not always and not _BARE_REF.search(q):
        return q
    return f"{q.rstrip()}\n[대상 조항] {doc_name(rule)} · {sp(rule)}"


def refs_block(rule: dict, aux: dict | None = None) -> str:
    """원문에 실재하는 표·그림·조항 표기 목록 주입 — foreign_reference 사전 예방.

    foreign_refs 게이트는 사후 검출·재생성만 한다(콜 낭비·REJECT 원인). 실재
    표기를 먼저 보여주면 첫 시도에서 창작 표기('표 3.14.4'류)가 줄어든다.
    목록 추출은 게이트(_REF_PAT)와 동일 어휘라 지시-게이트 정합이 보장된다."""
    src = rule.get("content", "")
    for k in ("ref_text", "def_text", "counterpart_text", "leaf_text"):
        if aux and aux.get(k):
            src += "\n" + str(aux[k])
    toks = sorted({m.group(0).strip() for m in _REF_PAT.finditer(src)})
    if not toks:
        return ("\n\n[참조 표기 규칙]\n- 위 원문에는 표·그림 번호 표기가 없습니다 — "
                "답변에서 표·그림·타 조항 번호를 창작하지 마십시오.")
    return ("\n\n[원문에 실재하는 참조 표기 — 표·그림·타 조항 지목은 이 목록 "
            "안에서만]\n" + ", ".join(toks[:20])
            + "\n이 목록에 없는 표·그림·조항 번호를 만들어 쓰면 그 문제는 "
            "무효입니다. 항 표기는 원문 그대로 쓰고 합성('401.1항')하지 마십시오.")


def doc_scope_block(rule: dict) -> str:
    """시나리오 트랙(적용성·조항호목·우선규정) 공통 — 문서 적용범위 상속 + 다양화.

    실측 편향 2건 대응: ① 문서 적용범위 밖 선종(부유식 구조물 지침에 컨테이너선)
    ② 상투 선명('블루스타' 8/9)·동일 톤수 반복."""
    subject = _doc_subject(rule)
    subj_line = (f"\n- 이 문서는 「{subject}」 전용입니다 — 시나리오의 대상은 "
                 f"반드시 {subject} 계열이어야 하며, 일반 화물선·컨테이너선 등 "
                 "다른 선종을 쓰면 그 문제는 무효입니다." if subject else "")
    return (f"\n\n[문서 맥락 — 시나리오 작성 규칙]\n"
            f"- 이 조항은 「{doc_name(rule)}」 소속입니다. 시나리오의 선박·설비"
            " 종류는 이 문서의 적용 대상 범위에서 고르십시오 — 문서가 특정"
            " 선종·구조물 전용이면 다른 선종을 쓰지 마십시오." + subj_line + "\n"
            "- 선명·회사명·톤수는 조항마다 서로 다른 허구 값을 쓰고, 흔한 상투적"
            " 이름의 반복을 피하십시오.")


_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """조항 원문의 실제 언어 — 메타(language)가 문서 단위라 영문 편(15편 등)이
    ko로 오기되는 문제 대응(전문가 검수 2026-07-21)."""
    t = (text or "")[:2000]
    hangul, latin = len(_HANGUL_RE.findall(t)), len(_LATIN_RE.findall(t))
    if hangul + latin < 20:
        return "unknown"
    return "ko" if hangul >= latin else "en"


def provision_id(chunk_id: str) -> str:
    """chunk_id → 전역 조항 식별자('KR:P10:C7:S1:A103') — 검수 10차 권고."""
    if "_RULE_" not in (chunk_id or ""):
        return ""
    pub = chunk_id.split("_", 1)[0]
    return pub + ":" + ":".join(chunk_id.split("_RULE_", 1)[1].split("_"))


_ELLIPSIS_SPLIT = re.compile(r"\s*(?:\.{3,}|…|⋯)\s*")


def _ws_span(quote: str, article: str) -> tuple[int, int]:
    """원문 좌표의 인용 스팬 — raw 실패 시 공백 제거 맵으로 역매핑(검수 11차).

    char_start/end는 항상 **비정규화 원문** 기준. 공백·개행 차이만 흡수하며,
    '...'·'…' 연결 인용은 각 조각이 순서대로 원문에 실재할 때만 전체를 덮는
    스팬으로 해소한다(v2 배치 실측: 발췌 연결이 미해소 검수행의 주 원인).
    그래도 못 찾으면 (-1, -1) → evidence_span_unresolved 게이트가 검수행 처리."""
    if not quote:
        return -1, -1
    i = article.find(quote)
    if i >= 0:
        return i, i + len(quote)
    stripped, idx = [], []
    for j, ch in enumerate(article):
        if not ch.isspace():
            stripped.append(ch)
            idx.append(j)
    hay = "".join(stripped)

    def find_ws(part: str, start: int = 0) -> tuple[int, int] | None:
        needle = "".join(c for c in part if not c.isspace())
        if not needle:
            return None
        k = hay.find(needle, start)
        return (k, k + len(needle)) if k >= 0 else None

    m = find_ws(quote)
    if m:
        return idx[m[0]], idx[m[1] - 1] + 1
    parts = [p for p in _ELLIPSIS_SPLIT.split(quote) if p.strip()]
    if len(parts) >= 2:
        spans, k = [], 0
        for p in parts:
            m = find_ws(p, k)
            if m is None:
                return -1, -1
            spans.append(m)
            k = m[1]
        return idx[spans[0][0]], idx[spans[-1][1] - 1] + 1
    return -1, -1


def _snap_quote(quote: str, article: str) -> str | None:
    """미해소 인용의 원문 표기 교정(quote snapping — v3 배치 실측 대응).

    모델이 인용 중간의 어미·기호를 미세 변형하는 유형은 프롬프트로 근절되지
    않는다. 공백 무시 유사도 85% 이상인 연속 원문 구간이 있으면 그 구간을
    원문 그대로 반환한다(LLM 콜 없는 결정적 후처리). 멀리 떨어진 조각을 하나로
    잇는 오스냅은 매칭 폭 상한(인용의 1.5배)으로 차단. 실패 시 None → 기존
    evidence_span_unresolved 경로 유지."""
    import difflib
    if not quote or not article:
        return None
    stripped, idx = [], []
    for j, ch in enumerate(article):
        if not ch.isspace():
            stripped.append(ch)
            idx.append(j)
    hay = "".join(stripped)
    needle = "".join(c for c in quote if not c.isspace())
    if len(needle) < 10 or not hay:
        return None
    sm = difflib.SequenceMatcher(None, hay, needle, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    if not blocks:
        return None
    a0, a1 = blocks[0].a, blocks[-1].a + blocks[-1].size
    matched = sum(b.size for b in blocks)
    if matched < 0.85 * len(needle) or (a1 - a0) > 1.5 * len(needle):
        return None
    return article[idx[a0]:idx[a1 - 1] + 1]


def evidence_meta(quote: str, article_text: str, chunk_id: str) -> dict:
    """근거 추적성 필드(검수 10·11차) — 원문 좌표 스팬·정규화 인용·해시·식별자.

    스팬 미해소 시 quote snapping을 시도해 인용을 원문 표기로 치환한다 —
    반환 dict의 "quote"가 호출부 dict의 원본 quote를 덮어쓰고(전 호출부가
    **evidence_meta를 quote 뒤에 spread), 원본은 quote_raw로 보존된다."""
    import hashlib
    start, end = _ws_span(quote, article_text)
    out = {}
    if quote and start < 0:
        snapped = _snap_quote(quote, article_text)
        if snapped:
            out = {"quote": snapped, "quote_raw": quote, "quote_snapped": True}
            quote = snapped
            start, end = _ws_span(quote, article_text)
    nq = norm(quote) if quote else ""
    return {**out, "normalized_quote": nq,
            "char_start": start, "char_end": end,
            "normalization_version": "ws-map-v1",
            "quote_hash": hashlib.sha1(nq.encode()).hexdigest()[:12] if nq else "",
            "provision_id": provision_id(chunk_id),
            "evidence_language": detect_language(article_text)}


def evidence_entry(rule: dict, quote: str = "") -> dict:
    # 기존 스위트 파일과 동일하게 section_path는 문자열(" / " 결합)
    card = CARDS.get(rule["chunk_id"]) or {}
    req_ids = [q.get("requirement_id") for q in card.get("requirements") or []
               if q.get("requirement_id")]
    # 폴백(검수 11차): 카드 미보유 조항은 조 단위 식별자라도 채운다
    req_ids = req_ids or [provision_id(rule["chunk_id"])]
    return {"chunk_id": rule["chunk_id"], "source_file": rule.get("_source_file", ""),
            "section_path": sp(rule), "quote": quote,
            "requirement_ids": req_ids, "article_text": rule["content"],
            **evidence_meta(quote, rule["content"], rule["chunk_id"])}


def base_row(rule: dict, track: str, prompt_id: str) -> dict:
    src_lang = detect_language(rule.get("content", ""))
    return {"question_id": f"{rule_uid(rule)}::suite", "track": track,
            "task_type": TRACK_FILES[track][1],
            "metadata": {"publisher": rule.get("publisher", "KR"),
                         "rule_language": rule.get("language", "ko"),
                         # 실감지 언어 — rule_language(문서 메타)와 분리 기록
                         "source_language": src_lang,
                         "cross_lingual": src_lang == "en",
                         # QA 쌍은 항상 모델 생성 — synthetic(시나리오 창작 여부)과 분리
                         "qa_synthetic": True,
                         # doc_title은 파일명 파생(청크 표제는 전 문서 동일 오염)
                         "doc_title": doc_name(rule),
                         "doc_title_raw": rule.get("doc_title", ""),
                         "doc_subject": _doc_subject(rule),
                         "doc_subject_aliases": sorted(_subject_aliases(
                             _doc_subject(rule), rule.get("_source_file", ""),
                             rule.get("_doc_scope_text", "")))[:12]
                         if _doc_subject(rule) else [],
                         "article_no": rule.get("article_no"),
                         "article_title": rule.get("article_title"),
                         "synthetic": True},
            "_gen": gen_meta(prompt_id)}


# ── spec (fallback) ─────────────────────────────────────────────────────
def gen_spec(rule: dict, aux: dict | None, feedback: str = "") -> dict | None:
    obj = chat_json(prompts.SPEC_QA.format(
        n_qa=1, rule_card=card_json(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + feedback, max_tokens=1600)
    qa = (obj.get("qa") or [{}])[0]
    if not qa.get("question") or not qa.get("answer"):
        return None
    row = base_row(rule, "spec", "SPEC_QA")
    row.update({
        "question": anchor_question(qa["question"], rule),
        "gold_answer": qa["answer"],
        "evidence": [evidence_entry(rule, qa.get("evidence_quote", ""))],
        "needs_review": not quote_in(qa.get("evidence_quote", ""), rule["content"]),
        "metadata": {**row["metadata"], "synthetic": False}})
    return row


# ── applicability (fallback) ────────────────────────────────────────────
APPLIC_LABELS = {"applicable", "not_applicable", "conditional"}


def _doc_subject(rule: dict) -> str:
    """특수 문서(선종·설비 전용 규칙/지침)의 대상어 — 파일명 첫 토큰.

    편-시리즈(강선규칙 본체)는 전 선종 대상이라 검사하지 않는다. 실측 결함:
    플로팅독 규칙 조항에 '일반 화물선' 시나리오가 applicable로 라벨됨 —
    조항 청크에 선종 제한이 없어도 상위 문서 적용범위가 우선한다."""
    stem = nfc(rule.get("_source_file") or "")
    if not stem or stem[0].isdigit():
        return ""
    tok = re.split(r"[\s_(]", stem)[0]
    return tok if len(tok) >= 3 else ""


def gen_applicability(rule: dict, aux: dict | None, feedback: str = "") -> dict | None:
    obj, sus = chat_scenario(
        prompts.APPLICABILITY.format(
            rule_card=card_json(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + doc_scope_block(rule) + feedback, rule["content"],
        lambda o: " ".join(i.get("scenario", "") for i in o.get("items", [])),
        max_tokens=3600,  # premise_check(항목당)+사고 분량
        essential=lambda o: any(str(i.get("scenario") or "").strip()
                                and str(i.get("reason") or "").strip()
                                for i in o.get("items") or []))
    valid = [i for i in obj.get("items", [])
             if i.get("scenario") and i.get("label") in APPLIC_LABELS]
    if not valid:
        return None
    # 라벨 교대 — 첫 항목만 취하면 applicable로 쏠린다(실측 9/9). 조 해시로
    # 절반은 미적용·조건부 시나리오를 선택해 라벨 분포를 균형시킨다.
    want_applicable = stable(rule_uid(rule)) % 2 == 0
    item = next((i for i in valid
                 if (i["label"] == "applicable") == want_applicable), valid[0])
    quote = item.get("evidence_quote", "")
    # 상위 문서 적용범위 정합 — 특수 문서인데 시나리오에 대상어가 없으면서
    # applicable이면 문서 범위 밖 대상일 수 있다(검수행)
    subject = _doc_subject(rule)
    scope_suspect = bool(subject and item["label"] == "applicable"
                         and subject not in item["scenario"])
    row = base_row(rule, "applicability", "APPLICABILITY")
    row.update({
        "question": (f"다음 선박/설비에 「{sp(rule)}」이 적용되는가? "
                     "적용 조건을 원문에 근거해 판단하라.\n\n[선박/설비 개요]\n"
                     + item["scenario"]),
        "expected_judgment": item["label"],
        "gold_answer": item.get("reason", ""),
        "evidence": [evidence_entry(rule, quote)],
        "needs_review": bool(sus) or scope_suspect
        or not quote_in(quote, rule["content"]),
        "metadata": {**row["metadata"], "doc_scope_suspect": scope_suspect,
                     "premise_check": {k: item[k] for k in
                                       ("premise_clauses", "premise_facts")
                                       if item.get(k)},
                     "paraphrase_suspects": [list(s) for s in sus]}})
    return row


# ── crossref ────────────────────────────────────────────────────────────
def gen_crossref(rule: dict, aux: dict, feedback: str = "") -> dict | None:
    obj = chat_json(prompts.CROSSREF_QA.format(
        main_path=sp(rule), main_text=rule["content"],
        ref_path=aux["ref_path"], ref_text=aux["ref_text"])
        + risk_block(rule) + refs_block(rule, aux) + feedback, max_tokens=1800)
    if not obj.get("question") or not obj.get("answer"):
        return None
    mq, rq = obj.get("main_quote", ""), obj.get("ref_quote", "")
    row = base_row(rule, "crossref", "CROSSREF_QA")
    row.update({
        "question": anchor_question(obj["question"], rule),
        "gold_answer": obj["answer"],
        "evidence": [
            evidence_entry(rule, mq),
            {"chunk_id": aux["ref_chunk_id"],
             "source_file": aux.get("ref_source_file") or rule.get("_source_file", ""),
             "section_path": aux["ref_path"], "quote": rq,
             "requirement_ids": [provision_id(aux["ref_chunk_id"])],
             "article_text": aux["ref_text"],
             **evidence_meta(rq, aux["ref_text"], aux["ref_chunk_id"])}],
        "needs_review": not (quote_in(mq, rule["content"]) and quote_in(rq, aux["ref_text"])),
        "metadata": {**row["metadata"], "ref_chunk_id": aux["ref_chunk_id"],
                     "xdoc_ref": bool(aux.get("xdoc")),
                     "synthetic": False}})
    return row


# ── def_link ────────────────────────────────────────────────────────────
def gen_def_link(rule: dict, aux: dict, feedback: str = "") -> dict:
    # 참조 체인 폐쇄 — 정의가 참조하는 조항이 해소됐으면 3번째 근거로 동봉
    ref_block = (prompts.DEF_LINK_REF_BLOCK.format(
        ref_path=aux["def_ref_path"], ref_text=aux["def_ref_text"])
        if aux.get("def_ref_text") else "")
    obj = chat_json(prompts.DEF_LINK_QA.format(
        term=aux["term"], def_path=aux["def_path"], definition_text=aux["def_text"],
        def_ref_block=ref_block,
        section_path=sp(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + feedback, max_tokens=2200)
    dq, rq = obj.get("definition_quote", ""), obj.get("requirement_quote", "")
    drq = obj.get("definition_ref_quote", "") if aux.get("def_ref_text") else ""
    quotes_ok = quote_in(dq, aux["def_text"]) and quote_in(rq, rule["content"])
    # 3-근거 문제는 참조 조항 인용도 필수·검증(검수 11차)
    if aux.get("def_ref_text"):
        quotes_ok = quotes_ok and bool(drq) and quote_in(drq, aux["def_ref_text"])
    # 질문에 정의 원문이 새면 2-근거 문제가 무너진다(정의 없이도 풀리게 됨)
    leak = bool(dq) and norm(dq)[:20] in norm(obj.get("question", ""))
    row = base_row(rule, "def_link", "DEF_LINK_QA")
    row.update({
        "question": anchor_question(obj.get("question", ""), rule),
        "gold_answer": obj.get("answer", ""),
        "evidence": [
            {"chunk_id": aux["def_chunk_id"], "source_file": rule.get("_source_file", ""),
             "section_path": aux["def_path"], "quote": dq,
             "requirement_ids": [provision_id(aux["def_chunk_id"])],
             "article_text": aux["def_text"],
             **evidence_meta(dq, aux["def_text"], aux["def_chunk_id"])},
            evidence_entry(rule, rq)] + ([
            {"chunk_id": aux["def_ref_chunk_id"],
             "source_file": rule.get("_source_file", ""),
             "section_path": aux["def_ref_path"], "quote": drq,
             "requirement_ids": [provision_id(aux["def_ref_chunk_id"])],
             "article_text": aux["def_ref_text"],
             **evidence_meta(drq, aux["def_ref_text"], aux["def_ref_chunk_id"])}]
            if aux.get("def_ref_text") else []),
        "needs_review": not quotes_ok or leak,
        "metadata": {**row["metadata"], "term": aux["term"], "synthetic": False}})
    return row


# ── precedence ──────────────────────────────────────────────────────────
# 재량 표현·확정 결론 유보 표현 — 전문가 검수 6차("특별히 고려할 수 있다" 실측)
_DISCRETION = re.compile(r"고려할 수 있|인정할 수 있|허용할 수 있|승인에 따라"
                         r"|인정하는 경우|승인을 받아")
_HEDGE = re.compile(r"승인|인정|재량|검토를 전제|될 수 있|판단할 수 없")


def gen_precedence(rule: dict, aux: dict, feedback: str = "") -> dict:
    if aux.get("counterpart_text"):
        block = prompts.PRECEDENCE_GENERAL_BLOCK.format(
            counterpart_path=aux["counterpart_path"], counterpart_text=aux["counterpart_text"])
    else:
        block = prompts.PRECEDENCE_SELF_BLOCK
    obj, sus = chat_scenario(
        prompts.PRECEDENCE_QA.format(
            marker_sentence=aux["marker_sentence"], general_block=block,
            section_path=sp(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + doc_scope_block(rule) + feedback, rule["content"],
        lambda o: o.get("scenario", ""), max_tokens=2400,
        essential=lambda o: bool(str(o.get("scenario") or "").strip()
                                 and str(o.get("answer") or "").strip()))
    sq, gq = obj.get("special_quote", ""), obj.get("general_quote", "")
    gq_src = aux.get("counterpart_text") or rule["content"]
    q_anchored = anchor_question(obj.get("question", ""), rule, always=True)
    q = (q_anchored if not obj.get("scenario")
         else f"{q_anchored}\n\n[상황]\n{obj['scenario']}")
    # v2: 본문/단서 분기 선판정(branch_check) — 없거나 판정값이 불량하면 검수행
    # v3: 관계 세분화 — 배제·대체 결론에는 명시적 배제 관계(relation)가 필요
    bc = obj.get("branch_check") or {}
    bc_ok = bc.get("matched") in ("general", "special", "both")
    rel = str(bc.get("relation") or "").upper()
    ans_text = obj.get("answer", "")
    exclusion_claim = any(k in ans_text for k in ("배제", "대체되", "적용하지 아니",
                                                  "적용되지 않는다"))
    rel_ok = rel in ("OVERRIDES", "SUBSTITUTES", "EXCEPTION", "LIMITED_EXCEPTION",
                     "DISCRETIONARY", "ADDITIONAL", "UNCLEAR")
    rel_mismatch = exclusion_claim and rel not in ("OVERRIDES", "SUBSTITUTES", "EXCEPTION")
    # 재량 표현("고려할 수 있다") 특별 규정을 확정 배제로 결론 — 전문가 검수 6차 실측
    disc_src = bool(_DISCRETION.search(
        (aux.get("marker_sentence") or "") + (bc.get("special_condition") or "")
        + (bc.get("explicit_marker") or "")))
    disc_mismatch = ((disc_src or rel == "DISCRETIONARY") and exclusion_claim
                     and not _HEDGE.search(ans_text))
    row = base_row(rule, "precedence", "PRECEDENCE_QA")
    ev = [evidence_entry(rule, sq)]
    if aux.get("counterpart_chunk_id"):
        ev.append({"chunk_id": aux["counterpart_chunk_id"],
                   "source_file": rule.get("_source_file", ""),
                   "section_path": aux.get("counterpart_path") or "",
                   "quote": gq, "requirement_ids": [],
                   "article_text": aux["counterpart_text"]})
    row.update({
        "question": q, "gold_answer": obj.get("answer", ""), "evidence": ev,
        "needs_review": (bool(sus) or not bc_ok or not rel_ok or rel_mismatch
                         or disc_mismatch
                         or not (quote_in(sq, rule["content"]) and quote_in(gq, gq_src))),
        "metadata": {**row["metadata"], "marker": aux["marker"],
                     "has_counterpart": bool(aux.get("counterpart_chunk_id")),
                     "branch_check": bc, "relation_mismatch": rel_mismatch,
                     "discretionary_mismatch": disc_mismatch,
                     "paraphrase_suspects": [list(s) for s in sus]}})
    return row


# ── unit_convert — 라벨은 프로그램 계산, LLM은 발췌 렌더링만 ────────────
SAT = {"gte": {"below": False, "at": True, "above": True},
       "gt": {"below": False, "at": False, "above": True},
       "lte": {"below": True, "at": True, "above": False},
       "lt": {"below": True, "at": False, "above": False}}
_EVAL_UNIT = {"시간": "h"}  # evaluator 변환표 등록명


def fmt_num(x: float) -> str:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


# 재량 완화 단서 — 기준 미달이어도 선급 재량으로 허용될 수 있는 조항(스모크 v2
# 실측: 코퍼댐 600mm + "다만 인화점 60°C 초과 … 적절히 참작하여도 좋다").
# 이런 조항의 non_compliant 판정은 유일하게 확정되지 않는다.
_RELIEF = re.compile(r"참작하여도 좋다|참작할 수 있다|완화할 수 있다|적절히 참작"
                     r"|선급이 인정하는 경우|승인을 받아.{0,14}(?:완화|생략|달리)")


def _find_relief(rule: dict, aux: dict) -> str:
    """기준 문장 자체 또는 직후 단서(200자)에서 재량 완화 표현 탐색."""
    sent = aux.get("rule_sentence") or ""
    m = _RELIEF.search(sent)
    if m:
        return m.group(0)
    content = rule.get("content", "")
    i = content.find(sent[:40])
    if i >= 0:
        m2 = _RELIEF.search(content[i + len(sent):i + len(sent) + 200])
        if m2:
            return m2.group(0)
    return ""


def gen_unit_convert(rule: dict, aux: dict, feedback: str = "") -> dict | None:
    thr, unit, op = aux["threshold"], aux["unit"], aux["op"]
    eu = _EVAL_UNIT.get(unit, unit)
    conv = aux["conv_unit"]
    # 라벨 균형 — 변형을 균등 추첨하면 gte 계열에서 2/3가 적합으로 쏠린다
    # (실측 7/8). 목표 라벨(적합/부적합)을 먼저 반씩 정하고 그에 맞는 변형을 고른다.
    h = stable(rule_uid(rule))
    want_sat = h % 2 == 0
    relief = _find_relief(rule, aux)
    cands = [v for v in ("below", "at", "above") if SAT[op][v] == want_sat]
    variant = cands[(h // 2) % len(cands)] if cands else "at"
    for v_try in (variant, "at"):
        orig_val = thr * {"below": 0.9, "at": 1.0, "above": 1.1}[v_try]
        conv_val = convert_value(orig_val, eu, _EVAL_UNIT.get(conv, conv))
        if conv_val is None:
            return None
        fact_str = fmt_num(conv_val)
        # 라벨 재계산: 표기된 문자열을 재파싱→역환산→비교. 표기 반올림으로
        # 라벨이 계획과 달라지면 그 변형은 버린다(at은 정확 환산이라 안전).
        n, _ = parse_number_with_unit(fact_str)
        back = convert_value(n, _EVAL_UNIT.get(conv, conv), eu)
        sat = {"gte": back >= thr, "gt": back > thr,
               "lte": back <= thr, "lt": back < thr}[op]
        if sat == SAT[op][v_try]:
            variant = v_try
            break
    else:
        return None
    conv_disp = UNIT_DISPLAY.get(conv, conv)
    qty = (aux.get("quantity") or "").strip()
    obj = chat_json(prompts.UNIT_EXCERPT.format(
        fact_value=fact_str, fact_unit=conv_disp,
        fact_name=f"'{qty}'" if qty else "해당 항목",
        rule_sentence=aux["rule_sentence"]) + feedback, max_tokens=1200)
    excerpt = obj.get("excerpt", "")
    fact_ok = norm(f"{fact_str}{conv_disp}") in norm(excerpt)
    # 수치가 규정의 물리량이 아닌 다른 항목 값으로 붙으면 문제 자체가 무효
    qty_ok = not qty or norm(qty) in norm(excerpt)
    label = "compliant" if sat else "non_compliant"
    unit_disp = UNIT_DISPLAY.get(unit, unit)
    row = base_row(rule, "unit_convert", "UNIT_EXCERPT")
    row.update({
        "question": (f"다음 제출문서 발췌가 근거 조항 「{sp(rule)}」의 요건을 "
                     "충족하는지 판정하라. 근거 문구를 인용하고 필요한 단위 환산 "
                     "과정을 명시하라."
                     f"\n\n[발췌: {obj.get('excerpt_title', '')}]\n{excerpt}"),
        "expected_judgment": label,
        # 재량 단서 조항의 미충족은 단정 대신 조건부 서술 — 부족한 것은 다른
        # 조항이 아니라 사안 사실(선급 참작 여부)이라 RAG로도 확정 불가.
        # 실서빙이 원하는 행동: 원칙 판정 + 재량 경로 안내(단정 금지).
        # 부등식 명시(v2a): 비교 방향을 기호로 고정해 경계 방향 오류를 차단한다.
        "gold_answer": (f"기준: '{aux['rule_sentence']}' — {fmt_num(thr)} {unit_disp} "
                        f"{aux['op_kr']}. 환산: 발췌의 설계값 {fact_str} {conv_disp}"
                        f" = {fmt_num(back)} {unit_disp}. "
                        f"비교: {fmt_num(back)} {unit_disp} "
                        f"{'>' if back > thr else ('<' if back < thr else '=')} "
                        f"기준 {fmt_num(thr)} {unit_disp} (요구: {aux['op_kr']}) "
                        f"→ 기준 {'충족(compliant)' if sat else '미충족(non_compliant)'}."
                        + (f" 다만 원문 단서('{relief}')에 따라 선급의 참작·인정 "
                           "대상이 될 수 있으므로, 단서 해당 여부와 선급 승인 "
                           "여부를 별도로 확인하여야 한다."
                           if relief and not sat else "")),
        "evidence": [evidence_entry(rule, aux["rule_sentence"])],
        "needs_review": not fact_ok or not qty_ok,
        "program": {"threshold": thr, "unit": unit, "operator": op,
                    "variant": variant, "fact_value": fact_str, "fact_unit": conv,
                    "quantity": qty, "label_source": "program",
                    "relief_clause": relief}})
    return row


# ── table_lookup ────────────────────────────────────────────────────────
def norm_table(s: str) -> str:
    """표 셀 대조용 정규화 — HTML 엔티티·LaTeX 래퍼 제거(전문가 검수 7차 오탐 대응).

    실측 오탐: 원문 "10년 &lt; 선령"(엔티티) vs 선택자 "10년 < 선령",
    원문 "$37 \\text{ m}^{2}$" vs 값 "37 m²"."""
    import html
    s = html.unescape(s or "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"[\$\\{}]|\^", "", s)
    s = s.replace("²", "2").replace("³", "3")
    return norm(s)


def _cell_in(cell: str, body: str) -> bool:
    """셀 값이 원문 표에 실재하는지 — 정확 일치 → 토큰 커버리지(복합 셀 대응)."""
    c = norm_table(cell)
    if not c:
        return False
    if c in body:
        return True
    # LLM이 여러 셀을 합성한 값("RLCA -40°C 평균 흡수에너지 27J 이상",
    # "7.50, 19.63, 16.13" — 쉼표 결합 실측) — 구분자 분절 토큰의 80% 이상이
    # 원문에 있으면 실재로 인정(날조는 미달)
    toks = [norm_table(t) for t in re.split(r"[\s|/,;·]+", cell)
            if len(t.strip(" ,;·")) >= 2]
    if len(toks) < 2:
        return False
    hit = sum(1 for t in toks if t and t in body)
    return hit / len(toks) >= 0.8


def requote_table(quote: str, article: str) -> str:
    """축자 실패·정규화 통과 quote → 원문 표의 실제 행 텍스트로 교체(재인용).

    표 인용은 값이 전부 실재해도 <td> 마크업 때문에 부분문자열 검증이 깨진다.
    quote 토큰의 80% 이상을 담는 원문 조각(HTML 행 우선, 최단)을 그대로 반환해
    축자 보증(quote_in)을 회복한다 — 값 검증(_cell_in)이 끝난 뒤에만 호출할 것."""
    toks = [norm_table(t) for t in re.split(r"[\s|/,;·]+", quote)
            if len(t.strip(" ,;·")) >= 2]
    if not toks:
        return ""
    cands = re.findall(r"<tr>.*?</tr>", article, re.S) \
        + [ln for ln in article.splitlines() if ln.strip()]
    best, best_cov = "", 0.0
    for cand in cands:
        b = norm_table(cand)
        cov = sum(1 for t in toks if t and t in b) / len(toks)
        if cov > best_cov or (cov == best_cov and best and len(cand) < len(best)):
            best, best_cov = cand, cov
    return best if best_cov >= 0.8 else ""


def gen_table_lookup(rule: dict, aux: dict, feedback: str = "") -> dict:
    obj = chat_json(prompts.TABLE_LOOKUP_QA.format(
        section_path=sp(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + feedback, max_tokens=2000)
    sel, col = obj.get("row_selector", ""), obj.get("column_selector", "")
    val = obj.get("answer_value", "")
    # 셀 경계를 넘는 인용은 태그 때문에 부분문자열 검증이 깨질 수 있어
    # 행·열 선택자와 기준값을 각각 원문 대조한다(엔티티·LaTeX 정규화).
    body = norm_table(rule["content"])
    ok = (bool(sel) and bool(val) and _cell_in(sel, body) and _cell_in(val, body)
          and (not col or _cell_in(col, body)))
    # 표 quote 재인용 — 값이 실재(_cell_in)하는데 마크업 때문에 축자 실패면
    # 원문 행 텍스트로 교체해 축자 보증 회복(검수행 방지)
    quote, requoted = obj.get("evidence_quote", ""), False
    if quote and not quote_in(quote, rule["content"]) and _cell_in(quote, body):
        new_q = requote_table(quote, rule["content"])
        if new_q:
            quote, requoted = new_q, True
    row = base_row(rule, "table_lookup", "TABLE_LOOKUP_QA")
    row.update({
        "question": anchor_question(obj.get("question", ""), rule, always=True),
        "gold_answer": obj.get("answer", ""),
        "evidence": [evidence_entry(rule, quote)],
        "needs_review": not ok,
        "metadata": {**row["metadata"], "n_table_rows": aux.get("n_rows"),
                     "requoted": requoted,
                     # 표 프로버넌스 — 검수·평가에서 셀 좌표 재현용
                     "table": {"row_key": sel, "column_key": col,
                               "cell_value": val,
                               "header_cells": aux.get("header_cells")}}})
    return row


# ── hierarchy ───────────────────────────────────────────────────────────
def gen_hierarchy(rule: dict, aux: dict, feedback: str = "") -> dict:
    obj, sus = chat_scenario(
        prompts.HIERARCHY_QA.format(
            leaf_path=aux["leaf_path"], leaf_text=aux["leaf_text"],
            section_path=sp(rule), article_text=rule["content"])
        + risk_block(rule) + refs_block(rule, aux) + doc_scope_block(rule) + feedback, rule["content"],
        lambda o: o.get("scenario", ""), max_tokens=4000,  # premise_check+사고 분량
        essential=lambda o: bool(str(o.get("scenario") or "").strip()
                                 and str(o.get("answer") or "").strip()))
    applies = str(obj.get("applies", "")).lower()
    # 경로 토큰("3항"·"(2)호"·"(가)목")이 answer에 없으면 구조 추론이 빠진 답
    path_ok = all(tok in norm(obj.get("answer", ""))
                  for tok in norm(aux["leaf_path"]).replace("항", "항|").replace(
                      "호", "호|").split("|") if tok)
    q_anchored = anchor_question(obj.get("question", ""), rule, always=True)
    q = (q_anchored if not obj.get("scenario")
         else f"{q_anchored}\n\n[상황]\n{obj['scenario']}")
    row = base_row(rule, "hierarchy", "HIERARCHY_QA")
    row.update({
        "question": q,
        "expected_judgment": {"yes": "applicable", "no": "not_applicable"}.get(applies, ""),
        "gold_answer": obj.get("answer", ""),
        "evidence": [evidence_entry(rule, obj.get("evidence_quote", ""))],
        "needs_review": (applies not in ("yes", "no") or not path_ok or bool(sus)
                         or not quote_in(obj.get("evidence_quote", ""), rule["content"])),
        "metadata": {**row["metadata"], "leaf_path": aux["leaf_path"],
                     "depth": aux["depth"],
                     "premise_check": {k: obj[k] for k in
                                       ("premise_upper", "premise_leaf",
                                        "premise_alignment") if obj.get(k)},
                     "paraphrase_suspects": [list(s) for s in sus]}})
    return row



# ── formula_calc — 수식 대입 계산형 (v2a, unit_convert 대체) ────────────
def gen_formula_calc(rule: dict, aux: dict | None, feedback: str = "") -> dict | None:
    obj = chat_json(prompts.FORMULA_CALC_QA.format(
        rule_card=card_json(rule), article_text=rule["content"])
        + feedback, max_tokens=2400)
    q, ans = obj.get("question"), obj.get("answer")
    if not q or not ans:
        return None
    steps = [str(x) for x in (obj.get("answer_steps") or [])]
    f_latex = (obj.get("formula_latex") or "").strip()
    fv = str(obj.get("final_value") or "").strip()
    judgment = (obj.get("judgment") or "").strip()
    joined = ans + "\n" + "\n".join(steps)
    flags = []
    # 게이트: 수식 원문 실존 / 최종값 답변 포함 / 판정 시 부등호 명시
    if not f_latex or norm(f_latex) not in norm(rule["content"]):
        flags.append("formula_not_in_source")
    if fv and fv.replace(",", "") not in joined.replace(",", ""):
        flags.append("final_value_missing")
    if judgment in ("충족", "미충족") and not re.search(r"[<>≤≥]", joined):
        flags.append("inequality_missing")
    row = base_row(rule, "formula_calc", "FORMULA_CALC_QA")
    row.update({
        "question": anchor_question(q, rule),
        "gold_answer": ans,
        "expected_judgment": {"충족": "compliant",
                              "미충족": "non_compliant"}.get(judgment),
        "evidence": [evidence_entry(rule, obj.get("evidence_quote", ""))],
        "needs_review": bool(flags) or not quote_in(
            obj.get("evidence_quote", ""), rule["content"]),
        "metadata": {**row["metadata"], "given": obj.get("given") or {},
                     "formula_latex": f_latex, "final_value": fv,
                     "final_unit": obj.get("final_unit", ""),
                     "gate_flags": flags}})
    return row


# ── figure_qa — 그림 근거형 (v2a 신설) ──────────────────────────────────
def gen_figure_qa(rule: dict, aux: dict | None, feedback: str = "") -> dict | None:
    obj = chat_json(prompts.FIGURE_QA.format(
        rule_card=card_json(rule), article_text=rule["content"])
        + feedback, max_tokens=1800)
    q, ans = obj.get("question"), obj.get("answer")
    if not q or not ans:
        return None
    fref = (obj.get("figure_ref") or "").strip()
    flags = []
    if not fref or norm(fref)[:20] not in norm(rule["content"]):
        flags.append("figure_ref_not_in_source")
    row = base_row(rule, "figure_qa", "FIGURE_QA")
    row.update({
        "question": anchor_question(q, rule),
        "gold_answer": ans,
        "evidence": [evidence_entry(rule, obj.get("evidence_quote", ""))],
        "needs_review": bool(flags) or not quote_in(
            obj.get("evidence_quote", ""), rule["content"]),
        "metadata": {**row["metadata"], "figure_ref": fref,
                     "gate_flags": flags}})
    return row


GEN = {"spec": gen_spec, "applicability": gen_applicability,
       "crossref": gen_crossref, "def_link": gen_def_link,
       "precedence": gen_precedence, "formula_calc": gen_formula_calc,
       "figure_qa": gen_figure_qa,
       "table_lookup": gen_table_lookup, "hierarchy": gen_hierarchy}


# ── 재생성 루프 — 검증기 REJECT 사유를 피드백으로 재생성 ────────────────
REGEN_FEEDBACK = """

[재작성 지시 — 직전 생성이 독립 검증에서 거절되었습니다]
거절 사유:
{issues}
위 사유를 모두 해결하여 처음부터 다시 작성하십시오. 특히 지적된 용어·전제·
분기 판단을 조항 원문과 다시 대조하십시오. 답은 원문만으로 유일하게
도출되어야 합니다."""


def incomplete(row: dict) -> bool:
    """빈 질문/답변 — 검증 콜을 쓰기 전에 걸러 즉시 재생성한다.

    실측: precedence 생성이 answer를 빈 문자열로 낸 행이 검증까지 흘러가
    콜만 소모하고 REJECT됐다."""
    return not (row.get("question") or "").strip() or not (row.get("gold_answer") or "").strip()


# 답변 속 표·그림·조항 참조 번호 — 원문·질문 어디에도 없으면 외부지식 혼입.
# 실측 REJECT: 답변이 '304.4항'·'표 3.14.4'를 언급했으나 원문에는 표 3.14.3만 존재.
_REF_PAT = re.compile(
    r"표\s*\d+(?:\.\d+)+|그림\s*\d+(?:\.\d+)+|\d{3}\.\s*\d+\s*항|\d{3}\.의\s*\d+\s*항")


# ── 라벨↔답변 결론 정합 (프로그램 확정) — 전문가 검수(2026-07-21) 반영 ──
# 실측: 답변이 "적용 대상이며 … label은 'applicable'이어야 한다. 수정한다"라고
# 결론 내리면서 expected_judgment=not_applicable인 행이 검증(LLM)까지 통과했다.
_CONCL_NEG = re.compile(r"적용되지 않|적용 대상이 아니|적용하지 않|미적용")
_CONCL_POS = re.compile(
    r"적용된다|적용 대상이다|적용 대상이며|적용되며|적용받는다|적용되지만|적용되고")
# 생성 과정 누출 — 라벨 설계·자기 수정 발화가 답변에 새는 경우
_META_LEAK = re.compile(
    r"label|라벨|시나리오를 만들|출제|수정한다|설정하고|이어야 한다\. 수정")

APPLIC_CONCL = {"applicable": _CONCL_POS, "not_applicable": _CONCL_NEG}


def judgment_conclusion_issues(row: dict) -> list[str]:
    """expected_judgment와 답변 텍스트의 결론·메타 발화 정합 — 검증 콜 이전 확정 검사."""
    issues = []
    ans = row.get("gold_answer") or ""
    if _META_LEAK.search(ans):
        issues.append("답변에 라벨 설계·자기 수정 과정(메타 발화)이 노출됨 — 최종 "
                      "사용자 답변만 서술하십시오")
    # 판정형 답변의 시나리오 사실 유보 — 시나리오 오독("있는 사실을 명시되지
    # 않았다"고 서술) 또는 시나리오 전제 누락, 어느 쪽이든 재생성(스모크 실측)
    if row.get("expected_judgment") and re.search(
            r"(?:시나리오|상황|개요)[^.。]{0,40}?명시(?:하지|되지)\s*않", ans):
        issues.append("답변이 시나리오 사실을 '명시되지 않았다'고 유보합니다 — "
                      "시나리오에 있는 사실은 확정 사실로 사용하고, 판정에 필요한 "
                      "사실이 정말 없다면 시나리오에 그 사실을 추가해 확정 결론을 "
                      "내도록 다시 작성하십시오")
    ej = row.get("expected_judgment")
    if ej in APPLIC_CONCL:
        # 부정의 부정("미적용 사유가 아니다"·"적용되지 않는 것이 아니다")은 긍정
        # 설명이지 반대 결론이 아니다 — 검사 전에 제거(전문가 검수 9차 오탐 대응)
        scan = re.sub(r"(?:미적용|적용되지 않는 것|적용 배제)(?:이|의)?"
                      r"(?:\s*사유)?(?:가|이|은|는)?\s*아니", "", ans)
        want, other = APPLIC_CONCL[ej], APPLIC_CONCL[
            "not_applicable" if ej == "applicable" else "applicable"]
        # 반대 결론이 존재하고 기대 결론이 없으면 확정 모순; 둘 다 있으면(부분 적용
        # 서술 — "1항은 적용되지만 2항은 미적용") 조항 전체 라벨과 충돌 → 재작성
        if other.search(scan) and not want.search(scan):
            issues.append(f"답변 결론이 기대 라벨({ej})과 반대입니다 — 라벨에 맞는 "
                          "시나리오·답변으로 다시 작성하십시오")
        elif other.search(scan) and want.search(scan):
            issues.append("답변이 적용/미적용 결론을 모두 포함합니다 — 조항 일부 항만 "
                          "비적용이면 질문을 그 항으로 한정해 결론을 하나로 만드십시오")
    return issues


# 자리표시 표준번호(ISO 0000 류) — 프롬프트 금지에도 재발 + 검증기 통과 실측(검수
# 9차). 문제는 번호 자체가 아니라 '원문 밖 창작': evidence에 실존하면 원문 충실성
# 차원에서 허용하고, QA에만 생긴 경우만 거절한다(QA 표준번호 ⊆ evidence 표준번호).
_PLACEHOLDER_STD = re.compile(  # 뒤 \b는 한글 조사("0000을")에서 미성립 → (?!\d)
    r"\b(?:ISO|IEC|KS|JIS|ASME|EN|API)[\s\-–—:]*0{3,5}(?!\d)", re.IGNORECASE)


def _normalize_standard(value: str) -> str:
    """ISO-0000, ISO 0000, ISO:0000을 동일하게 비교."""
    return re.sub(r"[\s\-–—:]+", " ", value.upper()).strip()


def ungrounded_placeholder_standards(row: dict) -> list[str]:
    """QA에서 창작된(evidence 원문에 없는) 자리표시 표준번호만 반환."""
    qa_text = " ".join([row.get("question") or "", row.get("gold_answer") or ""])
    evidence_text = " ".join(
        " ".join([ev.get("quote") or "", ev.get("article_text") or ""])
        for ev in row.get("evidence") or [])
    qa_stds = {_normalize_standard(m.group(0))
               for m in _PLACEHOLDER_STD.finditer(qa_text)}
    ev_stds = {_normalize_standard(m.group(0))
               for m in _PLACEHOLDER_STD.finditer(evidence_text)}
    return sorted(qa_stds - ev_stds)


# 답변 속 라틴 기호(Zs·Kd·C2류) — 원문·질문에 없으면 외부 지식/창작. 특히 수식
# 추출 유실로 원문 기호가 사라진 조항에서 모델이 기호를 '복원'해 쓰는 사례 실측
# (11편 A4.2.2: 원문 "…는 면재측…"(기호 유실)인데 답변이 Zs·Zw 사용 + 분기 오독).
# \b는 한글 인접("Zs가")에서 미성립 — ASCII 전후 부정 룩어라운드 사용
_SYM_ANS = re.compile(r"(?<![A-Za-z0-9])[A-Z](?:_?\{?[a-z0-9]{1,2}\}?)(?![a-z0-9])")
_SYM_WHITELIST = {  # 단위·원소·상용 약어 — 기호 창작이 아님
    "Hz", "Pa", "Nm", "Nmm", "No", "Cu", "Ni", "Cr", "Al", "Mg", "Ti", "Zn",
    "Fe", "Mn", "Si", "Mo", "Co", "Pb", "Sn"}


def _sym_norm(text: str) -> str:
    return re.sub(r"[${}_]", "", text or "")


def foreign_symbols(row: dict) -> list[str]:
    """질문·답변이 쓴 기호 중 근거 원문에 없는 것(표기 정규화 후 대조).

    접지원은 evidence 원문뿐이다 — 질문도 모델 산출물이라 질문을 접지로 삼으면
    질문·답변이 함께 창작한 기호가 통과한다(A4.2.2 Zs 순환 접지 실측)."""
    src = " ".join(
        _sym_norm(str(e.get("article_text", "")) + " " + str(e.get("section_path", "")))
        for e in row.get("evidence") or [])  # 제목 표기(IACS UR Z17 등)도 접지
    qa = _sym_norm(f"{row.get('question') or ''} {row.get('gold_answer') or ''}")
    out = set()
    for m in _SYM_ANS.finditer(qa):
        tok = m.group(0)
        if tok in _SYM_WHITELIST or len(tok) < 2:
            continue
        if tok not in src:
            out.add(tok)
    return sorted(out)


def xref_unexpanded(row: dict) -> bool:
    """crossref 답변이 참조 조번호만 반복하고 내용을 전개하지 않았는지.

    실측(40건 배치 10편 A103): 답 "3편 7장 101. 3항의 요건을 만족하여야 한다" —
    본 조항 단서의 되풀이라 참조 조항 없이도 쓸 수 있는 답(기준 6을 검증기가
    미검출). 판정: 답변에 참조 번호 언급이 있는데, 참조 조항 **고유** 실사
    (본 조항에 없는 한글 내용어)와의 겹침 < 3 이고 고유 수치 공유도 없으면
    미전개로 본다 — 계산형 답(수치 결합)은 수치 공유로 통과."""
    evs = row.get("evidence") or []
    if row.get("track") != "crossref" or len(evs) < 2:
        return False
    ans = row.get("gold_answer") or ""
    if not re.search(r"\d+\s*[편장]|\d{3,4}\.", ans):
        return False  # 참조 번호 언급 없음 — 내용 서술형 답
    words = lambda t: set(re.findall(r"[가-힣]{2,}", t or ""))  # noqa: E731
    nums = lambda t: {m.group(0) for m in re.finditer(  # noqa: E731
        r"\d[\d,]*(?:\.\d+)?", t or "") if len(m.group(0)) >= 2}
    ref_only_w = words(evs[1].get("article_text")) - words(evs[0].get("article_text"))
    ref_only_n = nums(evs[1].get("article_text")) - nums(evs[0].get("article_text"))
    return (len(words(ans) & ref_only_w) < 3
            and not (nums(ans) & ref_only_n))


# 정규 인용형 "NNN.의 N항"/"NNN. N항" — 문자 그대로 원문에 없어도, 해당 조가
# evidence에 실재하고 그 조 원문에 해당 항 마커("N.")가 있으면 접지로 인정.
# (원문 표기가 "101. 3항"인데 답이 표준형 "101.의 3항"을 써서 6시도 낭비 실측)
_REF_COMPOSED = re.compile(r"(\d{3,4})\.\s*의?\s*(\d+)\s*항")


def _composed_grounded(ref_text: str, row: dict) -> bool:
    m = _REF_COMPOSED.search(ref_text)
    if not m:
        return False
    art_no, para = m.group(1), m.group(2)
    for e in row.get("evidence") or []:
        sp_str = str(e.get("section_path") or "")
        if f"{art_no}." in sp_str and re.search(
                rf"(?m)^\s*{para}\.\s", e.get("article_text") or ""):
            return True
    return False


def foreign_refs(row: dict) -> list[str]:
    """gold_answer의 참조 번호 중 질문·근거 원문에 그대로 없는 것.

    조번호+항번호 조합은 원문 문자 일치 또는 구성 요소 실재(_composed_grounded)
    중 하나면 통과 — 검증 불가한 합성('104.5'류 날조)만 지적한다."""
    src = norm(row.get("question", "")) + "".join(
        norm(e.get("article_text", "")) for e in row.get("evidence") or [])
    out = []
    for m in _REF_PAT.finditer(row.get("gold_answer", "") or ""):
        if norm(m.group(0)) in src:
            continue
        if _composed_grounded(m.group(0), row):
            continue
        out.append(m.group(0).strip())
    return sorted(set(out))


def _subject_aliases(subj: str, source_file: str = "",
                     scope_text: str = "") -> set[str]:
    """대상어 별칭 — 하드코딩 사전 대신 전부 코퍼스·프로그램 도출(검수 11차 후속).

    ① 파일명 괄호 약어: '해상작업지원선(OSV) 지침' → OSV
    ② 적용범위 조항의 자기 지칭 선언: '(이하 구조물이라 한다)' → 구조물
    ③ 대상어 접미 부분어(3자+): 해상작업지원선 → 작업지원선·지원선
    ④ 문서 제목의 주제 토큰(불용어 제외): '선박의 방사 소음 지침' → 선박·방사·소음
      — 첫 토큰만으로는 시나리오가 다른 주제어로 서술될 때 오탐(v2 배치 실측)"""
    out = {subj}
    out |= {subj[i:] for i in range(1, max(1, len(subj) - 2))}
    out |= set(re.findall(r"\(([A-Za-z]{2,6})\)", source_file or ""))
    out |= {a.strip() for a in re.findall(
        r"이하\s*([가-힣A-Za-z][가-힣A-Za-z ]{1,14}?)\s*(?:이라|라)\s*한다",
        scope_text or "")}
    stop = {"지침", "규칙", "규정", "통합본", "개정", "별표", "부록", "적용",
            "및", "등", "등에", "관한", "대한", "위한", "기타"}
    for tok in re.split(r"[\s_()]+", source_file or ""):
        tok = re.sub(r"\d+", "", tok)
        if len(tok) < 2 or tok in stop or not _HANGUL_RE.search(tok):
            continue
        out.add(tok)
        if tok.endswith("의"):  # 조사 제거: '선박의' → '선박'
            out.add(tok[:-1])
        out |= {tok[i:] for i in range(1, len(tok) - 2)}  # 접미 부분어(3자+)
    out -= stop
    # PDF 추출 공백 오염("구 조물") 대비 — 별칭은 공백 제거형으로 정규화
    return {re.sub(r"\s+", "", a) for a in out if len(re.sub(r"\s+", "", a)) >= 2}


def _static_reasons(row: dict) -> list[str]:
    """정적 게이트가 남긴 흔적 → 검수 사유 코드."""
    md = row.get("metadata") or {}
    reasons = []
    if md.get("paraphrase_suspects"):
        reasons.append("paraphrase_suspect")
    if row.get("track") == "precedence" and not (md.get("branch_check") or {}).get("matched"):
        reasons.append("branch_check_missing")
    # 선판정 구조 미작성 — 계획 없이 쓴 답은 완결성 실패가 잦다(스모크 실측)
    if row.get("track") in ("applicability", "hierarchy") and not md.get("premise_check"):
        reasons.append("premise_check_missing")
    # 전 트랙 문서범위 게이트(검수 10차→11차 수정) — 특수 문서의 시나리오
    # 판정형 문제는 **시나리오 텍스트**에 그 대상어가 있어야 한다.
    # 주의: 앵커([대상 조항] 문서명)가 질문에 자동 포함되므로 질문 전체 검사는
    # 자기 무력화된다(검수 11차) — 앵커 줄을 제거한 텍스트로 검사.
    # 인용 스팬 해소 불가(검수 11차) — 추적 불가 정보를 저장만 하고 통과 금지
    for e in row.get("evidence") or []:
        if e.get("quote") and e.get("char_start", 0) < 0:
            reasons.append("evidence_span_unresolved")
            break
    subj = md.get("doc_subject") or ""
    q_text = row.get("question") or ""
    scope_tracks = ("applicability", "hierarchy", "precedence",
                    "unit_convert", "def_link")
    check = (row.get("track") in scope_tracks
             or (row.get("track") == "crossref" and "[상황]" in q_text))
    if subj and check:
        # 앵커 줄 제거 + 공백 무시 대조(시나리오 쪽 공백 오염도 흡수)
        scen = re.sub(r"\s+", "",
                      re.sub(r"\[대상 조항\][^\n]*", "", q_text))
        aliases = set(md.get("doc_subject_aliases") or []) | {subj}
        if not any(re.sub(r"\s+", "", a) in scen for a in aliases):
            reasons.append("doc_scope_mismatch")
    if md.get("discretionary_mismatch"):
        reasons.append("discretionary_override")  # 재량 규정을 확정 배제로 결론
    # 답변의 판정이 그림에 의존 — 그림은 evidence에 실릴 수 없어 검증 불가
    # (전문가 검수 6차: 그림 3·4 없이 모델링 결과 확정한 행이 ACCEPT됨)
    if (row.get("track") != "figure_qa"
            and re.search(r"그림\s*\d", row.get("gold_answer") or "")
            and "[인용된 그림]" not in ((row.get("evidence") or [{}])[0]
                                        .get("article_text") or "")):
        reasons.append("figure_dependent")
    # 재량 완화 단서 조항의 미충족 **단정** — 단서 성립 여부에 따라 결론이 갈려
    # 유일 판정 불가. 답변이 단서를 명시(조건부 서술)하면 정상, 단정하면 검수행.
    if (row.get("track") == "unit_convert"
            and row.get("expected_judgment") == "non_compliant"
            and _RELIEF.search((row.get("evidence") or [{}])[0].get("quote") or "")
            and not _RELIEF.search(row.get("gold_answer") or "")):
        reasons.append("relief_clause_uncertain")
    if row.get("needs_review") and not reasons:
        reasons.append("static_gate_flag")  # 인용 불일치·경로 토큰 등
    return reasons


def _finalize(row: dict, status: str, reasons: list[str]) -> None:
    """status 단일화(ACCEPT|REVIEW|REJECT|ERROR) + trainable 확정.

    needs_review는 하위 호환용으로 유지(= not trainable). ACCEPT만 학습 투입,
    REVIEW/REJECT/ERROR는 전부 검수 큐(폐기하지 않음)."""
    row["status"] = status
    row["trainable"] = status == "ACCEPT"
    row["review_reasons"] = reasons
    row["needs_review"] = not row["trainable"]
    # 품질 등급(외부 검토 반영): LLM 검수만 통과 = silver, 전문가 확인 = gold
    row["quality"] = ("gold" if (row.get("metadata") or {}).get("expert_reviewed")
                      else "silver")


# 근거 자체가 부족한 거절 — 재생성해도 고칠 수 없다(소스 문제).
# 맨 "누락"·"무관"은 답변 결함 서술("계산이 누락됨"·"무관한 수치")에도 매칭돼
# 재생성 가능 건을 조기 REVIEW로 유실시켰다(40건 배치 def_link 실측) — 소스
# 한계 전용 표현으로 한정.
_EVIDENCE_HINTS = ("원문에 없", "원문에 근거", "근거 부재", "부재", "유실",
                   "도출될 수 없", "도출되지 않", "외부 지식", "외부지식",
                   "외부 추론", "원문과 무관", "원문만으로는", "판단할 수 없",
                   "확인할 수 없")
# 답변 결함 표지 — 모델이 저지른 일(날조·치환·미수행)은 재생성으로 교정 가능.
# 소스 힌트와 공존하면 GENERATION 우선(예: "원문에 없는 수치를 사용함" = 날조).
_GEN_MARKERS = ("사용함", "사용했", "포함함", "포함됨", "포함되어", "제시함",
                "서술함", "창작", "날조", "추가함", "치환", "일치하지 않")


def _issue_class(issues: list[str]) -> str:
    text = " ".join(issues or [])
    ev = any(h in text for h in _EVIDENCE_HINTS)
    gen = any(m in text for m in _GEN_MARKERS)
    return "EVIDENCE" if ev and not gen else "GENERATION"


# 답변 국소 수리 대상 — 프로그램이 답을 조립하는 트랙(unit·table)은 형식이
# 깨지므로 제외. 사유가 전부 답변측 키워드일 때만 시도한다.
_REPAIRABLE_TRACKS = ("spec", "applicability", "crossref", "def_link",
                      "precedence", "hierarchy")
_ANSWER_SIDE = ("답변", "결론", "answer", "완전")


def _repair_answer(row: dict, issues: list[str]) -> str:
    """질문·시나리오는 보존하고 답변만 교정 — 전면 재생성의 새 결함 위험 회피."""
    evs = (row.get("evidence") or [])[:2]
    articles = "\n\n".join(
        f"[원문 {i}] {e.get('section_path', '')}\n{e.get('article_text', '')}"
        for i, e in enumerate(evs, 1))
    jn = (f"\n- 기대 판정: {row['expected_judgment']} — 결론은 이 판정과 일치해야 "
          "합니다." if row.get("expected_judgment") else "")
    try:
        obj = chat_json(prompts.ANSWER_REPAIR.format(
            articles=articles, question=row.get("question", ""),
            answer=row.get("gold_answer", ""), judgment_note=jn,
            issues="\n".join(f"- {i}" for i in issues[:3])), max_tokens=1400)
    except Exception:  # noqa: BLE001
        return ""
    return (obj.get("answer") or "").strip()


_ISSUE_QUOTE = re.compile(r"[‘'「\"]([^’'」\"]{6,60})[’'」\"]")
# 검증기가 "존재하지 않는 조항 번호"로 지목하는 합성 표기 "104.5" (조 104의 5항)
_ISSUE_ART_NO = re.compile(r"(\d{3,4})\.(\d{1,2})(?![.\d])")


def _claim_falsified(issues: list[str], row: dict) -> str | None:
    """검증기의 "원문에 없다" 주장을 프로그램으로 반증 — 반증 근거 문자열 반환.

    실측 오탐 2종: ① '제2급 압력용기만 생산하는 경우'가 원문 1항에 실재하는데
    '원문에 없는 전제'로 거절 ② '104.5'(=104.의 5항 합성 표기)를 '존재하지 않는
    조항 번호'로 거절 — 조 104가 근거 조이고 그 원문에 5항이 실재함."""
    src = norm(row.get("question", "")) + "".join(
        norm(e.get("article_text", "")) for e in row.get("evidence") or [])
    evs = row.get("evidence") or []
    for issue in issues or []:
        if not any(h in issue for h in _EVIDENCE_HINTS) and "존재하지 않" not in issue:
            continue
        for m in _ISSUE_QUOTE.finditer(issue):
            if norm(m.group(1)) in src:
                return m.group(1)
        if "조항 번호" in issue or "존재하지 않" in issue:
            for m in _ISSUE_ART_NO.finditer(issue):
                art, hang = m.group(1), m.group(2)
                for e in evs:
                    cid = e.get("chunk_id", "")
                    text = e.get("article_text", "")
                    if (cid.endswith(f"A{art}") or f"_A{art}-" in cid
                            or f"{art}." in text[:200]):
                        if re.search(rf"(?:^|\n)\s*{hang}\.\s", text):
                            return f"{art}.의 {hang}항 실재"
    return None


# 원문이 요건 판정을 그림에 위임("그림 N에 따라") — 그림은 evidence에 실을 수
# 없어 어떤 생성도 검증을 통과할 수 없다. 판정형 트랙은 생성 전에 건너뛰어
# 콜 낭비와 필연 REJECT를 예방한다(1차 수율 개선 — 검수 9차 #8 실측 유형).
_FIGURE_BOUND = re.compile(
    r"그림\s*\d[\d.]*(?:\s*및\s*그림\s*\d[\d.]*)?\s*에\s*(?:따라|의하여|나타낸)")
_JUDGMENT_TRACKS = {"applicability", "hierarchy", "precedence", "formula_calc"}


def gen_verified(track: str, rule: dict, aux: dict | None,
                 retries: int = 2) -> tuple[dict | None, dict | None]:
    """생성→검증→REJECT면 유형별 분기 재생성(최대 retries회).

    - GENERATION류(용어·전제·분기 오류): 사유를 피드백으로 재생성 — 교정 가능
    - EVIDENCE류(원문에 없음·도출 불가): 재생성으로 못 고침 — 즉시 REVIEW
      (source_insufficient)로 종료해 콜 낭비를 막는다
    끝까지 REJECT면 폐기하지 않고 검수 큐(REJECT)로 남긴다.
    """
    from .verify_suite import verify_row  # 지연 임포트(순환 방지)
    if (track in _JUDGMENT_TRACKS and _FIGURE_BOUND.search(rule.get("content", ""))
            and "[인용된 그림]" not in rule.get("content", "")):
        log(f"  [{track}] 그림 위임 조항 — 생성 전 스킵: "
            f"{rule.get('chunk_id', '')[-40:]}")
        return None, None
    feedback = ""
    row = verdict = None
    static_code = ""
    repaired = False  # 답변 국소 수리는 런당 1회
    for attempt in range(1, retries + 2):
        row = GEN[track](rule, aux, feedback)
        if row is None:
            return None, None
        row["metadata"]["verify_attempts"] = attempt
        if incomplete(row):  # 빈 질문/답변 — 검증 없이 즉시 재생성
            verdict, static_code = None, "incomplete_output"
            feedback = REGEN_FEEDBACK.format(
                issues="- 질문 또는 답변이 비어 있음 — 출력 JSON의 모든 필드를 "
                       "빠짐없이 채워 완전한 문제를 작성하십시오.")
            continue
        refs = foreign_refs(row)
        if refs:  # 원문에 없는 참조 번호 — 검증 콜 없이 즉시 재생성
            verdict = None
            static_code = "foreign_reference:" + ",".join(refs)[:100]
            feedback = REGEN_FEEDBACK.format(
                issues=f"- 답변에 원문·질문 어디에도 없는 참조 번호({', '.join(refs)})가 "
                       "포함됨 — 원문에 문자 그대로 존재하는 표·그림·조항 번호만 "
                       "인용하거나, 참조 번호 없이 내용을 서술하십시오. 항은 원문 "
                       "표기 그대로(예: '401.의 1항') 쓰고 '401.1항'처럼 합성하지 "
                       "마십시오.")
            continue
        syms = foreign_symbols(row)
        if syms:  # 원문에 없는 기호 사용 — 검증 콜 없이 즉시 재생성
            verdict = None
            static_code = "foreign_symbol:" + ",".join(syms)[:80]
            feedback = REGEN_FEEDBACK.format(
                issues=f"- 답변에 원문·질문 어디에도 없는 기호({', '.join(syms)})가 "
                       "사용됨 — 원문에서 수식 기호가 유실된 조항이면 기호를 "
                       "창작·복원하지 말고 원문의 한국어 명칭(예: '면재측의 "
                       "순단면계수')으로 서술하십시오.")
            continue
        concl = judgment_conclusion_issues(row)
        if concl:  # 라벨↔결론 모순·메타 발화 — 프로그램 확정, 즉시 재생성
            verdict = None
            static_code = "label_conclusion:" + concl[0][:100]
            feedback = REGEN_FEEDBACK.format(
                issues="\n".join(f"- {i}" for i in concl))
            continue
        if xref_unexpanded(row):  # 참조 내용 미전개 — 즉시 재생성
            verdict = None
            static_code = "xref_unexpanded"
            feedback = REGEN_FEEDBACK.format(
                issues="- 답변이 참조 조항의 번호만 반복하고 그 내용을 전개하지 "
                       "않음 — 참조 조항이 실제로 규정하는 요건·조건·효과(예: "
                       "'선급의 승인을 받아 …할 수 있다')를 답변에 풀어 쓰십시오. "
                       "본 조항의 지시 문구 되풀이는 무효입니다.")
            continue
        stds = ungrounded_placeholder_standards(row)
        if stds:  # 원문에 없는 자리표시 표준번호 창작 — 즉시 재생성
            verdict = None
            static_code = "ungrounded_standard_reference:" + ",".join(stds)[:70]
            feedback = REGEN_FEEDBACK.format(
                issues=f"- 근거 원문에 없는 자리표시 표준번호({', '.join(stds)})를 "
                       "창작함 — 원문에 실제로 인용된 표준 번호만 사용하거나, "
                       "'선급기술규칙에서 인용하는 공인 국제·국가 기준'처럼 "
                       "서술하십시오.")
            continue
        verdict = verify_row(row)
        if verdict["verdict"] == "ACCEPT":
            # 검증 통과했더라도 정적 게이트 플래그가 있으면 REVIEW(검수 후 투입)
            reasons = _static_reasons(row)
            _finalize(row, "REVIEW" if reasons else "ACCEPT", reasons)
            return row, verdict
        if _issue_class(verdict["issues"]) == "EVIDENCE":
            # 반증 검사: "원문에 없다"고 인용한 문구가 실재하면 판정 불신 → 1회 재검증
            false_claim = _claim_falsified(verdict["issues"], row)
            if false_claim:
                retry = verify_row(row)
                if retry["verdict"] == "ACCEPT":
                    row["metadata"]["verifier_claim_falsified"] = false_claim[:60]
                    reasons = _static_reasons(row)
                    _finalize(row, "REVIEW" if reasons else "ACCEPT", reasons)
                    return row, retry
                verdict = retry  # 재검증도 거절 — 그 사유로 진행
            _finalize(row, "REVIEW", ["source_insufficient:"
                                      + (verdict["issues"] or [""])[0][:120]])
            row["metadata"]["final_verdict"] = "REVIEW"
            return row, verdict
        issues_list = verdict["issues"] or []
        # 답변 국소 수리 — 사유가 전부 답변측이면 질문·시나리오를 보존한 채
        # 답변만 교정해 재검증(전면 재생성의 '새 시나리오 → 새 결함' 진동 회피)
        if (not repaired and issues_list and track in _REPAIRABLE_TRACKS
                and all(any(k in i for k in _ANSWER_SIDE) for i in issues_list)):
            repaired = True
            fixed = _repair_answer(row, issues_list)
            if fixed:
                cand = {**row, "gold_answer": fixed}
                if not (foreign_refs(cand) or judgment_conclusion_issues(cand)
                        or ungrounded_placeholder_standards(cand)):
                    row["gold_answer"] = fixed
                    row["metadata"]["answer_repaired"] = True
                    retryv = verify_row(row)
                    if retryv["verdict"] == "ACCEPT":
                        reasons = _static_reasons(row)
                        _finalize(row, "REVIEW" if reasons else "ACCEPT", reasons)
                        return row, retryv
                    verdict = retryv
                    issues_list = verdict["issues"] or issues_list
        issues = "\n".join(f"- {i}" for i in issues_list[:3])
        feedback = REGEN_FEEDBACK.format(issues=issues or "- (사유 미기재 거절)")
    if verdict is None:  # 재시도 소진까지 정적 게이트 통과 실패
        # incomplete → ERROR(생성기/서버 이상), 그 외 정적 사유 → REJECT(검수 큐)
        if static_code.startswith(("foreign_reference", "foreign_symbol",
                                   "label_conclusion", "xref_unexpanded",
                                   "ungrounded_standard_reference")):
            _finalize(row, "REJECT", [static_code])
        else:
            _finalize(row, "ERROR", [static_code or "incomplete_output"])
    else:
        _finalize(row, "REJECT",
                  [f"verifier:{(verdict['issues'] or ['사유 미기재'])[0][:120]}"])
    row["metadata"]["final_verdict"] = row["status"]
    return row, verdict


# ── main ────────────────────────────────────────────────────────────────
def route_stats(routed: list[dict]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for it in routed:
        stats[it["track"]] = stats.get(it["track"], 0) + 1
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publisher", default="KR")
    ap.add_argument("--doc-glob", default="*_chunks.jsonl")
    ap.add_argument("--min-tokens", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="조 수 제한(0=전체)")
    ap.add_argument("--n-compare", type=int, default=0,
                    help="선급 비교 문항 수(0=생략, cluster_map 필요)")
    ap.add_argument("--route-only", action="store_true",
                    help="배정 분포만 산출(LLM 호출 없음) → suite_routing.jsonl")
    ap.add_argument("--verify-loop", action="store_true",
                    help="생성 직후 검증 결합 — REJECT 사유를 피드백으로 재생성"
                         "(suite_verify 스테이지 선반영, verdict도 함께 기록)")
    ap.add_argument("--retries", type=int, default=2,
                    help="--verify-loop: REJECT 시 재생성 횟수(기본 2)")
    args = ap.parse_args(argv)

    routed = route_corpus(args.publisher, args.min_tokens, args.doc_glob)
    log(f"대상 조 {len(routed)}건 | 배정: " + json.dumps(
        route_stats(routed), ensure_ascii=False))

    if args.route_only:
        ROUTE_OUT.unlink(missing_ok=True)
        for it in routed:
            r, aux = it["rule"], dict(it["aux"] or {})
            for k in ("def_text", "counterpart_text", "ref_text"):  # 대용량 원문은 요약 기록
                if aux.get(k):
                    aux[k] = aux[k][:80] + "…"
            append_jsonl(ROUTE_OUT, {
                "id": rule_uid(r), "chunk_id": r["chunk_id"],
                "source_file": r["_source_file"], "article_no": r.get("article_no"),
                "article_title": r.get("article_title"), "track": it["track"],
                "features": sorted(r["_feats"]), "aux": aux})
        log(f"라우팅 기록 → {ROUTE_OUT}")
        return

    if args.limit:
        routed = routed[:args.limit]
    CARDS.update({r["chunk_id"]: r.get("card") or {}
                  for r in load_jsonl(OUT_DIR / "rule_cards.jsonl")})
    # resume: 트랙별 파일 전체에서 ::suite id 수집(트랙 재배정에도 중복 생성 방지)
    done: set[str] = set()
    for fname, _ in TRACK_FILES.values():
        done |= {q for q in load_done(OUT_DIR / fname, key="question_id")
                 if q.endswith("::suite")}

    def process(it: dict) -> None:
        rule, track, aux = it["rule"], it["track"], it["aux"]
        rid = rule_uid(rule)
        if f"{rid}::suite" in done:
            return
        try:
            if args.verify_loop:
                row, verdict = gen_verified(track, rule, aux, retries=args.retries)
            else:
                row, verdict = GEN[track](rule, aux), None
                if row is not None and incomplete(row):
                    log(f"-- {rid} [{track}] 빈 질문/답변 — 폐기(다음 실행에서 재시도)")
                    return
            if row is None:
                log(f"-- {rid} [{track}] 생성 불가(계획 폐기)")
                return
            append_jsonl(OUT_DIR / TRACK_FILES[track][0], row)
            if verdict is not None:
                append_jsonl(OUT_DIR / "suite_verdicts.jsonl", verdict)
            tag = ""
            if verdict is not None:
                tag = (f" [{verdict['verdict']}"
                       f"/{row['metadata'].get('verify_attempts', 1)}회]")
            log(f"{rid} [{track}] 완료{tag}"
                + (" (needs_review)" if row["needs_review"] else ""))
        except Exception as e:  # noqa: BLE001
            log(f"!! {rid} [{track}] 실패: {e}")

    pmap(process, routed, workers=int(os.environ.get("AIREG_WORKERS", "1")))

    if args.n_compare:
        # 선급 비교(쌍 단위 — 조당 1건 라우팅과 별개 트랙): 기존 구현 재사용
        from .build_crossref_qa import gen_compare, load_all_parents
        try:
            gen_compare(load_all_parents(), args.n_compare,
                        load_done(OUT_DIR / "compare_qa.jsonl", key="question_id"))
        except Exception as e:  # noqa: BLE001
            log(f"!! compare 실패: {e}")

    total = review = 0
    for fname, _ in TRACK_FILES.values():
        rows = [r for r in load_jsonl(OUT_DIR / fname)
                if str(r.get("question_id", "")).endswith("::suite")]
        total += len(rows)
        review += sum(1 for r in rows if r.get("needs_review"))
    log(f"스위트 QA(조당 1건) {total}건 (needs_review {review})")


if __name__ == "__main__":
    main()
