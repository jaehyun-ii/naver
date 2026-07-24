"""조 특성 트랙 라우터 — 조(parent)를 정적 분석해 조당 1개 QA 트랙을 배정.

학습 능력 축을 트랙으로 대응시키고, 조가 가진 특성에 맞는 트랙 하나를 고른다:
  precedence    일반·특별 규정 우선   — "…에도 불구하고"류 우선 적용 마커
  def_link      정의 조항 결합        — 같은 문서 정의 조의 용어를 본문 조가 사용
  unit_convert  계산·단위 환산        — 환산 가능 단위(mm↔m 등)의 수치 기준
  table_lookup  본문·표 결합          — 다행 표 링크(조건으로 행 선택 필요)
  hierarchy     조·항·호·목 구조      — 중첩 열거(상위 전제 스코프)
  crossref      상호참조 해석         — "202.의 1항"류 같은 문서 타 조 참조
  spec | applicability                — 특성 없는 조의 fallback(결정적 교대)

배정 원칙: 특성 트랙은 코퍼스 전역 빈도가 낮은 것부터(희소 우선) — 흔한 특성
(중첩·표·참조)이 희소 특성(우선 규정)을 덮지 않도록 한다. 특성이 하나도 없으면
스펙 QA / 적용성 판단을 해시 교대로 배정해 조당 1건을 보장한다.
LLM을 쓰지 않는다 — 배정은 결정론적이고 재실행 시 동일하다.
"""
from __future__ import annotations

import hashlib
import json
import re

from .common import (CHUNK_DIR, _attach_tables, _is_requirement_parent,
                     glob_docs, nfc)

# 특성 트랙(희소 우선 배정 대상) — 순서는 빈도 동률일 때의 우선순위
TRACKS = ["precedence", "def_link", "unit_convert", "table_lookup",
          "hierarchy", "crossref"]
# 특성 없는 조의 fallback — 해시 패리티로 교대
FALLBACKS = ("spec", "applicability")
ALL_TRACKS = TRACKS + list(FALLBACKS)


def _stable(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


# ── precedence: 우선 적용 마커 ──────────────────────────────────────────
PREC_RE = re.compile(
    r"에도 불구하고|불구하고|우선하여 적용|우선 적용|달리 규정|특별히 규정"
    r"|따로 정하는|적용하지 아니한다|적용하지 않는다")
# 한국어 조문의 타 조항 참조("202.의 1항") — build_crossref_qa.KR_REF와 동일 패턴
KR_REF = re.compile(r"(?<![\d.])(\d{3})\.(?:의\s*(\d+)\s*항)?")


def _resolve_no(no: str, r: dict, by_article_no: dict[str, list[dict]]) -> dict | None:
    """조 번호 → 같은 문서의 조. 같은 장 우선, 장 밖 동번호가 여럿이면 모호 → None.

    문서 하나에 장마다 같은 조 번호(301. 등)가 반복되므로 번호만으로 잇지 않는다
    — 실측: 4장 문맥의 '301.' 참조가 11장 301.(무관한 장)에 연결된 오류."""
    cands = [c for c in by_article_no.get(no, []) if c["chunk_id"] != r["chunk_id"]]
    same_ch = [c for c in cands
               if str(c.get("chapter_no")) == str(r.get("chapter_no"))]
    if len(same_ch) == 1:
        return same_ch[0]
    if not same_ch and len(cands) == 1:
        return cands[0]
    return None


def find_precedence(r: dict, by_article_no: dict[str, list[dict]]) -> dict | None:
    m = PREC_RE.search(r["content"])
    if not m:
        return None
    # 마커가 속한 문장(줄) — 프롬프트 인용 및 검증 앵커
    line = next((l for l in r["content"].split("\n") if m.group(0) in l), "")
    sent = line.strip()[:300]
    counterpart = None
    for ref in KR_REF.finditer(sent):
        cand = _resolve_no(ref.group(1), r, by_article_no)
        if cand is not None:
            counterpart = cand
            break
    return {"marker": m.group(0), "marker_sentence": sent,
            "counterpart_chunk_id": counterpart["chunk_id"] if counterpart else None,
            "counterpart_text": (counterpart["content"][:4000] if counterpart else None),
            "counterpart_path": (" / ".join(counterpart.get("section_path") or [])
                                 if counterpart else None)}


# ── def_link: 정의 조 용어 색인 ─────────────────────────────────────────
DEF_TITLE = re.compile(r"정의|용어")
# "선급등록(classification)이라 함은 …" / "“화물지역”이란 …" 류
TERM_RES = [
    re.compile(r"(?:^|\n)\s*\d{1,2}\.\s*([가-힣A-Za-z0-9·\- ]{2,40}?)\s*\("),
    re.compile(r"([가-힣A-Za-z0-9·\- ]{2,40}?)(?:\([^)]{1,80}\))?\s*(?:이라|라) 함은"),
    re.compile(r"[“\"]([^”\"]{2,40})[”\"](?:이란|란|이라 함은|라 함은)"),
]
TERM_STOP = {"규칙", "지침", "우리 선급", "이 규정", "선박", "경우"}


def extract_terms(def_parents: list[dict]) -> dict[str, dict]:
    """정의 조들에서 용어 → 정의 조 매핑. 긴 용어 우선 매칭을 위해 호출부에서 정렬."""
    terms: dict[str, dict] = {}
    for p in def_parents:
        for pat in TERM_RES:
            for m in pat.finditer(p["content"]):
                # 목록 번호가 용어에 붙는 파싱 오류("1 수밀"→"수밀") 제거 — 실측
                t = re.sub(r"^\d{1,2}[.)]?\s+", "", m.group(1).strip())
                if len(t) >= 2 and t not in TERM_STOP and t not in terms:
                    terms[t] = p
    return terms


# 정의 속 조 참조("402.1에 따른 탱크"·"402.의 1항") — 용어 정의 인접 창에서 탐색
_DEF_REF = re.compile(r"(\d{3,4})\.(?:\d{1,2}|\s*의?\s*\d{1,2}\s*항)?")


def find_def_link(r: dict, terms: dict[str, dict],
                  by_no: dict[str, list[dict]] | None = None) -> dict | None:
    for t in sorted(terms, key=len, reverse=True):
        d = terms[t]
        if d["chunk_id"] != r["chunk_id"] and t in r["content"]:
            aux = {"term": t, "def_chunk_id": d["chunk_id"],
                   "def_text": d["content"][:6000],
                   "def_path": " / ".join(d.get("section_path") or [])}
            # 참조 체인 폐쇄(검수 10차 CNG A411): 정의가 다른 조를 참조하면
            # 그 조를 해소해 3번째 근거로 동봉. 창은 해당 용어의 정의 세그먼트로
            # 한정(이웃 용어 참조 오귀속 방지). by_no는 파일 단위 인덱스이며,
            # 후보 다수면 정의 조와 같은 장 우선으로 좁힌다(검수 11차).
            # **참조가 있는데 해소 실패한 용어는 후보에서 제외**하고 다음 용어로
            # 넘어간다 — 미해소 참조형 정의는 판정을 완결할 수 없다.
            i = d["content"].find(t)
            seg = d["content"][i:i + 400]
            nxt = re.search(r"\n\s*\d{1,2}\.\s|(?<=말한다\.)\s", seg[len(t):])
            win = seg[:len(t) + nxt.start()] if nxt else seg
            m = _DEF_REF.search(win)
            if not m:
                return aux
            cand = [c for c in (by_no or {}).get(m.group(1)) or []
                    if c["chunk_id"] not in (d["chunk_id"], r["chunk_id"])]
            if len(cand) > 1:
                same_ch = [c for c in cand
                           if str(c.get("chapter_no")) == str(d.get("chapter_no"))]
                cand = same_ch if len(same_ch) == 1 else cand
            if len(cand) == 1:
                aux.update({
                    "def_ref_chunk_id": cand[0]["chunk_id"],
                    "def_ref_text": cand[0]["content"][:4000],
                    "def_ref_path": " / ".join(cand[0].get("section_path") or [])})
                return aux
            continue  # 참조 미해소 — 이 용어는 스킵, 다음 용어 시도
    return None


# ── unit_convert: 환산 가능 단위 기준값 ─────────────────────────────────
# condition_evaluator.UNIT_CONVERSIONS 차원 내에서만 파트너를 고른다(임의 변환 금지).
CONVERT_PARTNER = {"mm": "m", "cm": "mm", "m": "mm", "kg": "t", "t": "kg",
                   "kw": "mw", "mw": "kw", "w": "kw",
                   "min": "sec", "sec": "min", "h": "min", "시간": "min"}
UNIT_DISPLAY = {"mm": "mm", "cm": "cm", "m": "m", "kg": "kg", "t": "t",
                "kw": "kW", "mw": "MW", "w": "W", "min": "min", "sec": "sec",
                "h": "h", "시간": "시간"}
OP_MAP = {"이상": "gte", "이하": "lte", "초과": "gt", "미만": "lt"}
UNIT_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(mm|cm|kW|MW|kg|min|sec|시간|t|m|W|h)"
    r"(?![A-Za-z0-9²³/])\s*(이상|이하|초과|미만)")


# 적용 범위 서술("150 m 미만의 선박…")은 요건 기준값이 아니다 — 실측: 선박 길이
# 조건을 단면계수 기준으로 오인해 물리적으로 무의미한 발췌가 생성됨.
_SCOPE_TAIL = re.compile(r"^\s*(?:의|인)?\s*(?:선박|것|구조물|경우)")
# 허용·예외·준용 술어 — 이 절의 수치는 완화 조건이지 적합/부적합 게이트가 아니다.
# 실측 REJECT 2건: "9 mm 이하일 때에는 …하여도 좋으며"(예외 허용)를 요건으로 오인해
# 9.9 mm 적합 사례를 부적합으로 라벨, "150 m 이하 …준용할 수 있다"를 판정 기준으로 사용.
# 관형형 "…할 수 있는 장치"(능력 수식)는 허용이 아니다 — 종결·연결형만 매치.
_PERMISSIVE = re.compile(
    r"할 수(?:도)? 있(?:다|으며|고|음)|하여도 좋|해도 좋|아니할 수 있|않을 수 있"
    r"|준용할 수 있|생략할 수 있|무방하다")
# 수치 직전의 물리량 명칭("판두께가 9 mm") — 발췌 렌더 시 값의 대상을 고정해
# 다른 물리량(단면계수 등)에 수치를 붙이는 오류를 막는다.
# 실측: 조사 앞 토큰을 그대로 쓰면 '상방으로'·'최소한' 류 비물리량이 잡힌다 —
# 물리량 어미 화이트리스트로 정밀도를 우선하고, 실패 시 빈 값(프롬프트 폴백).
_QTY_RE = re.compile(r"([가-힣]{2,10}?)(?:으로|에서|에는|보다|의|은|는|이|가|을|를|도|로)?\s*$")
_QTY_NOUN = re.compile(
    r"(두께|길이|지름|직경|내경|외경|반지름|안지름|폭|너비|높이|깊이|간격|거리|크기"
    r"|용량|출력|압력|온도|시간|중량|무게|하중|속도|면적|치수|스팬|피치|반경"
    r"|파고|수두|틈새|간극)$")
_QTY_NOUN_ANY = re.compile(_QTY_NOUN.pattern[:-2] + ")")  # $ 없는 버전(후방 탐색용)


def _quantity_before(clause: str, num_str: str) -> str:
    pos = clause.find(num_str)
    if pos < 0:
        return ""
    qm = _QTY_RE.search(clause[:pos].rstrip())
    if not qm or not _QTY_NOUN.search(qm.group(1)):
        return ""
    return qm.group(1)


def _clause_around(content: str, start: int, end: int) -> str:
    """수치가 속한 절(문장) — '다.'(한국어 종결)·개행 경계. 소수점('0.0034')은
    경계로 보지 않아야 하므로 마침표 단독이 아니라 '다.'를 문장 끝으로 본다."""
    prev = max(content.rfind("다.", 0, start), content.rfind("\n", 0, start))
    seg_start = prev + 2 if prev != -1 and content[prev] == "다" else prev + 1
    nxt = content.find("다.", end)
    nl = content.find("\n", end)
    cands = [p for p in (nxt + 2 if nxt != -1 else -1, nl) if p != -1]
    return content[seg_start: min(cands) if cands else None]


# ── 전문가 검수 5차(2026-07-21) 반영 게이트 — 라벨은 프로그램 계산이므로
# 라우터가 조건의 성격을 못 가리면 라벨 자체가 틀린다(실측 10건 중 9건 결함).
# 부정 연산자: "X 미만이어서는 아니 된다" = ≥X (매핑 반전 필요)
_NEGATED = re.compile(r"^\s*(?:이어서는|여서는|하여서는)?\s*(?:아니\s*된다|안\s*된다"
                      r"|아니된다|되지\s*않는다|불가하다)")
_OP_INVERT = {"lt": "gte", "lte": "gt", "gt": "lte", "gte": "lt"}
_OP_KR = {"gte": "이상", "gt": "초과", "lte": "이하", "lt": "미만"}
# 분류 기준("1.5 m 이상인 스크린은 선루로 간주한다") — 적합/부적합 게이트 아님
_CLASSIFY = re.compile(r"간주한다|간주된다|으로 본다|로 본다|분류한다")
# 계산 상한·분기값("230 m 를 넘을 때에는 230 m 로 한다", "…일 때 :")
_CAP = re.compile(r"넘을 때에는|넘는 경우에는|넘을 필요는 없|넘는 때에는")
_BRANCH_COLON = re.compile(r"(?:일 때|인 경우)\s*[:：]")
# 면적·단면적 문맥에 맨 길이 단위 — 원문에서 ²(제곱)가 유실된 흔적(차원 불일치)
_AREA_CTX = re.compile(r"면적|단면적")


def find_unit_threshold(r: dict) -> dict | None:
    content = r["content"]
    for m in UNIT_RE.finditer(content):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = m.group(2).lower() if m.group(2) not in ("시간",) else "시간"
        if unit not in CONVERT_PARTNER:
            continue
        if _SCOPE_TAIL.match(content[m.end():m.end() + 12]):
            continue  # 적용 범위(대상 한정) 수치 — 요건 기준값 아님
        clause = _clause_around(content, m.start(), m.end())
        if _PERMISSIVE.search(clause):
            continue  # 허용·예외·준용 절의 수치 — 위반해도 부적합이 아니다
        if _CLASSIFY.search(clause) or _CAP.search(clause) \
                or _BRANCH_COLON.search(clause):
            continue  # 분류 기준·계산 상한·조건 분기값 — 합격 경계가 아니다
        if _AREA_CTX.search(clause) and unit in ("mm", "cm", "m"):
            continue  # 면적 문맥 + 길이 단위 = ² 유실 — 차원 불일치 위험
        if "×" in clause or "x" in clause.replace(m.group(0), ""):
            continue  # 다차원 치수(600 mm × 600 mm) — 단일 값 판정 불가
        # 상보 분기: 같은 기준값이 이하/이상(미만/초과) 양쪽으로 등장하면 분기 구조
        num = re.escape(m.group(1))
        ops_here = set(re.findall(rf"{num}\s*{re.escape(m.group(2))}\s*"
                                  r"(이상|이하|초과|미만)", content))
        if {"이상", "이하"} <= ops_here or {"초과", "미만"} <= ops_here \
                or ({"이하"} & ops_here and {"이상"} & ops_here):
            continue
        qty = _quantity_before(clause, m.group(1))
        if not qty:
            # 수치 뒤 수식형("600 mm 이하의 간격", "2.4 m 이상의 높이") 우선
            fwd = _QTY_NOUN_ANY.search(content[m.end():m.end() + 14])
            qty = fwd.group(1) if fwd else ""
        if not qty:
            # "다만," 절 분리로 물리량 명사가 직전 문장에 있는 경우(실측: FRP 두께)
            hits = _QTY_NOUN_ANY.findall(content[max(0, m.start() - 150):m.start()])
            qty = hits[-1] if hits else ""
        if not qty:
            continue  # 물리량 명칭 미확정 — 발췌가 무관 치수에 값을 붙일 위험(실측)
        start = content.rfind("\n", 0, m.start()) + 1
        end = content.find("\n", m.end())
        sent = content[start:end if end != -1 else None].strip()[:300]
        if "다음 식" in sent or "다음식" in sent:
            continue  # 수식 기반 요건 — 절대 기준값이 없어 판정 사례 성립 불가
        op = OP_MAP[m.group(3)]
        if _NEGATED.match(content[m.end():m.end() + 16]):
            op = _OP_INVERT[op]  # "미만이어서는 아니 된다" → 사실상 '이상' 요건
        return {"threshold": val, "unit": unit, "op": op,
                "op_kr": _OP_KR[op], "conv_unit": CONVERT_PARTNER[unit],
                "rule_sentence": sent, "quantity": qty}
    return None


# ── table_lookup: 다행 표 ───────────────────────────────────────────────
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)


def find_table(r: dict) -> dict | None:
    """다행 표 + 헤더가 복원돼 있어야 행 선택 문제가 성립한다.

    헤더 셀이 비었거나(기호 유실) 열이 1개뿐이면 TABLE_SCHEMA_INCOMPLETE —
    생성하면 LLM이 없는 헤더(Ks·Kr 등)를 지어 붙이므로 배정 자체를 막는다."""
    if "[인용된 표]" not in r["content"]:
        return None
    tbl = r["content"].split("[인용된 표]", 1)[1]
    rows = tbl.split("<tr")
    n_rows = len(rows) - 1
    if n_rows < 3:
        return None
    header_cells = [re.sub(r"<[^>]+>", "", c).strip()
                    for c in _CELL_RE.findall("<tr" + rows[1])]
    if len(header_cells) < 2:
        return None
    # 매트릭스형 표의 첫 칸(모서리)은 정상적으로 비므로 판정에서 제외
    core = header_cells[1:] if len(header_cells) > 2 else header_cells
    empty = sum(1 for c in core if not c or c in ("()", "( )"))
    if not core or empty / len(core) > 0.3:
        return None  # 헤더 미복원(기호 유실) — 문제 생성 금지
    return {"n_rows": n_rows, "header_cells": header_cells[:8]}


# ── hierarchy: 항(1.) > 호((1)) > 목((가)) 중첩 ─────────────────────────
L1 = re.compile(r"^\s{0,3}(\d{1,2})\.\s")
L2 = re.compile(r"^\s{0,3}\((\d{1,2})\)\s?")
L3 = re.compile(r"^\s{0,3}\(([가-힣])\)\s?")


def parse_outline(content: str) -> list[tuple[dict, str]]:
    """목((가)) 레벨 leaf와 그 항·호 조상 경로를 추출."""
    path: dict[int, str] = {}
    leaves: list[tuple[dict, str]] = []
    for line in content.split("\n"):
        if m := L1.match(line):
            path = {1: m.group(1)}
        elif m := L2.match(line):
            path[2] = m.group(1)
            path.pop(3, None)
        elif m := L3.match(line):
            path[3] = m.group(1)
            if 1 in path and 2 in path:
                leaves.append((dict(path), line.strip()[:300]))
    return leaves


def find_hierarchy(r: dict) -> dict | None:
    body = r["content"].split("[인용된 표]", 1)[0]  # 표 셀 열거 오탐 방지
    leaves = parse_outline(body)
    if leaves:
        p, text = leaves[0]
        return {"depth": 3, "leaf_path": f"{p[1]}항 ({p[2]})호 ({p[3]})목",
                "leaf_text": text}
    # 목이 없어도 항·호가 각 2개 이상이면 2단 스코프 문제가 성립.
    # 소속 항은 단일 순회로 추적한다 — 항 등장 시 갱신, 첫 호에서 확정.
    # (과거 `line is l2[0]` identity 비교는 별도 split 결과라 절대 참이 되지
    # 않아 마지막 항 번호가 붙는 오배정을 냈다 — "2항 (1)호" 부재 경로 실측.)
    n_hang = n_ho = 0
    cur, first_ho = "1", None
    for line in body.split("\n"):
        if mm := L1.match(line):
            cur = mm.group(1)
            n_hang += 1
        elif m2 := L2.match(line):
            n_ho += 1
            if first_ho is None:
                first_ho = (cur, m2.group(1), line.strip()[:300])
    if n_hang >= 2 and n_ho >= 2 and first_ho:
        owner, ho, text = first_ho
        return {"depth": 2, "leaf_path": f"{owner}항 ({ho})호", "leaf_text": text}
    return None


# ── crossref: 같은 문서 타 조 참조 ──────────────────────────────────────
_NOUN_RE = re.compile(r"[가-힣]{2,7}")
_REF_STOP = {"경우", "따라", "따른", "규정", "적합", "이상", "이하", "대하여",
             "위하여", "한다", "하여야", "있는", "관련"}
# ": ... 일 때 :"처럼 값이 사라진 매달린 정의 — 인라인 수식 유실 흔적
_DANGLING = re.compile(r"(?:일 때|경우|같다|따라)\s*[:：]\s*$", re.MULTILINE)


# 장 접두 조 참조 "M장 301.(의 N항)" — 편 접두가 붙으면 문서 간 참조(xdoc 처리)
CH_REF = re.compile(
    r"(?:(\d{1,2})\s*편\s*)?(\d{1,2})\s*장\s*(\d{3,4})\.(?:의\s*(\d+)\s*항)?")
# 문서 접두("저인화점연료선박규칙 6장 301."·"지침 1장 801.") — 같은 문서 조인 금지.
# 단독 '지침/규칙/기준'도 타 문서(적용지침·대응 규칙) 참조다 — 실측: "…함은 지침
# 1장 801."이 규칙 1장 801로 오연결됐다. 보수적으로 접미 일치 전부 스킵.
_NAMED_PREFIX = re.compile(r"(?:규칙|기준|지침)\s*$")
# 단독 '지침' 접두(합성어 OO지침 제외) — 같은 편의 적용지침(GUIDE) 세그먼트 참조
_GUIDE_PREFIX = re.compile(r"(?<![가-힣])지침\s*$")
# 청커 references target: "지침 1장 801.의 1항" / "1편 1장 105" / "5편 5장 405"
_META_REF = re.compile(
    r"^(지침)?\s*(?:(\d{1,2})\s*편\s*)?(\d{1,2})\s*장\s*(\d{3,4})"
    r"(?:\.의\s*(\d+)\s*항)?\.?$")


def _meta_refs(r: dict):
    """청커가 구조화한 references → (guide, 편, 장, 조, 항) 후보.

    본문 정규식이 놓치는 개행 분절("규 칙"·"1 장")을 청커 추출로 보강한다.
    조번호 없는 장·절 단위 target은 제외(결정성 없음)."""
    for ref in r.get("references") or []:
        if ref.get("ref_type") not in ("internal_article", "internal_rule",
                                       "internal_guidance"):
            continue
        m = _META_REF.match((ref.get("target") or "").strip())
        if m:
            yield (bool(m.group(1)) or ref["ref_type"] == "internal_guidance",
                   m.group(2), m.group(3), m.group(4), m.group(5))


def _gate_pick(cands: list[dict], hang: str | None, exclude_id: str) -> dict | None:
    """유일 후보 + 항 실재 + 값 유실 게이트 — 참조 해소 공통 종결부."""
    cands = [c for c in cands if c["chunk_id"] != exclude_id]
    if len(cands) != 1:
        return None
    ref_text = cands[0]["content"][:6000]
    if not _hang_ok(hang, ref_text) or len(_DANGLING.findall(ref_text)) >= 2:
        return None
    return cands[0]


def _hang_ok(hang: str | None, ref_text: str) -> bool:
    """"…의 M항" — 지목한 항이 참조 조문에 실재해야 한다."""
    return not hang or bool(re.search(rf"(?:^|\n)\s*{hang}\.\s", ref_text))


def _pack_ref(cand: dict, no: str) -> dict:
    return {"ref_chunk_id": cand["chunk_id"], "ref_article_no": no,
            "ref_text": cand["content"][:6000],
            "ref_path": " / ".join(cand.get("section_path") or [])}


def find_crossref(r: dict, by_article_no: dict[str, list[dict]],
                  by_ch_no: dict[tuple[str, str], list[dict]] | None = None,
                  guide_by_ch_no: dict[tuple[str, str], list[dict]] | None = None
                  ) -> dict | None:
    """참조 해석 + 참조 대상에 필요한 사실이 실재하는지 사전 확인.

    required_fact 게이트: ① 참조 문장의 내용어가 참조 조문에도 등장해야 하고
    ② 항 번호를 지목했다면 그 항이 실제로 있어야 하며 ③ 참조 조문에 값이
    유실된 매달린 정의(": …일 때 :")가 많으면 배정하지 않는다 — 실측: 값이
    빠진 참조 조문을 받은 생성기가 C1 계수를 날조해 보간함."""
    own = str(r.get("article_no") or "")
    # 장 접두 참조("4장 301.") 우선 — (장, 조) 이중 키라 동번호 모호성이 없다.
    # 내용어 게이트(①)는 생략: 조인 자체가 정밀해 과차단만 남는다.
    for m in CH_REF.finditer(r["content"]):
        if m.group(1):
            continue  # 편 접두 = 문서 간 참조 — find_xdoc_crossref가 처리
        ch, no, hang = m.group(2), m.group(3), m.group(4)
        prev = r["content"][max(0, m.start() - 16):m.start()]
        if _GUIDE_PREFIX.search(prev):
            # "지침 M장 NNN." — 같은 편의 적용지침(GUIDE) 세그먼트로 해소
            cand = _gate_pick((guide_by_ch_no or {}).get((ch, no), []),
                              hang, r["chunk_id"])
            if cand is None:
                continue
            return {**_pack_ref(cand, no), "ref_doc_type": "guidance"}
        if _NAMED_PREFIX.search(prev):
            # "OO규칙/기준 M장 NNN." — 실측 전부 타 문서 참조(자기 0).
            # 자기 문서 조로 조인하면 오연결 → find_nameddoc_crossref가 처리
            continue
        if no == own and ch == str(r.get("chapter_no")):
            continue
        cand = _gate_pick((by_ch_no or {}).get((ch, no), []), hang, r["chunk_id"])
        if cand is None:
            continue
        return _pack_ref(cand, no)
    # 청커 references 보강 — 본문 개행 분절 등으로 정규식이 놓친 같은 편 참조
    for guide, pyeon, ch, no, hang in _meta_refs(r):
        if pyeon:
            continue  # 편 지정 = 문서 간 — find_xdoc_crossref가 처리
        idx = guide_by_ch_no if guide else by_ch_no
        if guide or not (no == own and ch == str(r.get("chapter_no"))):
            cand = _gate_pick((idx or {}).get((ch, no), []), hang, r["chunk_id"])
            if cand is not None:
                out = _pack_ref(cand, no)
                if guide:
                    out["ref_doc_type"] = "guidance"
                return out
    for m in KR_REF.finditer(r["content"]):
        no = m.group(1)
        if no == own:
            continue
        # 장·편·문서명 접두가 붙은 번호("9장 201."·"지침 4편 9장 201.")는 이
        # 루프의 몫이 아니다 — 실측: "지침 4편 9장 201."의 201을 자기 문서
        # 2장 201로 오연결. 접두 참조는 CH/xdoc/named 해소기가 전담한다.
        prev = r["content"][max(0, m.start() - 16):m.start()]
        if re.search(r"(?:\d\s*[편장]|규칙|기준|지침)\s*$", prev):
            continue
        cand = _resolve_no(no, r, by_article_no)
        if cand is None:
            continue
        ref_text = cand["content"][:6000]
        # ① 참조 문장 내용어 ∩ 참조 조문 — 무관한 조 연결 차단
        s = r["content"].rfind("\n", 0, m.start()) + 1
        e = r["content"].find("\n", m.end())
        mention = r["content"][s:e if e != -1 else None]
        nouns = [t for t in _NOUN_RE.findall(mention) if t not in _REF_STOP]
        if nouns and not any(t in ref_text for t in nouns):
            continue
        hang = m.group(2)
        if not _hang_ok(hang, ref_text):
            continue
        # ③ 값 유실(매달린 정의) 조문은 사실 조회가 불가능
        if len(_DANGLING.findall(ref_text)) >= 2:
            continue
        return _pack_ref(cand, no)
    return None


# ── crossref(xdoc): 문서 간 참조 "N편 M장에서 규정하는 X" ────────────────
# 라우터는 같은 문서 조번호("202.의 1항")만 해석했다 — 강선규칙 편 간 참조
# (예: "5편 5장에서 규정하는 제1종 압력용기")는 편-문서 전역 인덱스에서
# 용어 렉시컬 매칭으로 결정적으로 해소한다(임베딩·LLM 불사용).
XDOC_REF = re.compile(
    r"(\d{1,2})\s*편\s*(\d{1,2})\s*장에서 규정하는\s*"
    r"([가-힣A-Za-z0-9\s]{2,25}?)(?:으로|로|과|와|의|을|를|은|는|이|가|에|\s*[,.)])")
_PYEON_STEM = re.compile(r"^(\d{1,2})편")
# 일반어 용어 — 렉시컬 매칭이 무관한 조(일반사항 등)로 이어질 위험이 커 배정 제외
_XDOC_TERM_STOP = {"최소한", "요건", "기준", "사항", "규정", "값", "경우", "내용",
                   "방법", "절차", "일반"}


def find_xdoc_crossref(r: dict, xdoc_index: dict[tuple[str, str], list[dict]],
                       guide_xdoc: dict[tuple[str, str], list[dict]] | None = None
                       ) -> dict | None:
    """문서 간 참조를 결정적으로 해소 — 두 형태를 정밀도 순으로 시도.

    ⓐ "N편 M장 NNN.(의 M항)" (코퍼스 1,880건, 문서명 접두 '선급 및 강선규칙'
       변형 포함): (편, 장, 조번호) 삼중 키 직접 조인 — 용어 매칭 불필요.
    ⓑ "N편 M장에서 규정하는 X" (26건): ① 용어가 조 제목에 있으면 최우선
       ② 아니면 본문 등장 횟수 최다 조. 제목 매치 다수·동률이면 모호 → 스킵."""
    own_key = None
    m_own = _PYEON_STEM.match(nfc(r.get("_source_file") or ""))
    if m_own:
        own_key = (m_own.group(1), str(r.get("chapter_no")))
    # ⓐ 편+장+조번호 — CH_REF의 편 접두 매치만 사용
    for m in CH_REF.finditer(r["content"]):
        if not m.group(1):
            continue  # 편 접두 없음 = 같은 문서 참조(find_crossref가 처리)
        key, no, hang = (m.group(1), m.group(2)), m.group(3), m.group(4)
        prev = r["content"][max(0, m.start() - 16):m.start()]
        is_guide = bool(_GUIDE_PREFIX.search(prev))
        if not is_guide and "지침" in prev[-6:]:
            continue  # "OO지침 4편 2장 102." — 합성어 지침 문서는 named가 처리
        idx = (guide_xdoc or {}) if is_guide else xdoc_index
        cand = _gate_pick([c for c in idx.get(key, [])
                           if str(c.get("article_no")) == no], hang, r["chunk_id"])
        if cand is None:
            continue  # 미존재·중복 — 모호
        return {**_pack_ref(cand, no),
                "ref_source_file": cand.get("_source_file", ""),
                "xdoc": True, "ref_doc_type": cand.get("document_type", "rule"),
                "ref_term": ("지침 " if is_guide else "")
                            + f"{key[0]}편 {key[1]}장 {no}."}
    # ⓑ 편+장+"에서 규정하는" 용어 참조
    for m in XDOC_REF.finditer(r["content"]):
        key, term = (m.group(1), m.group(2)), m.group(3).strip()
        if key == own_key or len(term) < 2 or term in _XDOC_TERM_STOP:
            continue  # 자기 장 참조·빈 용어·일반어(엉뚱한 조 연결 위험)
        cands = xdoc_index.get(key) or []
        titled = [c for c in cands if term in (c.get("article_title") or "")]
        if len(titled) == 1:
            cand = titled[0]
        elif titled:
            continue  # 제목 매치 다수 — 모호
        else:
            scored = [(c["content"].count(term), c) for c in cands
                      if term in c["content"]]
            if not scored:
                continue
            scored.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                continue  # 동률 — 모호
            cand = scored[0][1]
        if cand["chunk_id"] == r["chunk_id"]:
            continue
        if len(_DANGLING.findall(cand["content"][:6000])) >= 2:
            continue  # 값 유실 조문
        return {"ref_chunk_id": cand["chunk_id"],
                "ref_article_no": str(cand.get("article_no") or ""),
                "ref_text": cand["content"][:6000],
                "ref_path": " / ".join(cand.get("section_path") or []),
                "ref_source_file": cand.get("_source_file", ""),
                "xdoc": True, "ref_term": term}
    # ⓒ 청커 references 보강 — 편 지정 참조(개행 분절 등으로 정규식이 놓친 것)
    for guide, pyeon, ch, no, hang in _meta_refs(r):
        if not pyeon:
            continue  # 편 미지정 = 같은 문서 — find_crossref가 처리
        idx = (guide_xdoc or {}) if guide else xdoc_index
        cand = _gate_pick([c for c in idx.get((pyeon, ch), [])
                           if str(c.get("article_no")) == no], hang, r["chunk_id"])
        if cand is not None:
            return {**_pack_ref(cand, no),
                    "ref_source_file": cand.get("_source_file", ""),
                    "xdoc": True, "ref_doc_type": cand.get("document_type", "rule"),
                    "ref_term": ("지침 " if guide else "") + f"{pyeon}편 {ch}장 {no}."}
    return None


# 문서명 접두 참조 "저인화점연료선박규칙 6장 301.(의 M항)" — 실측 35건 전부 타 문서
NAMED_REF = re.compile(
    r"([가-힣]{3,15}(?:규칙|기준|지침))\s*(\d{1,2})\s*장\s*(\d{3,4})\.(?:의\s*(\d+)\s*항)?")


def find_nameddoc_crossref(r: dict, name_index: dict[str, dict[tuple[str, str],
                                                              list[dict]]]
                           ) -> dict | None:
    """문서명 + 장 + 조번호 → 이름이 유일하게 매칭되는 코퍼스 문서의 조로 조인.

    이름은 부분 문자열 매칭("연료선박규칙" ⊂ "저인화점연료선박규칙…") — 매칭
    문서가 여럿이거나 (장, 조) 후보가 유일하지 않으면 모호 → 스킵. 코퍼스에
    없는 문서(적용지침 등)는 자연히 미매칭으로 걸러진다."""
    for m in NAMED_REF.finditer(r["content"]):
        name = re.sub(r"\s", "", m.group(1))
        ch, no, hang = m.group(2), m.group(3), m.group(4)
        stems = [s for s in name_index if name in s]
        if len(stems) != 1:
            continue  # 미존재·다중 매칭 — 모호
        cands = [c for c in name_index[stems[0]].get((ch, no), [])
                 if c["chunk_id"] != r["chunk_id"]]
        if len(cands) != 1:
            continue
        ref_text = cands[0]["content"][:6000]
        if not _hang_ok(hang, ref_text) or len(_DANGLING.findall(ref_text)) >= 2:
            continue
        return {**_pack_ref(cands[0], no),
                "ref_source_file": cands[0].get("_source_file", ""),
                "xdoc": True, "ref_term": f"{m.group(1)} {ch}장 {no}."}
    return None


# ── 코퍼스 라우팅 ───────────────────────────────────────────────────────
def route_corpus(publisher: str = "KR", min_tokens: int = 100,
                 doc_glob: str = "*_chunks.jsonl") -> list[dict]:
    """요건 조당 1건 — (조, 배정 트랙, 트랙별 보조 정보) 목록을 반환."""
    targets: list[dict] = []
    xdoc_index: dict[tuple[str, str], list[dict]] = {}
    guide_xdoc: dict[tuple[str, str], list[dict]] = {}
    name_index: dict[str, dict[tuple[str, str], list[dict]]] = {}
    for f in glob_docs(CHUNK_DIR / publisher, doc_glob):
        parents, tables = [], {}
        for l in f.open(encoding="utf-8"):
            if '"parent"' not in l and '"table"' not in l:
                continue
            r = json.loads(l)
            if r.get("chunk_type") == "table":
                tables[r["chunk_id"]] = r
            elif r.get("chunk_level") == "parent":
                r["publisher"], r["_source_file"] = publisher, nfc(f.stem)
                parents.append(r)
        reqs = [r for r in parents if _is_requirement_parent(r, publisher, min_tokens)]
        # 특수 문서(선종·설비 전용)의 적용범위 조항 텍스트 부착 — 대상어 별칭
        # 도출원('이하 X라 한다' 자기 지칭 선언 등, 검수 11차 후속)
        if reqs and not nfc(f.stem)[0].isdigit():
            scope = next((p for p in parents
                          if "적용" in (p.get("article_title") or "")), None)
            if scope:
                for r in reqs:
                    r["_doc_scope_text"] = scope["content"][:1500]
        if not reqs:
            continue
        # 정의 조는 요건 필터에서 제외되지만 def_link·crossref의 참조 풀로는 필요하다.
        rule_side = [p for p in parents
                     if not p.get("document_type") or p["document_type"] == "rule"]
        # 표 병합은 rule_side 전체(reqs 포함) — 참조·정의 조문의 기준값도 표 안에
        # 있어, 대상 조만 병합하면 참조 텍스트의 빈 자리를 LLM이 외부지식으로
        # 메우는 환각이 생긴다(실측: C1 계수 표 누락 → 값 날조).
        _attach_tables(rule_side, tables)
        terms = extract_terms([p for p in rule_side
                               if DEF_TITLE.search(p.get("article_title") or "")])
        by_no: dict[str, list[dict]] = {}
        by_ch_no: dict[tuple[str, str], list[dict]] = {}
        for p in rule_side:
            if p.get("article_no"):
                by_no.setdefault(str(p["article_no"]), []).append(p)
                if p.get("chapter_no") is not None:
                    by_ch_no.setdefault(
                        (str(p["chapter_no"]), str(p["article_no"])), []).append(p)
        # 적용지침(GUIDE) 세그먼트 인덱스 — "지침 M장 NNN." 참조 해소용.
        # 표 병합은 rule_side와 동일하게 적용(지침 조문도 표 기준값을 가진다).
        guide_side = [p for p in parents if p.get("document_type") == "guidance"]
        _attach_tables(guide_side, tables)
        guide_by_ch_no: dict[tuple[str, str], list[dict]] = {}
        for p in guide_side:
            p["publisher"], p["_source_file"] = publisher, nfc(f.stem)
            if p.get("article_no") and p.get("chapter_no") is not None:
                guide_by_ch_no.setdefault(
                    (str(p["chapter_no"]), str(p["article_no"])), []).append(p)
        for r in reqs:
            feats = {
                "precedence": find_precedence(r, by_no),
                "def_link": find_def_link(r, terms, by_no),
                "unit_convert": find_unit_threshold(r),
                "table_lookup": find_table(r),
                "hierarchy": find_hierarchy(r),
                "crossref": find_crossref(r, by_no, by_ch_no, guide_by_ch_no),
            }
            r["_feats"] = {k: v for k, v in feats.items() if v}
            targets.append(r)
        # 문서 간 참조 인덱스 — 편-시리즈 문서의 (편, 장) → 조 목록 (규칙/지침 분리)
        m_stem = _PYEON_STEM.match(nfc(f.stem))
        if m_stem:
            for p in rule_side:
                if p.get("chapter_no") is not None:
                    xdoc_index.setdefault(
                        (m_stem.group(1), str(p["chapter_no"])), []).append(p)
            for p in guide_side:
                if p.get("chapter_no") is not None:
                    guide_xdoc.setdefault(
                        (m_stem.group(1), str(p["chapter_no"])), []).append(p)
        # 문서명 인덱스 — "저인화점연료선박규칙 6장 301." 류 named 참조 해소용
        name_index[re.sub(r"[\s_]", "", nfc(f.stem))] = by_ch_no

    # 2차: 같은 문서 crossref가 없는 조에 문서 간 참조 해소 시도
    # (편+장 번호형 → 용어형 → 문서명형 순 — 정밀도 순서)
    for r in targets:
        if "crossref" not in r["_feats"]:
            x = (find_xdoc_crossref(r, xdoc_index, guide_xdoc)
                 or find_nameddoc_crossref(r, name_index))
            if x:
                r["_feats"]["crossref"] = x

    # 희소 우선 배정 — 전역 빈도가 낮은 특성부터. 동률은 TRACKS 순.
    counts = {t: sum(1 for r in targets if t in r["_feats"]) for t in TRACKS}
    order = sorted(TRACKS, key=lambda t: (counts[t], TRACKS.index(t)))
    routed = []
    deflink_seen: dict[tuple[str, str], int] = {}
    for r in targets:
        track = next((t for t in order if t in r["_feats"]), None)
        if track == "def_link":
            # 같은 (문서, 용어) 정의 반복 상한 2건 — 실측: 한 지침의 '전문공급자'
            # 정의가 4/9를 차지해 데이터 다양성이 무너짐. 초과분은 차순위 특성으로.
            key = (r["_source_file"], r["_feats"]["def_link"]["term"])
            if deflink_seen.get(key, 0) >= 2:
                track = next((t for t in order
                              if t in r["_feats"] and t != "def_link"), None)
            else:
                deflink_seen[key] = deflink_seen.get(key, 0) + 1
        if track is None:  # 특성 없음(또는 상한 초과) → 스펙/적용성 결정적 교대
            track = FALLBACKS[_stable(f"{r['_source_file']}|{r['chunk_id']}") % 2]
        routed.append({"rule": r, "track": track,
                       "aux": r["_feats"].get(track)})
    return routed
