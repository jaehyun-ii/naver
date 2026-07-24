"""스위트 QA 블라인드 검증 — 의미 수준 오류(용어 치환·전제 불일치) 검출.

정적 인용 검증(부분문자열)은 '원문을 그대로 복사했는가'만 보장하고, 질문·답변이
원문 **의미**와 일치하는지는 못 본다. 실측 사례: 원문 "타두재가 없는 선박 → 최대
길이의 97%"를 "타주가 없는 선박 → 97%"로 서술한 QA가 인용 검증을 통과했다.

이 모듈은 검증 LLM이 생성 맥락 없이 [조항 원문 + QA]만 보고 재검증한다.
- 유사 용어 쌍(첫 글자 동일·길이 근접, QA가 한쪽만 언급)을 정적으로 검출해
  검증 프롬프트에 점검 항목으로 주입한다 — 검출은 프로그램, 판단은 LLM.
- AIREG_VERIFY_LLM_BASE/MODEL로 생성 모델과 분리 가능(verify.py와 동일).

입력: 스위트 트랙 파일 전체(build_suite_qa.TRACK_FILES + compare_qa.jsonl)
출력: suite_verdicts.jsonl {question_id, track, verdict ACCEPT|REJECT, issues, ...}
소비: build_training.suite_rows(REJECT 제외), build_master(quality 반영)

    python -m scripts.aireg_kr.verify_suite [--limit N] [--all]
"""
from __future__ import annotations

import argparse
import os
import re

from . import prompts
from .build_suite_qa import TRACK_FILES
from .common import (OUT_DIR, VERIFY_LLM_BASE, VERIFY_LLM_MODEL, append_jsonl,
                     chat_json, gen_meta, load_done, load_jsonl, log, pmap)

VERDICTS_OUT = OUT_DIR / "suite_verdicts.jsonl"
# 스위트 트랙 파일 + 선급 비교(2-원문 검증 동일하게 적용)
SOURCE_FILES = [fname for fname, _ in TRACK_FILES.values()] + ["compare_qa.jsonl"]

TOKEN_RE = re.compile(r"[가-힣]{2,7}")
GENERIC = {"경우", "선박", "이상", "이하", "초과", "미만", "규정", "조항", "다음",
           "또는", "위한", "있는", "없는", "한다", "하는", "하여", "된다", "대한",
           "말한", "이서", "되는", "따라", "때에", "선상", "있다", "있도록",
           "하며", "하도록", "되도록", "이며", "위하여", "대하여", "있어야",
           # 답변 템플릿 라벨 단어 — 원문에도 흔해 용어 쌍으로 오검출됨(실측: 기준/기체)
           "기준", "근거", "발췌", "판정", "답변", "인용", "원문", "설계값"}
# 동사 활용형 어미 — 프로그램 확정 판정에는 명사형 용어만 사용(있다/있도록·하며류 오탐 차단)
_VERBAL_END = ("다", "며", "고", "서", "여", "록", "면", "니", "지")


def _nounish(t: str) -> bool:
    return not t.endswith(_VERBAL_END)
# 어미 조사 제거(간이) — 긴 것부터 시도, 어간 2자 이상 유지
JOSA = ["으로부터", "에서는", "에서", "으로", "이라", "라", "은", "는", "이", "가",
        "을", "를", "의", "에", "와", "과", "도", "로", "만"]


def _stem(t: str) -> str:
    for j in JOSA:
        if t.endswith(j) and len(t) - len(j) >= 2:
            return t[:len(t) - len(j)]
    return t


def confusable_pairs(article: str, qa_text: str, max_pairs: int = 4) -> list[tuple[str, str]]:
    """원문에 함께 등장하는 유사 용어 쌍 중 QA가 한쪽만 언급한 것을 검출.

    휴리스틱(조사 제거 어간, 첫 글자 동일 + 길이 차 ≤2)이라 과검출은 있지만,
    결과는 검증 LLM의 점검 항목으로만 쓰여 무해하다 — 검출은 프로그램,
    진위 판단은 LLM. 우선순위는 원문 등장 빈도(도메인 핵심 용어가 반복된다)."""
    freq: dict[str, int] = {}
    for t in TOKEN_RE.findall(article):
        st = _stem(t)
        if len(st) >= 2 and st not in GENERIC:
            freq[st] = freq.get(st, 0) + 1
    tokens = sorted(freq)
    pairs = []
    for i, a in enumerate(tokens):
        for b in tokens[i + 1:]:
            if b[0] != a[0]:  # 정렬돼 있어 첫 글자가 다르면 이후는 전부 다름
                break
            if abs(len(a) - len(b)) > 2 or a in b or b in a:
                continue
            if (a in qa_text) != (b in qa_text):  # QA가 한쪽만 언급 — 치환 의심
                pairs.append((a, b))
    pairs.sort(key=lambda p: -(freq[p[0]] + freq[p[1]]))
    return pairs[:max_pairs]


# 라틴 기호 쌍(Zb/Zs·Kix/Kiy류) — 한글 전용 토큰화의 사각지대(실측: Z값 오적용).
# 같은 밑글자·다른 첨자 기호가 원문에 함께 있는데 QA가 한쪽만 쓰면 치환 의심.
# \b는 한글 인접("Zb는")에서 미성립 — ASCII 전후 부정 룩어라운드 사용
_SYM_TOK = re.compile(r"(?<![A-Za-z0-9])[A-Z](?:_?\{?[a-z0-9]{1,2}\}?)(?![a-z0-9])")


def _sym_norm(t: str) -> str:
    return re.sub(r"[${}_]", "", t or "")


def symbol_pairs(article: str, qa_text: str, max_pairs: int = 4) -> list[tuple[str, str]]:
    syms = sorted({m.group(0) for m in _SYM_TOK.finditer(_sym_norm(article))})
    qa = _sym_norm(qa_text)
    by_base: dict[str, list[str]] = {}
    for s in syms:
        by_base.setdefault(s[0], []).append(s)
    pairs = []
    for group in by_base.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if (a in qa) != (b in qa):
                    pairs.append((a, b))
    return pairs[:max_pairs]


def side_pairs(article: str, qa_text: str, max_pairs: int = 2) -> list[tuple[str, str]]:
    """'~측' 대립 용어 쌍(면재측/부착판측) — 첫 글자 동일 조건에 안 걸리는 유형."""
    sides = sorted({m.group(0) for m in re.finditer(r"[가-힣]{1,4}측", article)})
    pairs = []
    for i, a in enumerate(sides):
        for b in sides[i + 1:]:
            if (a in qa_text) != (b in qa_text):
                pairs.append((a, b))
    return pairs[:max_pairs]


# ── 프로그램 선행 검사: 수치 앵커 용어 대조 ─────────────────────────────
# 후행 마침표는 수치에 포함하지 않는다 — "203."은 기준값이 아니라 조 번호다.
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*%?")
# 수치 뒤에 붙으면 참조·연차 표기로 간주(기준값 아님): "203.", "5항", "15년" 등
_REF_TAIL = re.compile(r"^\s*(?:\.|항|장|절|편|호|조|년|번)")


def numeric_term_mismatch(row: dict) -> list[dict]:
    """답변의 수치 주장을 원문 줄에 앵커해 용어 치환을 결정론적으로 검출.

    답변 문장이 유사 용어 쌍의 A를 쓰면서 수치 X를 주장하는데, 원문에서 X가
    등장하는 줄들이 전부 B에만 결부돼 있으면(A 등장 0회) 치환 확정.
    실측 사례를 정확히 잡는다: 답변 "타주가 없는 선박 → 97%" vs
    원문 "타두재가 없는 선박 ... 97 %로 하여야 한다".
    LLM이 필요 없어 자기 검증 편향(생성=검증 모델)의 사각도 없다."""
    answer = row.get("gold_answer", "")
    found = []
    for e in (row.get("evidence") or [])[:3]:
        article = e.get("article_text", "")
        # 문장 단위 분해 — 한 줄에 여러 문장이 있으면 줄 앵커는 용어가 뭉개진다
        segs = [s.strip() for s in re.split(r"(?<=다)\.\s+|\n", article) if s.strip()]
        qa_ctx = f"{row.get('question', '')}\n{answer}"
        kor = [(a, b) for a, b in confusable_pairs(article, qa_ctx)
               if _nounish(a) and _nounish(b)]  # 명사형만 — 활용형 오탐 차단
        # 기호 쌍(Zb/Zs)·'~측' 쌍은 형태가 고정이라 nounish 검사 불요
        ans_n = _sym_norm(answer)
        for a, b in kor + symbol_pairs(article, qa_ctx) + side_pairs(article, qa_ctx):
            used, absent = (a, b) if (a in answer or a in ans_n) else (b, a)
            if used not in answer and used not in ans_n:
                continue
            for sent in re.split(r"(?<=다)\.\s+", answer):
                if used not in sent and used not in _sym_norm(sent):
                    continue
                for m in NUM_RE.finditer(sent):
                    num = m.group(0).strip().replace(" ", "")
                    if len(num) < 2:  # 한 자리 수는 어디에나 앵커돼 무의미
                        continue
                    if _REF_TAIL.match(sent[m.end():]):
                        continue  # 조 번호·연차 표기 — 기준값 아님
                    # 표 HTML 세그먼트는 행마다 수치가 흩어져 앵커 의미가 없다
                    anchor = [s for s in segs
                              if num in s.replace(" ", "") and "<t" not in s[:200]]
                    with_used = [s for s in anchor
                                 if used in s or used in _sym_norm(s)]
                    with_absent = [s for s in anchor
                                   if absent in s or absent in _sym_norm(s)]
                    if with_absent and not with_used:
                        found.append({"used": used, "expected": absent,
                                      "value": num,
                                      "article_line": with_absent[0][:200]})
    return found


def ungrounded_numbers(row: dict, max_n: int = 5) -> list[str]:
    """답변의 수치 중 원문·질문 어디에도 없는 것 — 날조 의심 후보.

    환산·계산 결과(26.4시간 = 1584/60 등)는 정당할 수 있으므로 확정 판정이
    아니라 검증 LLM의 점검 항목으로만 쓴다 — 검출은 프로그램, 판단은 LLM."""
    ground = norm_nums(row.get("question", ""))
    for e in row.get("evidence") or []:
        ground |= norm_nums(e.get("article_text", ""))
    claimed = norm_nums(row.get("gold_answer", ""))
    return sorted(claimed - ground)[:max_n]


def norm_nums(text: str) -> set[str]:
    return {m.group(0).replace(",", "").replace(" ", "").rstrip(".")
            for m in NUM_RE.finditer(text or "")
            if len(m.group(0).strip()) >= 2}


def build_prompt(row: dict) -> str:
    evs = (row.get("evidence") or [])[:3]  # def_link 참조 체인은 3-근거
    articles = "\n\n".join(
        f"[원문 {i}] {e.get('section_path', '')}\n{e.get('article_text', '')}"
        for i, e in enumerate(evs, 1))
    qa_text = f"{row.get('question', '')}\n{row.get('gold_answer', '')}"
    pairs = []
    for e in evs:
        art = e.get("article_text", "")
        pairs += confusable_pairs(art, qa_text)
        pairs += symbol_pairs(art, qa_text)
        pairs += side_pairs(art, qa_text)
    term_watch = (prompts.SUITE_VERIFY_TERM_WATCH.format(
        pairs=", ".join(f"{a}/{b}" for a, b in dict.fromkeys(pairs)))
        if pairs else "")
    ung = ungrounded_numbers(row)
    if ung:
        term_watch += prompts.SUITE_VERIFY_NUM_WATCH.format(nums=", ".join(ung))
    # 상호참조 QA — 특별요건 우선 점검(검수 9차 #16: 프롬프트 지시만으로 미재현,
    # 일반 대안 재허용 서술이 ACCEPT 통과 실측 → 검증기 강제 점검으로 이중화)
    if row.get("track") == "crossref" and len(evs) >= 2:
        term_watch += prompts.SUITE_VERIFY_XREF_WATCH
    judgment_note = (f" — 기대 판정: {row['expected_judgment']}"
                     if row.get("expected_judgment") else "")
    judgment_guide = (prompts.SUITE_VERIFY_JUDGMENT_GUIDE.format(
        judgment=row["expected_judgment"])
        if row.get("expected_judgment") else "")
    return prompts.SUITE_VERIFY.format(
        term_watch=term_watch, judgment_guide=judgment_guide, articles=articles,
        question=row.get("question", ""), judgment_note=judgment_note,
        answer=row.get("gold_answer", ""))


# 검증 기준 번호 → 표준 이슈 코드(외부 검토 반영: 자유문장 issues만으로는 통계·
# 보상모델 활용이 어려움). 코드는 SUITE_VERIFY 기준 1~7과 1:1.
_ISSUE_CODES = {"1": "PREMISE_MISMATCH", "2": "NOT_DERIVABLE_FROM_SOURCE",
                "3": "TERM_MISMATCH", "4": "NUMERIC_BOUNDARY_ERROR",
                "5": "SCENARIO_INCONSISTENT", "6": "INCOMPLETE_ANSWER",
                "7": "PREMISE_INSUFFICIENT"}


def issue_codes(issues: list[str]) -> list[str]:
    out: list[str] = []
    for i in issues or []:
        m = re.match(r"\s*(?:기준\s*)?(\d)\s*(?:[.::]|위반)", i)
        code = _ISSUE_CODES.get(m.group(1)) if m else None
        if code and code not in out:
            out.append(code)
    return out


def verify_row(row: dict) -> dict:
    # 1단계: 프로그램 검사(수치 앵커 용어 대조) — 확정 검출이면 LLM 호출 불필요
    mismatches = numeric_term_mismatch(row)
    if mismatches:
        m = mismatches[0]
        return {"question_id": row["question_id"], "track": row.get("track", ""),
                "task_type": row.get("task_type", ""),
                "verdict": "REJECT",
                "issues": [f"3. 용어 치환(프로그램 확정): 답변은 '{m['used']}'에 대해 "
                           f"{m['value']} 기준을 서술하지만, 원문에서 해당 기준은 "
                           f"'{m['expected']}'에 결부됨 — 원문: \"{m['article_line']}\""],
                "term_mismatch": True, "severity": "critical",
                "issue_codes": ["TERM_MISMATCH"],
                "label_source": "program",
                "_gen": gen_meta("SUITE_VERIFY", model="program/numeric_term_anchor")}
    # 2단계: 블라인드 LLM 검증(전제·유일 도출·용어·수치 경계)
    # 검증은 판단 품질이 생명 — reasoning effort를 기본 high로 (실측: low는 결함
    # 4건 중 0건 검출, high는 3건 + 나머지 1건은 foreign_refs 정적 게이트가 선검출)
    obj = chat_json(build_prompt(row), max_tokens=1400,
                    base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL,
                    reasoning_effort=os.environ.get(
                        "AIREG_VERIFY_REASONING_EFFORT", "high"),
                    # 제공자 노드 스톨 대비 짧은 타임아웃 — 통상 10~40s, 스톨 시
                    # 빨리 끊고 재시도가 다른 노드로 가는 편이 이득(실측 6분 행)
                    timeout=int(os.environ.get("AIREG_VERIFY_TIMEOUT", "150")))
    verdict = obj.get("verdict", "")
    if verdict not in ("ACCEPT", "REJECT"):
        verdict = "REJECT"  # 형식 불량은 보수적으로 거절(검수 큐행)
    return {"question_id": row["question_id"], "track": row.get("track", ""),
            "task_type": row.get("task_type", ""),
            "verdict": verdict,
            "issues": obj.get("issues") or [],
            "issue_codes": issue_codes(obj.get("issues") or []),
            "term_mismatch": bool(obj.get("term_mismatch")),
            "severity": obj.get("severity", ""),
            "label_source": "llm",
            "_gen": gen_meta("SUITE_VERIFY", model=VERIFY_LLM_MODEL)}


def load_rejected() -> set[str]:
    """소비자용 헬퍼 — REJECT된 question_id 집합."""
    return {v["question_id"] for v in load_jsonl(VERDICTS_OUT)
            if v.get("verdict") == "REJECT"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="검증할 QA 수(0=전체)")
    ap.add_argument("--all", action="store_true",
                    help="needs_review 행도 검증(기본은 학습 후보만)")
    args = ap.parse_args(argv)

    done = load_done(VERDICTS_OUT, key="question_id")
    targets = []
    for fname in SOURCE_FILES:
        for row in load_jsonl(OUT_DIR / fname):
            if row["question_id"] in done:
                continue
            if row.get("needs_review") and not args.all:
                continue  # 정적 검증 탈락분은 이미 학습 제외 — 호출 절약
            targets.append(row)
    if args.limit:
        targets = targets[:args.limit]
    log(f"검증 대상 {len(targets)}건 (검증 모델: {VERIFY_LLM_MODEL})")

    def process(row: dict) -> None:
        try:
            v = verify_row(row)
            append_jsonl(VERDICTS_OUT, v)
            mark = "" if v["verdict"] == "ACCEPT" else f" ← {v['issues'][:1]}"
            log(f"{row['question_id']} [{row.get('track', '?')}] {v['verdict']}{mark}")
        except Exception as e:  # noqa: BLE001
            log(f"!! {row['question_id']} 실패: {e}")

    pmap(process, targets, workers=int(os.environ.get("AIREG_WORKERS", "1")))
    rows = load_jsonl(VERDICTS_OUT)
    rej = sum(1 for r in rows if r["verdict"] == "REJECT")
    term = sum(1 for r in rows if r.get("term_mismatch"))
    log(f"검증 누계 {len(rows)}건 — REJECT {rej} (용어 치환 {term}) → {VERDICTS_OUT}")


if __name__ == "__main__":
    main()
