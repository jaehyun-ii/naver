"""AIReg-Bench 선급 변형 — 생성 파이프라인 (rule-by-rule, resume 가능).

조별로 5단계(rule card → 사양 개요 → profile → excerpt → LLM 어노테이션 초안)를
순차 실행하고 data_aireg/*.jsonl 에 append한다. 이미 생성된 id는 건너뛰므로
중단 후 재실행해도 이어서 진행된다.

    python -m scripts.aireg_kr.run_all --n-rules 30 --chapter 7

조당 LLM 호출 17회 × GB10 14B(~7 tok/s, force_reasoning 사고 포함) ≈ 30-45분/조.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from . import prompts
from .common import (OUT_DIR, append_jsonl, chat, chat_json, gen_meta,
                     load_done, load_jsonl, log, pmap, rule_uid, select_rules)

RULE_CARDS = OUT_DIR / "rule_cards.jsonl"
OVERVIEWS = OUT_DIR / "case_overviews.jsonl"
PROFILES = OUT_DIR / "profiles.jsonl"
EXCERPTS = OUT_DIR / "excerpts.jsonl"
ANNOTS = OUT_DIR / "annotations.jsonl"

# 조당 excerpt 구성: (케이스 번호, 종류) — 적합 40% / 미묘 부적합 20% / 명확 부적합 20% / 판단불가 20%
KIND_PLAN = [
    (1, "compliant"),
    (1, "subtle_nc"),
    (1, "insufficient"),
    (2, "compliant"),
    (2, "clear_nc"),
]
N_CASES = 2
# 문서당 1문항 모드: 문서 순번 순환으로 같은 라벨 비율(40/20/20/20) 유지
# (경계값·예외 대체는 파일럿(조당 5) 트랙에만 적용 — 문서당 1문항은 기본 종류 유지)
KINDS_PER_DOC = ["compliant", "subtle_nc", "insufficient", "compliant", "clear_nc"]

LABEL = {"compliant": "compliant", "subtle_nc": "non_compliant",
         "clear_nc": "non_compliant", "insufficient": "uncertain",
         "boundary_c": "compliant", "exception_nm": "non_compliant"}


# ── 결정적 슬롯 대체 — 라벨 비율(40/20/20/20)은 그대로, 사례 다양성만 확장 ──
def _card_has_threshold(card: dict) -> bool:
    for r in card.get("requirements") or []:
        if (r.get("threshold") or "").strip():
            return True
        for c in r.get("conditions") or []:
            if c.get("operator") in ("이상", "이하", "초과", "미만") and str(c.get("value") or "").strip():
                return True
    return False


def _card_has_exception(card: dict) -> bool:
    return any((r.get("exception") or "").strip() for r in card.get("requirements") or [])


def plan_for(card: dict) -> list[tuple[int, str]]:
    """수치 기준이 있으면 두 번째 적합 슬롯을 경계값 사례로, 예외가 있으면
    미묘 부적합 슬롯을 예외 부분충족 사례로 대체. 카드 내용 기반이라 재현 가능."""
    plan = list(KIND_PLAN)
    if _card_has_threshold(card):
        plan[plan.index((2, "compliant"))] = (2, "boundary_c")
    if _card_has_exception(card):
        plan[plan.index((1, "subtle_nc"))] = (1, "exception_nm")
    return plan


def fmt_section_path(chunk: dict) -> str:
    sp = chunk.get("section_path") or []
    return " > ".join(sp)


# ── Step 0 ──────────────────────────────────────────────────────────────
def gen_rule_card(chunk: dict, done: set[str]) -> dict:
    rid = rule_uid(chunk)
    if rid in done:
        return next(r for r in load_jsonl(RULE_CARDS) if r["id"] == rid)
    card = chat_json(prompts.RULE_CARD.format(
        doc_title=chunk["doc_title"],
        section_path=fmt_section_path(chunk),
        article_text=chunk["content"],
    ), max_tokens=2000)
    row = {
        "id": rid,
        "chunk_id": chunk["chunk_id"],
        "publisher": chunk.get("publisher", "KR"),
        "language": chunk.get("language", "ko"),
        "source_file": chunk.get("_source_file", ""),
        "doc_title": chunk.get("doc_title", ""),
        "section_path": fmt_section_path(chunk),
        "chapter_no": chunk.get("chapter_no"),
        "article_no": chunk.get("article_no"),
        "article_title": chunk.get("article_title"),
        "article_text": chunk["content"],
        "card": card,
        "_gen": gen_meta("RULE_CARD"),
    }
    append_jsonl(RULE_CARDS, row)
    return row


# ── Step 1 ──────────────────────────────────────────────────────────────
def parse_cases(text: str) -> list[str]:
    # 모델이 "**Case 1.**"처럼 볼드로 감싸는 경우가 있어 관대하게 매칭
    blocks = re.split(r"^\**\s*Case\s+\d+\.\**\s*", text, flags=re.MULTILINE)
    return [" ".join(b.split()) for b in blocks[1:] if b.strip()]


def gen_overviews(rule: dict, done: set[str], n_cases: int = N_CASES) -> dict[int, str]:
    have = {int(r["id"].rsplit("case", 1)[1]): r["text"]
            for r in load_jsonl(OVERVIEWS) if r["rule_id"] == rule["id"]}
    if len(have) >= n_cases:
        return have
    card = rule["card"]
    applic = json.dumps(card.get("applicability", {}), ensure_ascii=False)
    for attempt in range(3):
        ans = chat(prompts.CASE_OVERVIEW.format(
            n_cases=n_cases, rule_title=card.get("rule_title", rule["article_title"]),
            applicability=applic,
        ), max_tokens=2200, temperature=0.7 + 0.1 * attempt, must_contain="Case")
        cases = parse_cases(ans)
        # 개요의 비한국어 혼입은 profile·excerpt 전체로 전파되므로 원천에서 차단
        if len(cases) >= n_cases and not any(NON_KOREAN.search(c) for c in cases[:n_cases]):
            break
    if len(cases) < n_cases:
        raise RuntimeError(f"{rule['id']}: 사양 개요 파싱 실패 ({len(cases)}개)")
    out = {}
    for i, text in enumerate(cases[:n_cases], start=1):
        append_jsonl(OVERVIEWS, {"id": f"{rule['id']}::case{i}", "rule_id": rule["id"], "text": text,
                                 "_gen": gen_meta("CASE_OVERVIEW")})
        out[i] = text
    return out


# ── Step 2 ──────────────────────────────────────────────────────────────
def gen_profile(rule: dict, case_no: int, case_text: str, kind: str, done: set[str]) -> dict:
    pid = f"{rule['id']}::case{case_no}::{kind}"
    if pid in done:
        return next(r for r in load_jsonl(PROFILES) if r["id"] == pid)
    common = dict(rule_card=json.dumps(rule["card"], ensure_ascii=False),
                  article_text=rule["article_text"], case_overview=case_text)
    prompt_id = "PROFILE"
    if kind in ("subtle_nc", "clear_nc"):
        style, note = prompts.STYLE_SUBTLE if kind == "subtle_nc" else prompts.STYLE_CLEAR
        p = prompts.PROFILE_NON_COMPLIANT.format(style=style, style_note=note, **common)
    elif kind == "compliant":
        p = prompts.PROFILE_COMPLIANT.format(**common)
    elif kind == "boundary_c":
        p, prompt_id = prompts.PROFILE_BOUNDARY.format(**common), "PROFILE_BOUNDARY"
    elif kind == "exception_nm":
        p, prompt_id = prompts.PROFILE_EXCEPTION_NM.format(**common), "PROFILE_EXCEPTION_NM"
    else:
        p = prompts.PROFILE_INSUFFICIENT.format(**common)
    text = chat(p, max_tokens=1800, temperature=0.6, must_contain="인용 근거")
    for attempt in range(2):
        if not NON_KOREAN.search(text):
            break
        log(f"    profile 재생성({pid}): non_korean_mix")
        text = chat(p, max_tokens=1800, temperature=0.7 + 0.1 * attempt, must_contain="인용 근거")
    row = {"id": pid, "rule_id": rule["id"], "case_no": case_no, "kind": kind,
           "target_label": LABEL[kind], "text": text, "_gen": gen_meta(prompt_id)}
    append_jsonl(PROFILES, row)
    return row


# ── Step 3 ──────────────────────────────────────────────────────────────
# 생성물 자동 검증: 규정 인용/판단 언어(label leakage)와 비한국어 혼입은 재생성 사유
LEAKAGE = re.compile(r"규정|조항|선급\s*기준|하여야\s*하")
NON_KOREAN = re.compile(r"[一-鿿]|[가-힣]+[a-zA-Z]+")


def excerpt_quality_flags(body: str) -> list[str]:
    flags = []
    if LEAKAGE.search(body):
        flags.append("label_leakage")
    if NON_KOREAN.search(body):
        flags.append("non_korean_mix")
    return flags


def gen_excerpt(rule: dict, case_text: str, profile: dict, done: set[str]) -> dict:
    if profile["id"] in done:
        return next(r for r in load_jsonl(EXCERPTS) if r["id"] == profile["id"])
    best, best_flags = None, None
    for attempt in range(3):
        ans = chat(prompts.EXCERPT.format(
            article_text=rule["article_text"], case_overview=case_text, profile=profile["text"],
        ), max_tokens=2400, temperature=0.6 + 0.1 * attempt, must_contain="본문")
        m = re.search(r"\**본문\**\s*[:：]\s*\n?(.*)", ans, flags=re.DOTALL)
        body = m.group(1).strip() if m else ans
        flags = excerpt_quality_flags(body)
        if best is None or len(flags) < len(best_flags):
            best, best_flags = ans, flags
        if not flags:
            break
        log(f"    excerpt 재생성({profile['id']}): {flags}")
    ans = best
    m = re.search(r"\**본문\**\s*[:：]\s*\n?(.*)", ans, flags=re.DOTALL)
    body = m.group(1).strip() if m else ans
    title = ""
    mt = re.search(r"\**문서 제목\**\s*[:：]\s*(.+)", ans)
    if mt:
        title = mt.group(1).strip()
    row = {"id": profile["id"], "rule_id": rule["id"], "case_no": profile["case_no"],
           "kind": profile["kind"], "target_label": profile["target_label"],
           "doc_type_title": title, "text": body, "raw": ans,
           "quality_flags": best_flags, "_gen": gen_meta("EXCERPT")}
    append_jsonl(EXCERPTS, row)
    return row


# ── 원샷 모드: 조당 생성 1콜 (카드·개요·profile·어노테이션 생략) ─────────
def gen_oneshot(chunk: dict, kind: str) -> None:
    """조항 원문 → excerpt 직접 생성. 검증은 verify(블라인드 1콜)가 전담.

    rule_cards에는 조인용 최소 행(card=None, LLM 0콜)만 남긴다 — 스펙QA·
    상호참조·구조화 사례 트랙이 필요하면 --cards-only로 카드를 별도 보강한다."""
    rid = rule_uid(chunk)
    eid = f"{rid}::case1::{kind}"
    if eid in load_done(EXCERPTS):
        return
    if rid not in load_done(RULE_CARDS):
        append_jsonl(RULE_CARDS, {
            "id": rid, "chunk_id": chunk["chunk_id"],
            "publisher": chunk.get("publisher", "KR"),
            "language": chunk.get("language", "ko"),
            "source_file": chunk.get("_source_file", ""),
            "doc_title": chunk.get("doc_title", ""),
            "section_path": fmt_section_path(chunk),
            "chapter_no": chunk.get("chapter_no"),
            "article_no": chunk.get("article_no"),
            "article_title": chunk.get("article_title"),
            "article_text": chunk["content"],
            "card": None,
            "_gen": gen_meta("ONESHOT_CASE", model="deterministic"),
        })
    obj, body, flags = {}, "", ["empty"]
    for attempt in range(3):
        obj = chat_json(prompts.ONESHOT_CASE.format(
            article_text=chunk["content"],
            kind_instruction=prompts.KIND_INSTRUCTIONS[kind],
        ), max_tokens=2400)
        body = str(obj.get("excerpt") or "").strip()
        flags = excerpt_quality_flags(body)
        if len(body) < 200:
            flags.append("too_short")
        if not flags:
            break
        log(f"    oneshot 재생성({eid}): {flags}")
    append_jsonl(EXCERPTS, {
        "id": eid, "rule_id": rid, "case_no": 1, "kind": kind,
        "target_label": LABEL[kind],
        "doc_type_title": str(obj.get("excerpt_title") or ""),
        "text": body, "fact_plan": obj.get("fact_plan"),
        "quality_flags": flags, "_gen": gen_meta("ONESHOT_CASE")})


# ── Step 4 ──────────────────────────────────────────────────────────────
def gen_annotation(rule: dict, excerpt: dict, done: set[str]) -> None:
    if excerpt["id"] in done:
        return
    ann = chat_json(prompts.ANNOTATE.format(
        rule_card=json.dumps(rule["card"], ensure_ascii=False),
        article_text=rule["article_text"], excerpt=excerpt["text"],
    ), max_tokens=1800)
    score = ann.get("compliance_score")
    label = excerpt["target_label"]
    consistent = (
        (label == "compliant" and isinstance(score, int) and score >= 4)
        or (label == "non_compliant" and isinstance(score, int) and score <= 2)
        or (label == "uncertain" and score == 3)
    )
    append_jsonl(ANNOTS, {"id": excerpt["id"], "rule_id": rule["id"],
                          "target_label": label, "annotation": ann,
                          "label_consistent": consistent, "_gen": gen_meta("ANNOTATE")})


# ── main ────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rules", type=int, default=30)
    ap.add_argument("--publisher", default="KR", help="발행처 (data_chunks/ 하위 디렉터리명)")
    ap.add_argument("--doc-glob", default=None, help="발행처 내 문서 파일 글롭")
    ap.add_argument("--chapters", default=None,
                    help="우선순위 순 장 번호(쉼표 구분, KR 기본 '7,2', 비-KR 기본 전체)")
    ap.add_argument("--per-doc", action="store_true",
                    help="판정 트랙을 문서당 1문항으로: 문서별 대표 조 1개 × excerpt 1건 (5콜/문서)")
    ap.add_argument("--publishers", default=None,
                    help="--per-doc 대상 발행처(쉼표 구분, 기본 전체 7개)")
    ap.add_argument("--cards-only", action="store_true",
                    help="rule card만 생성(조당 1콜) — 스펙QA·적용성·상호참조 트랙의 조당 확장용")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AIREG_WORKERS", "1")),
                    help="조항 병렬 워커 수 (MLX 서버 동시 배치 활용, 기본 1)")
    ap.add_argument("--oneshot", action="store_true",
                    help="조당 문항 1개를 생성 1콜로 (카드·profile 생략, 라벨 순환)")
    args = ap.parse_args(argv)

    if args.per_doc:
        from .common import ALL_PUBLISHERS, select_one_per_doc
        rules = select_one_per_doc(publishers=args.publishers or ALL_PUBLISHERS)
        if args.n_rules:
            rules = rules[:args.n_rules]
        log(f"대상 문서 {len(rules)}개 (문서당 1문항, 라벨 순환)")
    else:
        chapters = args.chapters if args.chapters is not None else ("7,2" if args.publisher == "KR" else "")
        rules = select_rules(chapters=chapters, n=args.n_rules,
                             publisher=args.publisher, doc_glob=args.doc_glob)
        log(f"대상 조항 {len(rules)}개 ({args.publisher}, 장={chapters or '전체'})")

    def process(job: tuple[int, dict]) -> None:
        idx, chunk = job
        rid = rule_uid(chunk)
        try:
            log(f"({idx}/{len(rules)}) {rid} {chunk.get('article_title', '')}")
            if args.oneshot:
                gen_oneshot(chunk, KINDS_PER_DOC[(idx - 1) % len(KINDS_PER_DOC)])
                log(f"  {rid} 완료")
                return
            rule = gen_rule_card(chunk, load_done(RULE_CARDS))
            if args.cards_only:
                return
            plan = ([(1, KINDS_PER_DOC[(idx - 1) % len(KINDS_PER_DOC)])] if args.per_doc
                    else plan_for(rule["card"]))
            n_cases = 1 if args.per_doc else N_CASES
            overviews = gen_overviews(rule, load_done(OVERVIEWS), n_cases=n_cases)
            done_p, done_e, done_a = (load_done(PROFILES), load_done(EXCERPTS), load_done(ANNOTS))
            for case_no, kind in plan:
                profile = gen_profile(rule, case_no, overviews[case_no], kind, done_p)
                excerpt = gen_excerpt(rule, overviews[case_no], profile, done_e)
                gen_annotation(rule, excerpt, done_a)
            log(f"  {rid} 완료")
        except Exception as e:  # noqa: BLE001
            log(f"  !! {rid} 실패: {e} — 다음 조항으로")

    pmap(process, enumerate(rules, start=1), workers=args.workers)
    log("파이프라인 종료")


if __name__ == "__main__":
    main()
