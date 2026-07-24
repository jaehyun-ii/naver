"""[레거시 단독 실행용] 스위트 태스크 ②·③ — 스펙 QA(조당 N개) + 적용성 판단.

파이프라인 기본 흐름에서는 build_suite_qa(조당 1건, 조 특성 라우팅)가 이 트랙을 담당한다.
조당 다건 확장(파일럿 심화)이 필요할 때만 단독 실행하십시오.

AIReg-KR 판정 트랙(④)과 달리 excerpt 합성이 필요 없어 조당 LLM 2회로 끝난다.
rule_cards.jsonl(판정 파이프라인 Step 0 산출물)을 입력으로 재사용.

- 스펙 QA(②): rule card 요건(기준치·예외·증거)에서 실무형 QA 생성.
  KRX-Bench 품질검수 기준 적용(대상 명시·정답 유일성·외부지식 금지) +
  evidence_quote가 원문에 실제로 존재하는지 자동 검증(불일치 → needs_review).
  실제 조문에 근거하므로 synthetic=false.
- 적용성 판단(③): applicability 조건 기준 적용/미적용/조건부 시나리오 2개.
  시나리오는 합성이므로 synthetic=true.

출력은 benchmark_qa.jsonl과 동일 스키마(question_id/task_type/question/gold_answer/
evidence/metadata) — 평가 하니스에서 파일 concat으로 태스크 스위트 구성.

    python -m scripts.aireg_kr.build_spec_qa   →  data_aireg/{spec_qa,applicability_qa}.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

from . import prompts
from .common import OUT_DIR, append_jsonl, chat_json, load_done, load_jsonl, log

SPEC_OUT = OUT_DIR / "spec_qa.jsonl"
APPLIC_OUT = OUT_DIR / "applicability_qa.jsonl"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def quote_in_article(quote: str, article: str) -> bool:
    q, a = norm(quote), norm(article)
    if not q:
        return False
    if q in a:
        return True
    # 생성 모델이 인용을 다듬는 경우 대비: 앞 40자 부분 일치 허용
    return len(q) >= 40 and q[:40] in a


def meta(rule: dict, synthetic: bool) -> dict:
    return {
        "publisher": rule.get("publisher", "KR"),
        "rule_language": rule.get("language", "ko"),
        "doc_title": rule.get("doc_title", ""),
        "article_no": rule.get("article_no"),
        "article_title": rule.get("article_title"),
        "synthetic": synthetic,
    }


def evidence(rule: dict, quote: str = "", requirement_id: str = "") -> list[dict]:
    return [{
        "chunk_id": rule["chunk_id"],
        "source_file": rule.get("source_file", ""),
        "section_path": rule["section_path"],
        "quote": quote,
        "requirement_ids": [requirement_id] if requirement_id else [],
        "article_text": rule["article_text"],
    }]


# ── 태스크 ②: 스펙 QA ───────────────────────────────────────────────────
def gen_spec_qa(rule: dict, n_qa: int) -> None:
    obj = chat_json(prompts.SPEC_QA.format(
        n_qa=n_qa, rule_card=json.dumps(rule["card"], ensure_ascii=False),
        article_text=rule["article_text"]), max_tokens=2200)
    for i, qa in enumerate(obj.get("qa", [])[:n_qa], start=1):
        if not qa.get("question") or not qa.get("answer"):
            continue
        quote_ok = quote_in_article(qa.get("evidence_quote", ""), rule["article_text"])
        append_jsonl(SPEC_OUT, {
            "question_id": f"{rule['id']}::spec{i}",
            "task_type": "스펙 조회형",
            "question": qa["question"],
            "gold_answer": qa["answer"],
            "evidence": evidence(rule, qa.get("evidence_quote", ""),
                                 qa.get("related_requirement_id", "")),
            "needs_review": not quote_ok,
            "metadata": meta(rule, synthetic=False),
        })


# ── 태스크 ③: 적용성 판단 ───────────────────────────────────────────────
APPLIC_LABELS = {"applicable", "not_applicable", "conditional"}


def gen_applicability(rule: dict) -> None:
    obj = chat_json(prompts.APPLICABILITY.format(
        rule_card=json.dumps(rule["card"], ensure_ascii=False),
        article_text=rule["article_text"]), max_tokens=1800)
    for i, item in enumerate(obj.get("items", [])[:2], start=1):
        label = item.get("label", "")
        if not item.get("scenario") or label not in APPLIC_LABELS:
            continue
        q = (f"다음 선박/설비에 「{rule['section_path']}」이 적용되는가? "
             "적용 조건을 원문에 근거해 판단하라.\n\n[선박/설비 개요]\n" + item["scenario"])
        append_jsonl(APPLIC_OUT, {
            "question_id": f"{rule['id']}::applic{i}",
            "task_type": "적용성 판단형",
            "question": q,
            "expected_judgment": label,
            "gold_answer": item.get("reason", ""),
            "evidence": evidence(rule),
            "needs_review": False,
            "metadata": meta(rule, synthetic=True),
        })


# ── main ────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qa", type=int, default=4, help="조당 스펙 QA 최대 개수")
    args = ap.parse_args(argv)

    rules = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    done_spec = {qid.split("::spec")[0] for qid in load_done(SPEC_OUT, key="question_id")}
    done_app = {qid.split("::applic")[0] for qid in load_done(APPLIC_OUT, key="question_id")}

    for rule in rules:
        try:
            if rule["id"] not in done_spec:
                gen_spec_qa(rule, args.n_qa)
            if rule["id"] not in done_app:
                gen_applicability(rule)
            log(f"{rule['id']} 완료")
        except Exception as e:  # noqa: BLE001
            log(f"!! {rule['id']} 실패: {e}")

    n_s, n_a = len(load_jsonl(SPEC_OUT)), len(load_jsonl(APPLIC_OUT))
    review = sum(1 for r in load_jsonl(SPEC_OUT) if r.get("needs_review"))
    log(f"스펙 QA {n_s}건(인용 불일치 {review}) | 적용성 {n_a}건")


if __name__ == "__main__":
    main()
