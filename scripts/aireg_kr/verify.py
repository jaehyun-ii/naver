"""AIReg-Bench 선급 변형 — 독립 검증 스테이지 (List2QA 필터 기준 확장).

생성 체인과 분리된 검증자가 excerpt를 **블라인드 재판정**한다: target_label을
주지 않고 검증자가 독립 도출한 라벨을 프로그램이 대조해 verdict를 계산한다.

  ACCEPT            재판정 라벨 일치 + 전 기준 PASS
  ACCEPT_WITH_EDIT  라벨 일치, style_quality(형식)만 FAIL — 표현 수정 후 사용 가능
  REJECT            라벨 불일치 또는 faithfulness/숫자단위/논리 FAIL

검증 모델은 env로 분리(AIREG_VERIFY_LLM_BASE/MODEL, 기본 = 생성 엔드포인트).
build_training이 verdicts.jsonl 존재 시 REJECT를 학습 데이터에서 제외한다.

    python -m scripts.aireg_kr.verify [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os

from . import prompts
from .common import (OUT_DIR, VERIFY_LLM_BASE, VERIFY_LLM_MODEL, append_jsonl,
                     chat_json, gen_meta, load_done, load_jsonl, log, pmap)

EXCERPTS = OUT_DIR / "excerpts.jsonl"
RULE_CARDS = OUT_DIR / "rule_cards.jsonl"
VERDICTS = OUT_DIR / "verdicts.jsonl"

# 라벨 불일치보다 기준 FAIL이 우선 리뷰 대상은 아님 — 불일치가 가장 심각
_CRITICAL = ("faithfulness", "numeric_unit_consistency", "logical_validity")


def compute_verdict(target_label: str, v: dict) -> tuple[str, str, list[str]]:
    """(verdict, review_priority, reasons). 검증자 응답 결손은 REJECT로 보수 처리."""
    reasons: list[str] = []
    derived = v.get("derived_label")
    if derived not in ("compliant", "non_compliant", "uncertain"):
        return "REJECT", "HIGH", ["invalid_verifier_output"]
    if derived != target_label:
        reasons.append(f"label_mismatch:{target_label}->{derived}")
    for k in _CRITICAL:
        if (v.get(k) or {}).get("label") != "PASS":
            reasons.append(f"{k}_fail")
    style_fail = (v.get("style_quality") or {}).get("label") != "PASS"
    if reasons:
        return "REJECT", "HIGH" if reasons[0].startswith("label_mismatch") else "MEDIUM", reasons
    if style_fail:
        return "ACCEPT_WITH_EDIT", "LOW", ["style_quality_fail"]
    return "ACCEPT", "LOW", []


def verify_one(rule: dict, excerpt: dict) -> dict:
    v = chat_json(prompts.VERIFY.format(
        # oneshot 모드는 카드 없이(card=None) 조항 원문만으로 검증
        rule_card=json.dumps(rule.get("card") or {}, ensure_ascii=False),
        article_text=rule["article_text"],
        excerpt=excerpt["text"],
    ), max_tokens=2200, base=VERIFY_LLM_BASE, model=VERIFY_LLM_MODEL)
    verdict, priority, reasons = compute_verdict(excerpt["target_label"], v)
    return {
        "id": excerpt["id"], "rule_id": excerpt["rule_id"],
        "kind": excerpt.get("kind", ""), "target_label": excerpt["target_label"],
        "derived_label": v.get("derived_label"),
        "verdict": verdict, "review_priority": priority, "reasons": reasons,
        "verification": v,
        "_gen": gen_meta("VERIFY", model=VERIFY_LLM_MODEL),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="검증할 excerpt 수(0=전체)")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("AIREG_WORKERS", "1")))
    args = ap.parse_args(argv)

    rules = {r["id"]: r for r in load_jsonl(RULE_CARDS)}
    excerpts = load_jsonl(EXCERPTS)
    done = load_done(VERDICTS)
    todo = [e for e in excerpts if e["id"] not in done and e["rule_id"] in rules]
    if args.limit:
        todo = todo[:args.limit]
    log(f"검증 대상 {len(todo)}/{len(excerpts)} (완료 {len(done)}) — "
        f"verifier={VERIFY_LLM_MODEL}@{VERIFY_LLM_BASE}")

    counts: dict[str, int] = {}

    def process(job: tuple[int, dict]) -> None:
        i, ex = job
        try:
            row = verify_one(rules[ex["rule_id"]], ex)
            append_jsonl(VERDICTS, row)
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
            log(f"({i}/{len(todo)}) {ex['id']} → {row['verdict']}"
                + (f" {row['reasons']}" if row["reasons"] else ""))
        except Exception as e:  # noqa: BLE001
            log(f"({i}/{len(todo)}) !! {ex['id']} 실패: {e} — 다음으로")

    pmap(process, enumerate(todo, 1), workers=args.workers)
    log(f"검증 종료: {counts}")


if __name__ == "__main__":
    main()
