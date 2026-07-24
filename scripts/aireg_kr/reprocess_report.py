"""§10 — 기존 VERIFIER_DISAGREES 재처리 전후 비교 리포트.

v1(중의적 '적용됩니까' 질문) 스냅샷과 v2(modality-aware 질문) 재검증 결과를
case_id로 대조해 해결 유형을 분류한다.

    python -m scripts.aireg_kr.reprocess_report
    → data_aireg/review/reprocess_report.jsonl / .md
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .common import OUT_DIR, load_jsonl, log

REVIEW_DIR = OUT_DIR / "review"


def classify_resolution(before: dict, after: dict | None, *,
                        case_kind: str = "", case_removed: bool = False) -> str:
    """§5 결과 유형 — 해소 경로별 분류."""
    if case_removed:
        # 예외 구조 게이트 REJECT 등으로 케이스 자체가 제거됨
        return "RULE_CARD_REJECTED" if case_kind.startswith("exception") else "CASE_REJECTED"
    if after is None:
        return "NOT_REVERIFIED"
    a_agree = after["label_consistency"]["agreement"]
    if a_agree == "THREE_WAY_MATCH":
        if case_kind.startswith("exception"):
            return "RESOLVED_EXCEPTION_STRUCTURE"
        if case_kind == "compliance_behavior":
            return "RESOLVED_FACT_REALIZATION"
        if case_kind == "numeric_boundary":
            return "RESOLVED_NUMERIC_REASONING"
        return "RESOLVED_BY_INTENT_REWRITE"
    if a_agree == "VERIFIER_ABSTENTION_CONFLICT":
        return "RESOLVED_VERIFIER_ABSTENTION"             if before["label_consistency"]["agreement"] != a_agree else "EXPERT_REVIEW_REQUIRED"
    if a_agree == "SEMANTIC_AXIS_MISMATCH":
        return "STILL_AMBIGUOUS"
    if a_agree == "PROGRAM_DISAGREES":
        return "PROGRAM_EVALUATOR_ERROR"
    return "EXPERT_REVIEW_REQUIRED"


def main() -> None:
    # 비교 기준: 최신 스냅샷 우선(v2 단일패스 → v1) — 직전 잔여 불일치의 재처리를 추적
    before_cases = {c["case_id"]: c for c in load_jsonl(REVIEW_DIR / "cases_before.jsonl")}
    v2 = REVIEW_DIR / "case_verifications_v2_singlepass.jsonl"
    src = v2 if v2.exists() else REVIEW_DIR / "case_verifications_before.jsonl"
    before_ver = {v["case_id"]: v for v in load_jsonl(src)}
    # v2 스냅샷 기준이면 케이스 정의도 당시 것과 대체로 동일 — 질문은 신 케이스에서 읽음
    for cid in before_ver:
        before_cases.setdefault(cid, {})
    after_cases = {c["case_id"]: c for c in load_jsonl(OUT_DIR / "cases.jsonl")}
    after_ver = {v["case_id"]: v for v in load_jsonl(OUT_DIR / "case_verifications.jsonl")}

    rows = []
    for cid, bv in before_ver.items():
        if bv["label_consistency"]["agreement"] == "THREE_WAY_MATCH":
            continue  # 재처리 대상은 불일치 건
        bc, ac, av = before_cases.get(cid, {}), after_cases.get(cid, {}), after_ver.get(cid)
        case_kind = bc.get("case_kind", "")
        case_removed = cid not in after_cases
        rows.append({
            "sample_id": cid,
            "before": {
                "question": (bc.get("prompt") or "").split("[질문]")[-1].strip()[:80],
                "generation_label": bv["label_consistency"].get("generation_label"),
                "program_label": bv["label_consistency"].get("program_label"),
                "verifier_label": bv["label_consistency"].get("verifier_label"),
                "agreement": bv["label_consistency"]["agreement"],
            },
            "after": ({
                "question": (ac.get("question") or "")[:80],
                "rule_modality": ac.get("rule_modality"),
                "question_intent": ac.get("question_intent"),
                "generation_label": av["label_consistency"].get("generation_primary_label"),
                "program_label": av["label_consistency"].get("program_primary_label"),
                "verifier_label": av["label_consistency"].get("verifier_primary_label"),
                "agreement": av["label_consistency"]["agreement"],
            } if av else None),
            "resolution": classify_resolution(bv, av, case_kind=case_kind,
                                              case_removed=case_removed),
        })

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with (REVIEW_DIR / "reprocess_report.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = Counter(r["resolution"] for r in rows)
    b_match = 0
    a_match = sum(1 for r in rows if (r["after"] or {}).get("agreement") == "THREE_WAY_MATCH")
    n_after = sum(1 for r in rows if r["after"])
    lines = ["# VERIFIER_DISAGREES 재처리 리포트", "",
             f"- 재처리 대상(구 불일치): {len(rows)}건",
             f"- 재검증 완료: {n_after}건",
             f"- intent rewrite 후 3자 일치: {a_match}건 "
             f"({a_match / n_after:.0%})" if n_after else "- 재검증 없음",
             f"- 해결 유형: {dict(res)}", ""]
    for r in rows:
        a = r["after"] or {}
        lines.append(f"## {r['sample_id']}")
        lines.append(f"- 전: {r['before']['question']!r} → "
                     f"prog {r['before']['program_label']} / ver {r['before']['verifier_label']}")
        lines.append(f"- 후: {a.get('question')!r} ({a.get('question_intent')}) → "
                     f"prog {a.get('program_label')} / ver {a.get('verifier_label')} "
                     f"= {a.get('agreement')}")
        lines.append(f"- 결과: {r['resolution']}")
        lines.append("")
    (REVIEW_DIR / "reprocess_report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"재처리 리포트 {len(rows)}건 → {REVIEW_DIR / 'reprocess_report.md'} | {dict(res)}")


if __name__ == "__main__":
    main()
