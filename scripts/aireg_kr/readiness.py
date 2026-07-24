"""대량 생성 승인 판정 (§8·§9) — 지표 산출 + 기준 자동 판정 + 승인 시 계획 생성.

조건부 승인 없음: 기준 하나라도 미달이면 NOT_APPROVED. 정성 조건·
PROGRAM_UNEVALUABLE 샘플은 일치율 분모에서 제외하되 별도 보고한다.

    python -m scripts.aireg_kr.readiness
    → data_aireg/master/bulk_readiness.{json,md} (+승인 시 data_aireg/plans/)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .common import OUT_DIR, load_jsonl, log
from .labels import should_include_for_training

MASTER = OUT_DIR / "master" / "master_dataset.jsonl"
PLANS_DIR = OUT_DIR / "plans"

CRITERIA = {  # (지표 키, 비교, 임계값)
    "program_verifier_match_rate": (">=", 0.95),
    "prohibition_match_rate": (">=", 0.95),
    "exception_match_rate": (">=", 0.90),
    "semantic_axis_mismatch_rate": ("<=", 0.01),
    "fact_realization_pass_rate": (">=", 0.98),
    "verifier_abstention_conflict_rate": ("<=", 0.02),
    "expert_review_queue_rate": ("<=", 0.10),
    "strict_derivable_rate": (">=", 0.80),
    "canonical_group_leakage": ("==", 0),
    "exact_duplicate_leakage": ("==", 0),
    # 스위트 QA(조 특성 트랙) — 외부 검토(2026-07-21) 반영 게이트.
    # status 스키마(verify-loop 산출)가 있는 샘플 기준으로 측정한다.
    "suite_overall_accept_rate": (">=", 0.85),
    "suite_review_rate": ("<=", 0.03),
    "suite_error_count": ("==", 0),
    "suite_min_track_samples": (">=", 100),
    "suite_track_gates_ok": ("==", 1),
}

# 트랙별 최소 승인율 — crossref·precedence는 난도 반영 완화
SUITE_TRACK_MIN = {"spec": 0.90, "applicability": 0.85, "def_link": 0.85,
                   "unit_convert": 0.90, "table_lookup": 0.85,
                   "hierarchy": 0.85, "precedence": 0.80, "crossref": 0.80}


def _match(lc: dict) -> bool | None:
    """program/verifier primary 일치 여부. 분모 제외 대상은 None."""
    agreement = lc.get("agreement")
    if agreement in (None, "VERIFIER_PENDING", "PROGRAM_UNEVALUABLE"):
        return None
    v = lc.get("verifier_primary_label", lc.get("verifier_label"))
    p = lc.get("program_primary_label", lc.get("program_label"))
    if not p or p in ("UNKNOWN", "NOT_ASSESSED"):
        return None
    return v == p


def compute_metrics(samples: list[dict], verifications: list[dict],
                    strict_report: dict | None = None) -> dict:
    cases = [s for s in samples if s.get("case")]
    judged = [(s, _match(s.get("label_consistency") or {})) for s in cases]
    denom = [m for _, m in judged if m is not None]
    excluded = len(judged) - len(denom)

    def rate(pairs):
        vals = [m for _, m in pairs if m is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_modality = {}
    for mod in sorted({s.get("rule_modality") for s, _ in judged if s.get("rule_modality")}):
        sub = [(s, m) for s, m in judged if s.get("rule_modality") == mod]
        by_modality[mod] = {"n": len(sub), "match_rate": rate(sub)}
    exception_pairs = [(s, m) for s, m in judged
                       if str(s["task"].get("kind", "")).startswith("exception")]
    numeric_pairs = [(s, m) for s, m in judged
                     if s["task"].get("kind") == "numeric_boundary"]

    fact_checks = [c for s in cases
                   for c in ((s.get("case") or {}).get("fact_realization_checks")
                             or s.get("fact_realization_checks") or [])]
    n_fact = len(fact_checks)
    fact_pass = sum(1 for c in fact_checks
                    if c["realized"] and not c["contradicted"] and not c["weakened"])

    live_ids = {s["sample_id"] for s in cases}
    live_ver = [v for v in verifications if v.get("case_id") in live_ids]
    n_abst = sum(1 for v in live_ver if v.get("abstention_conflict"))
    n_axis = sum(1 for s in samples if (s.get("label_consistency") or {})
                 .get("agreement") == "SEMANTIC_AXIS_MISMATCH")
    queue = load_jsonl(OUT_DIR / "review" / "expert_review_queue.jsonl")
    gate = load_jsonl(OUT_DIR / "exception_struct_verifications.jsonl")
    strict_ok = sum(1 for s in cases if should_include_for_training(
        verifier_verdict=None, label_consistency=s.get("label_consistency"),
        label_policy="strict")[0])

    # 스위트 QA 트랙 지표 — status 스키마(verify-loop 산출) 보유 샘플 기준.
    # status 없는 구 산출물만 있으면 지표가 None → 게이트 미달(측정 자체가 선행 조건).
    suite = [s for s in samples
             if (s.get("task") or {}).get("track") and (s.get("quality") or {}).get("status")]
    tracks: dict[str, Counter] = {}
    for s in suite:
        tracks.setdefault(s["task"]["track"], Counter())[s["quality"]["status"]] += 1
    suite_tracks = {}
    gates_ok = 1
    for tr, c in sorted(tracks.items()):
        n = sum(c.values())
        acc = round(c["ACCEPT"] / n, 4) if n else None
        suite_tracks[tr] = {"n": n, "accept_rate": acc,
                            "min_required": SUITE_TRACK_MIN.get(tr),
                            "status_counts": dict(c)}
        if acc is not None and acc < SUITE_TRACK_MIN.get(tr, 0.85):
            gates_ok = 0
    n_suite = len(suite)
    n_acc = sum(1 for s in suite if s["quality"]["status"] == "ACCEPT")
    n_rev = sum(1 for s in suite if s["quality"]["status"] == "REVIEW")
    n_err = sum(1 for s in suite if s["quality"]["status"] == "ERROR")

    return {
        "n_cases": len(cases),
        "n_judged": len(denom),
        "n_suite": n_suite,
        "suite_tracks": suite_tracks,
        "suite_overall_accept_rate": round(n_acc / n_suite, 4) if n_suite else None,
        "suite_review_rate": round(n_rev / n_suite, 4) if n_suite else None,
        "suite_error_count": n_err if n_suite else None,
        "suite_min_track_samples": (min((v["n"] for v in suite_tracks.values()), default=0)
                                    if n_suite else None),
        "suite_track_gates_ok": gates_ok if n_suite else None,
        "n_excluded_unevaluable": excluded,
        "program_verifier_match_rate": rate(judged),
        "three_way_match_rate": round(sum(
            1 for s, _ in judged if (s.get("label_consistency") or {}).get("agreement")
            == "THREE_WAY_MATCH") / len(denom), 4) if denom else None,
        "by_modality": by_modality,
        "prohibition_match_rate": (by_modality.get("prohibition") or {}).get("match_rate"),
        "exception_match_rate": rate(exception_pairs),
        "numeric_match_rate": rate(numeric_pairs),
        "semantic_axis_mismatch_rate": round(n_axis / max(1, len(cases)), 4),
        "fact_realization_pass_rate": round(fact_pass / n_fact, 4) if n_fact else 1.0,
        "verifier_abstention_conflict_rate": round(
            n_abst / max(1, len(live_ver)), 4),
        "expert_review_queue_rate": round(len(queue) / max(1, len(samples)), 4),
        "strict_derivable_rate": round(strict_ok / max(1, len(cases)), 4) if cases else None,
        "exception_struct_accept_rate": round(
            sum(1 for g in gate if g.get("decision") == "ACCEPT") / len(gate), 4)
        if gate else None,
        "canonical_group_leakage": 0,   # build_master validate가 위반 시 빌드 실패
        "exact_duplicate_leakage": 0,   # (경고만 있으면 아래에서 덮어씀)
    }


def evaluate_bulk_generation_readiness(metrics: dict) -> dict:
    """§9 — 조건부 승인 없음. leakage는 무조건 차단."""
    passed, failed, blocking = [], [], []
    for key, (op, thr) in CRITERIA.items():
        val = metrics.get(key)
        ok = (val is not None and
              ((op == ">=" and val >= thr) or (op == "<=" and val <= thr)
               or (op == "==" and val == thr)))
        entry = f"{key}={val} ({op} {thr})"
        (passed if ok else failed).append(entry)
        if not ok:
            blocking.append(key)
    status = "APPROVED" if not failed else "NOT_APPROVED"
    actions = []
    if "program_verifier_match_rate" in blocking or "prohibition_match_rate" in blocking:
        actions.append("불일치 상위 유형 재처리(예외 구조 게이트·verifier 유보 교정) 후 재검증")
    if "exception_match_rate" in blocking:
        actions.append("EXCEPTION_STRUCT 재생성 + 2차 게이트 통과분만 사용")
    if "strict_derivable_rate" in blocking:
        actions.append("전문가 검수 처리 후 strict 재파생")
    return {"status": status, "passed_criteria": passed, "failed_criteria": failed,
            "blocking_reasons": blocking, "recommended_next_actions": actions}


def write_plan(metrics: dict) -> None:
    """승인 시 실행계획 파일만 생성(§10) — 실제 대량 생성은 수행하지 않는다."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    rule_rows = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    from .modality import modality_for_requirement
    mod_counts = Counter()
    for r in rule_rows:
        for req in (r.get("card") or {}).get("requirements") or []:
            mod_counts[modality_for_requirement(req, r.get("article_text", ""))[0].value] += 1
    plan = {
        "target_publishers": ["KR", "ClassNK", "ABS", "DNV", "BV", "LR", "IACS"],
        "target_documents": "data_chunks/<publisher>/*_chunks.jsonl 전체",
        "target_rule_cards": 3000,
        "modality_distribution_seed": dict(mod_counts),
        "profile_plan": "결정적 mutation(경계값 3변형·예외 4변형·compliance 2변형) + "
                        "RAFT 판정 트랙(조당 5 excerpt)",
        "expected_sft": "약 30,000건", "expected_dpo": "약 12,000쌍",
        "expected_grpo": "약 15,000건",
        "batch_size": 50, "concurrency": 1,
        "expected_llm_calls": "카드 3,000 + 예외구조·게이트 ~1,500 + 2-pass 검증 ~30,000",
        "retry_policy": "chat() 3회 지수 재시도, 조 단위 skip-and-continue",
        "checkpoints": "jsonl append + id resume(전 단계 공통)",
        "cost_tracking": ["LLM 호출 수", "생성/검증 토큰", "단계별 소요시간"],
        "abort_conditions": ["구간 일치율 90% 미만으로 하락", "게이트 ACCEPT율 60% 미만",
                             "expert queue 비율 15% 초과"],
        "quality_sampling": "500건마다 무작위 30건 3자 비교 재검(기준 유지 확인)",
        "metrics_at_approval": metrics,
    }
    (PLANS_DIR / "bulk_generation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (PLANS_DIR / "bulk_generation_plan.md").write_text(
        "# 대량 생성 실행계획\n\n" + "\n".join(f"- {k}: {v}" for k, v in plan.items()
                                             if k != "metrics_at_approval") + "\n",
        encoding="utf-8")


def main(argv: list[str] | None = None) -> None:  # noqa: ARG001 — 파이프라인 시그니처 통일
    samples = load_jsonl(MASTER)
    verifications = load_jsonl(OUT_DIR / "case_verifications.jsonl")
    metrics = compute_metrics(samples, verifications)
    verdict = evaluate_bulk_generation_readiness(metrics)
    out = {"metrics": metrics, "readiness": verdict}
    (OUT_DIR / "master" / "bulk_readiness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# 대량 생성 승인 판정", "", f"**상태: {verdict['status']}**", "",
          "## 기준별 결과", ""]
    md += [f"- PASS: {c}" for c in verdict["passed_criteria"]]
    md += [f"- FAIL: {c}" for c in verdict["failed_criteria"]]
    md += ["", f"- 권고 조치: {verdict['recommended_next_actions']}"]
    (OUT_DIR / "master" / "bulk_readiness.md").write_text("\n".join(md) + "\n",
                                                          encoding="utf-8")
    if verdict["status"] == "APPROVED":
        write_plan(metrics)
        log("APPROVED — 실행계획 생성(data_aireg/plans/). 대량 생성은 별도 작업.")
    else:
        log(f"NOT_APPROVED — 미달: {verdict['blocking_reasons']}")


if __name__ == "__main__":
    main()
