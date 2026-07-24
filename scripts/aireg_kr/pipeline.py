"""AIReg-KR 학습 데이터 생성 파이프라인 — 단일 진입점.

    python -m scripts.aireg_kr list                  # 스테이지 목록
    python -m scripts.aireg_kr status                # 산출물 현황(행 수)
    python -m scripts.aireg_kr run                   # 전체 실행(기본 스테이지)
    python -m scripts.aireg_kr run --stages suite,master
    python -m scripts.aireg_kr run --from verify     # verify부터 끝까지
    python -m scripts.aireg_kr run --dry-run         # 실행 계획만 출력

스테이지 순서(기본):
    cards → judgment → verify → cases → suite → suite_verify → benchmark → master → readiness
(training은 프로토타입 RAFT 조립 — master가 정식 산출이라 기본 제외, --stages로 명시 실행)

개별 스크립트는 여전히 단독 실행 가능하다(python -m scripts.aireg_kr.run_all ...).
이 모듈은 각 스테이지의 main(argv)를 순서대로 호출할 뿐이며, 모든 스테이지가
id 기반 resume이므로 파이프라인 재실행은 안전(멱등)하다.
"""
from __future__ import annotations

import argparse
import importlib
import os
import time

from .common import OUT_DIR, log


def _sel(a: argparse.Namespace) -> list[str]:
    """조항 선정 공통 플래그(run_all 계열)."""
    v = ["--publisher", a.publisher]
    if a.doc_glob:
        v += ["--doc-glob", a.doc_glob]
    if a.n_rules:
        v += ["--n-rules", str(a.n_rules)]
    elif a.mode == "per-doc":
        v += ["--n-rules", "0"]  # per-doc 기본은 전체 문서(run_all 기본 30 절단 방지)
    return v


def _limit(a: argparse.Namespace) -> list[str]:
    return ["--limit", str(a.limit)] if a.limit else []


def _judgment_argv(a: argparse.Namespace) -> list[str]:
    mode = {"oneshot": ["--oneshot"], "per-doc": ["--per-doc"], "pilot": []}[a.mode]
    v = mode + _sel(a)
    if a.mode == "per-doc" and a.publishers:
        v += ["--publishers", a.publishers]
    return v


# (이름, 설명, 모듈, argv 빌더). 순서가 실행 순서다.
STAGES: list[tuple[str, str, str, callable]] = [
    ("cards", "Rule Card 정규화 — 스위트 QA·케이스 트랙의 입력",
     "run_all", lambda a: ["--cards-only"] + _sel(a)
     + (["--per-doc"] if a.mode == "per-doc" else [])),
    ("judgment", "판정 트랙 생성(발췌 합성) — mode: oneshot|per-doc|pilot",
     "run_all", _judgment_argv),
    ("verify", "독립 검증자 블라인드 재판정 → verdicts.jsonl",
     "verify", _limit),
    ("cases", "결정적 경계값·예외 케이스(구조화→생성→검증)",
     "build_cases", lambda a: ["--structure-exceptions", "--build", "--verify"] + _limit(a)),
    ("suite", "스위트 QA 통합 생성 — 조당 1건, 조 특성 라우팅(스펙·적용성·상호참조·"
     "정의결합·우선규정·단위환산·표참조·조항호목 + 선급비교)",
     "build_suite_qa", lambda a: ["--publisher", a.publisher,
                                  "--n-compare", str(a.n_compare)] + _limit(a)
     + (["--doc-glob", a.doc_glob] if a.doc_glob else [])
     + (["--verify-loop", "--retries", str(a.retries)] if a.verify_loop else [])),
    ("suite_verify", "스위트 QA 블라인드 검증(용어 대응·전제 일치) → suite_verdicts.jsonl",
     "verify_suite", _limit),
    ("benchmark", "평가셋 조립(결정적) → benchmark_qa.jsonl",
     "build_qa", lambda a: []),
    ("master", "마스터 조립 + canonical split → master/{sft,dpo,grpo}_*.jsonl",
     "build_master", lambda a: ["--seed", str(a.seed), "--label-policy", a.label_policy]
     + (["--strict"] if a.strict else [])),
    ("readiness", "대량 생성 승인 게이트 → bulk_readiness",
     "readiness", lambda a: []),
    ("training", "[기본 제외] 프로토타입 RAFT 조립 → training/{sft,preference}.jsonl",
     "build_training", lambda a: []),
]
DEFAULT_SKIP = {"training"}
NAMES = [s[0] for s in STAGES]

# 산출물 현황(status) — 파일: 라벨
ARTIFACTS = [
    ("rule_cards.jsonl", "규칙 카드"), ("excerpts.jsonl", "발췌(판정)"),
    ("annotations.jsonl", "어노테이션"), ("verdicts.jsonl", "검증 verdict"),
    ("exception_structs.jsonl", "예외 구조화"), ("cases.jsonl", "구조화 케이스"),
    ("case_verifications.jsonl", "케이스 검증"),
    ("spec_qa.jsonl", "스펙 QA"), ("applicability_qa.jsonl", "적용성 QA"),
    ("crossref_qa.jsonl", "상호참조 QA"), ("compare_qa.jsonl", "선급 비교 QA"),
    ("def_link_qa.jsonl", "정의 결합 QA"), ("precedence_qa.jsonl", "우선규정 QA"),
    ("unit_convert_qa.jsonl", "단위환산 QA"), ("table_lookup_qa.jsonl", "표 참조 QA"),
    ("hierarchy_qa.jsonl", "조항호목 QA"), ("suite_routing.jsonl", "트랙 라우팅"),
    ("suite_verdicts.jsonl", "스위트 검증"),
    ("benchmark_qa.jsonl", "벤치마크(평가)"),
    ("master/master_dataset.jsonl", "마스터"),
    ("training/sft.jsonl", "RAFT SFT(프로토)"), ("training/preference.jsonl", "RAFT DPO(프로토)"),
] + [(f"master/{k}_{s}.jsonl", f"master {k.upper()} {s}")
     for k in ("sft", "dpo", "grpo") for s in ("train", "validation", "test")]


def _count_lines(name: str) -> int | None:
    p = OUT_DIR / name
    if not p.exists():
        return None
    with p.open("rb") as f:
        return sum(1 for l in f if l.strip())


def cmd_list() -> None:
    for name, desc, mod, _ in STAGES:
        flag = " (기본 제외)" if name in DEFAULT_SKIP else ""
        print(f"  {name:<10} {desc}{flag}  [scripts.aireg_kr.{mod}]")


def cmd_status() -> None:
    print(f"산출물 디렉터리: {OUT_DIR}")
    for name, label in ARTIFACTS:
        n = _count_lines(name)
        print(f"  {label:<16} {name:<34} {'-' if n is None else f'{n:,}행'}")


def _pick_stages(a: argparse.Namespace) -> list[tuple]:
    if a.stages:
        req = [s.strip() for s in a.stages.split(",")]
        bad = [s for s in req if s not in NAMES]
        if bad:
            raise SystemExit(f"알 수 없는 스테이지: {bad} (가능: {NAMES})")
        return [s for s in STAGES if s[0] in req]
    names = NAMES[:]
    if a.from_stage:
        names = names[names.index(a.from_stage):]
    if a.until:
        names = names[:names.index(a.until) + 1]
    skip = DEFAULT_SKIP | {s.strip() for s in (a.skip or "").split(",") if s.strip()}
    return [s for s in STAGES if s[0] in names and s[0] not in skip]


def cmd_run(a: argparse.Namespace) -> None:
    if a.workers:  # 개별 스테이지는 AIREG_WORKERS를 기본값으로 읽는다
        os.environ["AIREG_WORKERS"] = str(a.workers)
    stages = _pick_stages(a)
    plan = ", ".join(s[0] for s in stages)
    log(f"파이프라인 실행 계획: {plan} (mode={a.mode}, publisher={a.publisher})")
    for name, _desc, mod, argv_fn in stages:
        argv = [str(x) for x in argv_fn(a)]
        if a.dry_run:
            log(f"[dry-run] {name}: python -m scripts.aireg_kr.{mod} {' '.join(argv)}")
            continue
        log(f"━━ 스테이지 {name} 시작 ━━ ({mod} {' '.join(argv)})")
        t0 = time.time()
        importlib.import_module(f"{__package__}.{mod}").main(argv)
        log(f"━━ 스테이지 {name} 완료 ({time.time() - t0:.0f}s) ━━")
    if not a.dry_run:
        log("파이프라인 완료")
        cmd_status()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.aireg_kr",
        description="AIReg-KR 학습 데이터 생성 파이프라인(단일 진입점)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="스테이지 목록")
    sub.add_parser("status", help="산출물 현황")
    r = sub.add_parser("run", help="파이프라인 실행")
    r.add_argument("--stages", help="실행할 스테이지(쉼표 구분, 순서 무시하고 정의 순서로 실행)")
    r.add_argument("--from", dest="from_stage", choices=NAMES, help="이 스테이지부터")
    r.add_argument("--until", choices=NAMES, help="이 스테이지까지")
    r.add_argument("--skip", help="건너뛸 스테이지(쉼표 구분)")
    r.add_argument("--mode", choices=["oneshot", "per-doc", "pilot"], default="oneshot",
                   help="판정 트랙 모드(기본 oneshot — 현 운영 트랙)")
    r.add_argument("--publisher", default="KR")
    r.add_argument("--publishers", help="per-doc 모드 대상 발행처(쉼표 구분)")
    r.add_argument("--doc-glob", default=None)
    r.add_argument("--n-rules", type=int, default=0, help="대상 조 수(0=스크립트 기본)")
    r.add_argument("--n-compare", type=int, default=0, help="suite: 선급 비교 문항 수(0=생략)")
    r.add_argument("--verify-loop", dest="verify_loop", action="store_true",
                   default=True, help="suite: 생성 직후 검증 결합(기본 켜짐)")
    r.add_argument("--no-verify-loop", dest="verify_loop", action="store_false",
                   help="suite: 검증 루프 없이 생성만(구 방식 — suite_verify 별도 실행)")
    r.add_argument("--retries", type=int, default=2, help="suite: REJECT 재생성 횟수")
    r.add_argument("--limit", type=int, default=0, help="verify/cases/suite/suite_verify 대상 수 제한")
    r.add_argument("--workers", type=int, default=0, help="LLM 병렬 워커(AIREG_WORKERS)")
    r.add_argument("--seed", type=int, default=42, help="master split seed")
    r.add_argument("--label-policy", default="report-only")
    r.add_argument("--strict", action="store_true", help="master: 라벨 불일치 시 실패")
    r.add_argument("--dry-run", action="store_true", help="스테이지별 실행 계획만 출력")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        cmd_list()
    elif args.cmd == "status":
        cmd_status()
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
