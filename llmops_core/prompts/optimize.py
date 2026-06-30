"""프롬프트 엔지니어링 체계 — 시스템 프롬프트 변형을 정량 비교·승격.

프롬프트를 '버전 자산'(GitPromptStore)으로 관리하는 데 더해, 변형들을 모델로 실제
평가해 가장 좋은 변형을 고른다(judge 불필요 reference 메트릭). 선택된 변형을 prod 라벨로
승격(promote)하면 코드 배포 없이 무중단 교체된다.

학습 컨테이너에서 실행(모델 추론 필요):
    python -m llmops_core.prompts.optimize \
        --cases /work/eval.jsonl --variants /work/variants.txt \
        --base <model> [--adapter /work/adapter] --name qa-system --promote
variants.txt: 한 줄=한 시스템 프롬프트 변형.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmops_core.evaluation.harness import run_reference_metrics
from llmops_core.evaluation.local_eval import _generate, _load_cases, _load_model


def evaluate_variant(model, tok, device, system: str, cases) -> dict:
    """system 프롬프트를 주입해 각 질문에 답 생성 → reference 메트릭.

    생성 경로는 모델 평가(local_eval._generate)와 동일하게 공유한다 — 프롬프트 변형
    비교와 모델 평가가 같은 추론 경로·메트릭을 쓰도록(평가=서빙 정합).
    """
    scored = [
        c.model_copy(update={"answer": _generate(model, tok, device, c.question, 64, system)})
        for c in cases
    ]
    return run_reference_metrics(scored)


def main() -> None:
    p = argparse.ArgumentParser(description="프롬프트 변형 평가·선택")
    p.add_argument("--cases", required=True)
    p.add_argument("--variants", required=True, help="줄 단위 시스템 프롬프트 변형")
    p.add_argument("--base", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    p.add_argument("--adapter", default=None)
    p.add_argument("--name", default="system-prompt", help="GitPromptStore 프롬프트명")
    p.add_argument("--promote", action="store_true", help="최우수 변형을 prod 라벨로 승격")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cases = _load_cases(args.cases)
    variants = [ln.strip() for ln in Path(args.variants).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    model, tok, device = _load_model(args.base, args.adapter)

    results = []
    for i, v in enumerate(variants):
        m = evaluate_variant(model, tok, device, v, cases)
        results.append({"variant": v, "metrics": m})
        print(f"[prompt] variant {i}: f1={m['reference_f1']:.4f} match={m['answer_match']:.2f} "
              f":: {v[:50]}", flush=True)

    best = max(results, key=lambda r: r["metrics"]["reference_f1"])
    payload = {"best": best, "all": results}

    if args.promote:
        from llmops_core.prompts import GitPromptStore

        store = GitPromptStore()
        pr = store.create_version(args.name, best["variant"])
        store.promote(args.name, pr.version, "prod")
        payload["promoted"] = {"name": args.name, "version": pr.version, "label": "prod"}
        print(f"[prompt] 최우수 변형 → {args.name} v{pr.version} @prod 승격", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("[prompt] BEST :: " + best["variant"][:80], flush=True)


if __name__ == "__main__":
    main()
