"""HPO CLI — Optuna로 학습 하이퍼파라미터 탐색 (학습 컨테이너 내 실행).

각 trial = 샘플 파라미터로 짧은 SFT → reference 평가 점수. 최적 파라미터를 출력/JSON 저장.

    python -m llmops_core.tuning.run \
        --train /work/train.jsonl --eval /work/eval.jsonl \
        --base naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
        --trials 6 --steps 15 --out /work/hpo.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from llmops_core.evaluation.harness import run_reference_metrics
from llmops_core.evaluation.local_eval import _generate, _load_cases, _load_model
from llmops_core.training.run import _load_sft_rows
from llmops_core.training.sft import PeftSFTConfig, run_sft_peft
from llmops_core.tuning.optuna_study import HPOConfig, optimize


def _build_objective(base: str, train_rows: list[dict], eval_cases, steps: int):
    def objective(trial) -> float:
        lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
        lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
        lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
        with tempfile.TemporaryDirectory() as td:
            cfg = PeftSFTConfig(
                base_model=base, output_dir=td, max_steps=steps,
                per_device_train_batch_size=1, gradient_accumulation_steps=1,
                learning_rate=lr, lora_r=lora_r, lora_alpha=lora_alpha,
            )
            run_sft_peft(cfg, train_rows)
            model, tok, device = _load_model(base, td)
            scored = [c.model_copy(update={"answer": _generate(model, tok, device, c.question, 64)})
                      for c in eval_cases]
            metrics = run_reference_metrics(scored)
            del model
        score = metrics["reference_f1"]
        print(f"[hpo] trial {trial.number}: lr={lr:.2e} r={lora_r} a={lora_alpha} "
              f"→ f1={score:.4f}", flush=True)
        return score

    return objective


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna HPO for SFT")
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--base", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    train_rows = _load_sft_rows(args.train)
    eval_cases = _load_cases(args.eval)
    print(f"[hpo] train={len(train_rows)} eval={len(eval_cases)} trials={args.trials}", flush=True)

    objective = _build_objective(args.base, train_rows, eval_cases, args.steps)
    best_params, best_value = optimize(HPOConfig(n_trials=args.trials), objective)

    payload = {"best_params": best_params, "best_value": best_value}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("[hpo] BEST " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
