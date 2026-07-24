"""HPO CLI — Optuna로 학습 하이퍼파라미터 탐색 (SFT · DPO · GRPO).

각 trial = 샘플 파라미터로 짧은 학습 → **매소드별 평가 점수(최대화)**:
  sft  : reference_f1         · train {messages|text/response}, eval {question,expected}
  dpo  : preference_accuracy  · train/eval {prompt,chosen,rejected} (+ β 탐색)
  grpo : mean reward          · train/eval {prompt}               (+ num_generations 탐색)

    python -m llmops_core.tuning.run --method dpo \
        --train pref.jsonl --eval pref_eval.jsonl --trials 8 --steps 20 --out hpo.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from llmops_core.evaluation.harness import run_reference_metrics
from llmops_core.evaluation.local_eval import (
    _generate, _load_cases, _load_model, run_preference_eval,
)
from llmops_core.training.run import _load_sft_rows
from llmops_core.training.sft import (
    PeftSFTConfig, run_dpo_peft, run_grpo_peft, run_sft_peft,
)
from llmops_core.tuning.optuna_study import HPOConfig, optimize


def _load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_pref_rows(path: str) -> list[dict]:
    """{prompt, chosen, rejected} 선호 행."""
    out = []
    for o in _load_jsonl(path):
        if o.get("prompt") and o.get("chosen") is not None and o.get("rejected") is not None:
            out.append({"prompt": o["prompt"], "chosen": o["chosen"], "rejected": o["rejected"]})
    return out


def _load_prompt_rows(path: str) -> list[dict]:
    """{prompt} 프롬프트 행(GRPO — 정답 불필요)."""
    out = []
    for o in _load_jsonl(path):
        p = o.get("prompt") or o.get("question") or o.get("text")
        if p:
            out.append({"prompt": p})
    return out


def _length_reward(text: str) -> float:
    """GRPO 데모 보상(학습 기본 보상과 동일한 길이 휴리스틱). 도메인 보상으로 교체 가능."""
    n = len((text or "").strip())
    return 1.0 if 20 <= n <= 400 else max(0.0, 1.0 - abs(n - 200) / 400)


def _build_objective(method: str, base: str, train_rows: list, eval_rows: list, steps: int):
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
            if method == "dpo":
                beta = trial.suggest_float("beta", 0.05, 0.5, log=True)
                cfg = replace(cfg, dpo_beta=beta)
                run_dpo_peft(cfg, train_rows)
                model, tok, device = _load_model(base, td)
                score = run_preference_eval(model, tok, device, eval_rows)["preference_accuracy"]
                extra = f"beta={beta:.3f} pref_acc={score:.4f}"
            elif method == "grpo":
                ng = trial.suggest_categorical("num_generations", [2, 4, 8])
                cfg = replace(cfg, extra_sft_args={"num_generations": ng})
                run_grpo_peft(cfg, train_rows)
                model, tok, device = _load_model(base, td)
                rewards = [_length_reward(_generate(model, tok, device, r["prompt"], 128))
                           for r in eval_rows]
                score = round(sum(rewards) / max(len(rewards), 1), 4)
                extra = f"num_gen={ng} reward={score:.4f}"
            else:  # sft
                run_sft_peft(cfg, train_rows)
                model, tok, device = _load_model(base, td)
                scored = [c.model_copy(update={"answer": _generate(model, tok, device, c.question, 64)})
                          for c in eval_rows]
                score = run_reference_metrics(scored)["reference_f1"]
                extra = f"f1={score:.4f}"
            del model
        print(f"[hpo:{method}] trial {trial.number}: lr={lr:.2e} r={lora_r} a={lora_alpha} {extra}",
              flush=True)
        return score

    return objective


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna HPO for SFT/DPO/GRPO")
    p.add_argument("--method", choices=["sft", "dpo", "grpo"], default="sft")
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--base", default="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.method == "dpo":
        train_rows, eval_rows = _load_pref_rows(args.train), _load_pref_rows(args.eval)
    elif args.method == "grpo":
        train_rows, eval_rows = _load_prompt_rows(args.train), _load_prompt_rows(args.eval)
    else:
        train_rows, eval_rows = _load_sft_rows(args.train), _load_cases(args.eval)
    print(f"[hpo:{args.method}] train={len(train_rows)} eval={len(eval_rows)} "
          f"trials={args.trials}", flush=True)

    objective = _build_objective(args.method, args.base, train_rows, eval_rows, args.steps)
    best_params, best_value, trials = optimize(HPOConfig(n_trials=args.trials), objective)

    payload = {"method": args.method, "best_params": best_params,
               "best_value": best_value, "trials": trials}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("[hpo] BEST " + json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
