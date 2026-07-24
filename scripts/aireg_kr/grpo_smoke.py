"""GRPO smoke run — 실제 TRL·PEFT 환경에서 메타데이터 pass-through·정렬·보상 검증.

목표는 모델 품질이 아니라 메커니즘 검증이다:
(1) 추가 컬럼(gold_label·program_label·condition_results…)이 reward **kwargs로 전달
(2) num_generations>1에서 completion-메타데이터 정렬 유지
(3) 복수 reward(label_accuracy/program_label/format) 정상 합산·개별 로깅
(4) LoRA/QLoRA 최소 step 학습 + 그룹 내 reward variance 발생 여부 관측

운영 경로(run_grpo_peft)를 그대로 호출한다 — 별도 재구현 금지.
샘플은 Master 파생 grpo 파일에서 뽑는다: 판정 트랙(적합/부적합/판단불가) +
구조화 사례 트랙(경계값·예외 4변형, build_cases 실데이터). 실데이터가 없는
필수 유형만 합성 보충하고 synthetic=true로 표시해 보고한다.

GPU·패키지 미충족 시 성공으로 처리하지 않고 ENV_VALIDATION_PENDING으로 기록한다.

    python -m scripts.aireg_kr.grpo_smoke [--max-steps 2] [--num-generations 2] [--qlora]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import re
import sys
from pathlib import Path

from .answer_format import ANSWER_FORMAT_JUDGMENT, format_reward
from .common import OUT_DIR, log
from .labels import canonicalize_label
from .rewards import (_match_label, condition_coverage_reward,
                      extract_decision_label, intent_alignment_reward,
                      program_label_reward)

ART_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "grpo_smoke"

# §13 필수 구성 → (선별 조건, 합성 fallback 라벨)
REQUIRED_MIX = [
    ("APPLICABLE", lambda r: canonicalize_label(r.get("gold_label")) == "APPLICABLE"
     and not r.get("no_golden")),
    ("NOT_APPLICABLE", lambda r: canonicalize_label(r.get("gold_label")) == "NOT_APPLICABLE"),
    ("INSUFFICIENT_INFORMATION",
     lambda r: canonicalize_label(r.get("gold_label")) == "INSUFFICIENT_INFORMATION"),
    ("EXCEPTION_APPLIES", lambda r: r.get("kind") == "exception_all_met"),
    ("EXCEPTION_PARTIALLY_MET", lambda r: r.get("kind") == "exception_partially_met"),
    ("NUMERIC_BOUNDARY", lambda r: r.get("kind") == "numeric_boundary"),
]


def collect_environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "trl", "peft", "datasets",
                 "accelerate", "bitsandbytes"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "?")
        except ImportError:
            env[name] = None
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if env["cuda_available"]:
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["cuda_available"] = False
    return env


def env_ready(env: dict) -> list[str]:
    missing = [k for k in ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
               if not env.get(k)]
    if not env.get("cuda_available"):
        missing.append("cuda")
    return missing


def _synthetic(mix_name: str, i: int) -> dict:
    gold = {"APPLICABLE": "APPLICABLE", "NOT_APPLICABLE": "NOT_APPLICABLE",
            "INSUFFICIENT_INFORMATION": "INSUFFICIENT_INFORMATION"}.get(
        mix_name, "NOT_APPLICABLE")
    return {
        "prompt": (f"규정: 총톤수 10,000톤 이상인 유조선은 보조조타장치를 비치하여야 한다.\n"
                   f"사례({mix_name} 검증 {i}): 유조선, 12,000 GT.\n질문: 적용 여부를 판단하십시오."),
        "gold_label": gold, "program_label": gold, "evaluable": False,
        "kind": f"synthetic_{i}", "task_type": "compliance_judgment",
        "answer_format": ANSWER_FORMAT_JUDGMENT, "rule_id": "SYNTHETIC",
        "sample_id": f"synthetic::{i}", "canonical_rule_group_id": "crg_synthetic",
        "condition_results": [], "missing_fields": [], "synthetic": True,
    }


def select_samples(limit: int = 10) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for split in ("train", "validation", "test"):
        p = OUT_DIR / "master" / f"grpo_{split}.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
    picked, coverage = [], {}
    for name, pred in REQUIRED_MIX:
        row = next((r for r in rows if pred(r) and r["sample_id"] not in
                    {p["sample_id"] for p in picked}), None)
        if row:
            picked.append(row)
            coverage[name] = "real"
        elif name in ("APPLICABLE", "NOT_APPLICABLE", "INSUFFICIENT_INFORMATION"):
            picked.append(_synthetic(name, len(picked)))
            coverage[name] = "synthetic"
        else:
            coverage[name] = "MISSING"   # 예외·경계값은 합성 대체 금지(§18)
    return picked[:limit], coverage


def make_rewards(debug_path: Path):
    """metadata_alignment(디버그) + label_accuracy + program_label + format."""
    f = debug_path.open("w", encoding="utf-8")

    def metadata_alignment_reward(prompts, completions, gold_label=None,
                                  sample_id=None, canonical_rule_group_id=None,
                                  **kwargs):
        assert gold_label is not None, "gold_label 미전달 — pass-through 실패"
        assert sample_id is not None, "sample_id 미전달"
        assert canonical_rule_group_id is not None, "canonical_rule_group_id 미전달"
        assert len(completions) == len(gold_label) == len(sample_id) \
            == len(canonical_rule_group_id), "메타데이터-completion 길이 불일치"
        for i in range(len(completions)):
            f.write(json.dumps({"call": "metadata_alignment", "i": i,
                                "sample_id": sample_id[i], "gold_label": gold_label[i],
                                "prompt_head": (prompts[i][-1]["content"]
                                                if isinstance(prompts[i], list)
                                                else str(prompts[i]))[:60]},
                               ensure_ascii=False) + "\n")
        f.flush()
        return [0.0] * len(completions)

    def label_accuracy_reward(prompts, completions, gold_label=None,
                              sample_id=None, condition_results=None, **kwargs):
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else (c[-1].get("content", "") if c else "")
            pred = extract_decision_label(text)
            gold = gold_label[i] if gold_label else None
            score = 1.0 if _match_label(pred, gold) else 0.0
            f.write(json.dumps({"call": "label_accuracy", "i": i,
                                "sample_id": sample_id[i] if sample_id else None,
                                "pred": pred, "gold": gold, "reward": score,
                                "n_conds": len((condition_results[i] if condition_results else None) or []),
                                "completion_head": text[:120]},
                               ensure_ascii=False) + "\n")
            out.append(score)
        f.flush()
        return out

    return [metadata_alignment_reward, label_accuracy_reward,
            program_label_reward, intent_alignment_reward,
            condition_coverage_reward, format_reward], f


def summarize_rewards(trainer_log: Path) -> dict:
    """trainer 로그에서 reward별 평균·표준편차, variance 발생 그룹 신호 추출."""
    stats: dict = {"steps": []}
    if not trainer_log.exists():
        return stats
    for line in trainer_log.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if "reward" not in rec:
            continue
        stats["steps"].append({
            "step": rec.get("step"),
            "reward_mean": rec.get("reward"), "reward_std": rec.get("reward_std"),
            "frac_reward_zero_std": rec.get("frac_reward_zero_std"),
            "per_reward": {k.split("/")[1]: {"mean": rec.get(k), "std": rec.get(
                k.replace("/mean", "/std"))}
                for k in rec if k.startswith("rewards/") and k.endswith("/mean")},
        })
    nz = [s for s in stats["steps"]
          if s.get("frac_reward_zero_std") is not None and s["frac_reward_zero_std"] < 1.0]
    stats["steps_with_nonzero_variance_groups"] = len(nz)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="GRPO smoke run (실 TRL 환경)")
    ap.add_argument("--model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--qlora", action="store_true")
    args = ap.parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    env = collect_environment()
    (ART_DIR / "environment.json").write_text(
        json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = env_ready(env)
    summary: dict = {"success": False, "steps_completed": 0,
                     "num_generations": args.num_generations,
                     "metadata_alignment": "NOT_RUN",
                     "reward_functions": ["metadata_alignment_reward", "label_accuracy_reward",
                                          "program_label_reward", "intent_alignment_reward",
                                          "condition_coverage_reward", "format_reward"],
                     "peft_method": "qlora" if args.qlora else "lora", "errors": []}
    if missing:
        summary["status"] = "ENV_VALIDATION_PENDING"
        summary["errors"] = [f"환경 미충족: {missing}"]
        (ART_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"환경 미충족({missing}) — ENV_VALIDATION_PENDING")
        return 1

    samples, coverage = select_samples()
    summary["sample_coverage"] = coverage
    summary["real_exception_samples"] = coverage.get("EXCEPTION_PARTIALLY_MET") == "real"
    summary["synthetic_fallback_used"] = any(v == "synthetic" for v in coverage.values())
    with (ART_DIR / "input_samples.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"smoke 샘플 {len(samples)}건 · 커버리지 {coverage}")
    if coverage.get("EXCEPTION_PARTIALLY_MET") == "MISSING":
        summary["status"] = "CODE_COMPLETE_REAL_EXCEPTION_DATA_PENDING"
        summary["errors"].append("실제 exception_partially_met 샘플 부재 — 합성 대체 금지")
        (ART_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    from transformers import TrainerCallback

    class _LogCapture(TrainerCallback):
        def __init__(self, path: Path):
            self.f = path.open("w", encoding="utf-8")

        def on_log(self, _args, state, control, logs=None, **kw):
            if logs:
                self.f.write(json.dumps({"step": state.global_step, **logs},
                                        ensure_ascii=False) + "\n")
                self.f.flush()

    from llmops_core.training.sft import PeftSFTConfig, run_grpo_peft

    cfg = PeftSFTConfig(
        base_model=args.model, output_dir=str(ART_DIR / "adapter"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1, max_seq_length=1024,
        load_in_4bit=args.qlora,
        extra_sft_args={"num_generations": args.num_generations},
    )
    # instruct 모델이 chat template로 응답하도록 conversational 형태 사용 —
    # TRL 표준 지원이며 메타 컬럼 pass-through에는 영향 없음
    samples = [{**s, "prompt": [{"role": "user", "content": s["prompt"]}]}
               for s in samples]
    (ART_DIR / "config.json").write_text(json.dumps({
        "model": args.model, "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "peft": summary["peft_method"], "n_samples": len(samples),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    rewards, debug_f = make_rewards(ART_DIR / "reward_debug.jsonl")
    log_cb = _LogCapture(ART_DIR / "trainer_log.jsonl")
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            run_grpo_peft(cfg, samples, reward_funcs=rewards, mlflow_callback=log_cb)
        summary["success"] = True
        summary["steps_completed"] = args.max_steps
        summary["metadata_alignment"] = "PASS"
        summary["status"] = "COMPLETE"
    except AssertionError as e:
        summary["metadata_alignment"] = "FAIL"
        summary["status"] = "FAILED"
        summary["errors"].append(f"메타데이터 정렬 실패: {e}")
    except Exception as e:  # noqa: BLE001
        summary["status"] = "FAILED"
        summary["errors"].append(f"{type(e).__name__}: {e}")
    finally:
        debug_f.close()
        log_cb.f.close()
        out_text = captured.getvalue()
        sys.stdout.write(out_text)
        m = re.search(r"adapter_delta_norm[\"']?\s*[:=]\s*([\d.eE+-]+)", out_text)
        summary["adapter_delta_norm"] = float(m.group(1)) if m else None

    if summary["success"]:
        by_sample = {s["sample_id"]: s["prompt"][-1]["content"][:60]
                     if isinstance(s["prompt"], list) else s["prompt"][:60]
                     for s in samples}
        for line in (ART_DIR / "reward_debug.jsonl").read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["call"] != "metadata_alignment":
                continue
            if by_sample.get(rec["sample_id"], "")[:60] != rec["prompt_head"]:
                summary["metadata_alignment"] = "FAIL"
                summary["success"] = False
                summary["errors"].append(f"정렬 불일치: {rec['sample_id']}")
                break
        summary["reward_stats"] = summarize_rewards(ART_DIR / "trainer_log.jsonl")

    (ART_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"summary → {ART_DIR / 'summary.json'} : {summary['status']}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
