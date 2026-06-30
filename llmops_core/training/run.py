"""학습 CLI — 컨테이너에서 LoRA/DoRA SFT·DPO 1회 실행.

학습 컨테이너(deploy/serving/Dockerfile.train) 안에서 호출한다:

    # SFT(Instruction)
    python -m llmops_core.training.run --method sft --train /data/train.jsonl \
        --output-dir /out/adapter --max-steps 30 [--dora]

    # DPO(Preference)
    python -m llmops_core.training.run --method dpo --train /data/pref.jsonl \
        --output-dir /out/adapter --max-steps 30 [--dora]

SFT 입력 JSONL(줄 단위): {"messages":[...]} 또는 {"text","response"}.
DPO 입력 JSONL(줄 단위): {"prompt","chosen","rejected"}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmops_core.dataset.build import to_sft_examples
from llmops_core.training.sft import (
    PeftSFTConfig,
    run_dpo_peft,
    run_grpo_peft,
    run_ppo_peft,
    run_sft_peft,
)


def _load_sft_rows(path: str) -> list[dict]:
    """JSONL → trl conversational 행 리스트. messages/{text,response} 혼용 허용."""
    rows: list[dict] = []
    labeled: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "messages" in obj:
            rows.append({"messages": obj["messages"]})
        elif "text" in obj and "response" in obj:
            labeled.append(obj)
    for ex in to_sft_examples(labeled):
        rows.append({"messages": [{"role": m.role, "content": m.content} for m in ex.messages]})
    return rows


def _load_pref_rows(path: str) -> list[dict]:
    """JSONL → {prompt,chosen,rejected} 행 리스트."""
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("prompt") and obj.get("chosen") and obj.get("rejected"):
            rows.append({"prompt": obj["prompt"], "chosen": obj["chosen"],
                         "rejected": obj["rejected"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="trl+peft LoRA/DoRA SFT·DPO")
    p.add_argument("--method", choices=["sft", "dpo", "grpo", "ppo"], default="sft")
    p.add_argument("--train", required=True, help="학습 JSONL 경로")
    p.add_argument(
        "--base-model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    )
    p.add_argument("--output-dir", default="outputs/adapter")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help=">0이면 epoch 무시(빠른 검증)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--dora", action="store_true", help="DoRA(weight-decomposed LoRA) 사용")
    p.add_argument("--qlora", action="store_true", help="QLoRA(4bit) — bitsandbytes 필요")
    args = p.parse_args()

    cfg = PeftSFTConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        max_seq_length=args.max_seq_length,
        use_dora=args.dora,
        load_in_4bit=args.qlora,
    )
    pet = "DoRA" if args.dora else "LoRA"
    if args.method == "sft":
        rows = _load_sft_rows(args.train)
        print(f"[run] SFT/{pet} · {len(rows)}건 · base={args.base_model}", flush=True)
        out = run_sft_peft(cfg, rows)
    elif args.method == "dpo":
        rows = _load_pref_rows(args.train)
        print(f"[run] DPO/{pet} · {len(rows)}건 · base={args.base_model}", flush=True)
        out = run_dpo_peft(cfg, rows)
    else:  # grpo/ppo — {prompt} 또는 {text}만 사용(보상 기반)
        rows = _load_pref_rows(args.train) or _load_sft_rows(args.train)
        prompts = [r.get("prompt") for r in rows if isinstance(r, dict) and r.get("prompt")]
        if not prompts:  # sft 포맷이면 user 메시지를 프롬프트로
            prompts = [m["content"] for r in rows for m in r.get("messages", [])
                       if m.get("role") == "user"]
        print(f"[run] {args.method.upper()}/{pet} · {len(prompts)}건 · base={args.base_model}",
              flush=True)
        out = run_grpo_peft(cfg, prompts) if args.method == "grpo" else run_ppo_peft(cfg, prompts)
    print(f"[run] 어댑터 저장 완료 → {out}", flush=True)


if __name__ == "__main__":
    main()
