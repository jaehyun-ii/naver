"""DAG 단계 2 — 파인튜닝(unsloth+trl). GPU 노드(workload=training)에서 실행.

학습 메트릭은 MLflow 콜백으로 tracking에 연결, 체크포인트는 S3에 저장.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B")
    p.add_argument("--tenant", default="shared")
    p.add_argument("--dataset", default="data/train.jsonl")
    args = p.parse_args(argv)

    from datasets import load_dataset  # trl/datasets

    from llmops_core.common.config import get_settings
    from llmops_core.training import SFTJobConfig, run_sft

    s = get_settings()
    output = f"s3://{s.bucket('models')}/{args.tenant}/sft"
    cfg = SFTJobConfig(base_model=args.base_model, output_dir=output)
    ds = load_dataset("json", data_files=args.dataset, split="train")

    # MLflow autolog 콜백 연결
    try:
        from transformers.integrations import MLflowCallback

        callback = MLflowCallback()
    except Exception:  # noqa: BLE001
        callback = None

    run_sft(cfg, ds, mlflow_callback=callback)
    print(f"finetune done -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
