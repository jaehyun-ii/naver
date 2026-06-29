"""LoRA 병합 CLI — 학습 컨테이너(llmops/train)에서 어댑터를 베이스에 흡수.

    python -m llmops_core.registry.merge \
        --base naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
        --adapter /out/adapter --output-dir /out/merged

결과(safetensors)는 transformers hf_server가 어댑터 없이 바로 로드 가능.
"""

from __future__ import annotations

import argparse

from llmops_core.registry.convert import MergeConfig, merge_lora


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA merge → safetensors")
    p.add_argument("--base", required=True, help="베이스 모델 id/경로")
    p.add_argument("--adapter", required=True, help="학습 산출 LoRA 어댑터 경로")
    p.add_argument("--output-dir", required=True, help="병합 가중치 저장 위치")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    out = merge_lora(
        MergeConfig(
            base_model=args.base,
            adapter_path=args.adapter,
            output_dir=args.output_dir,
            dtype=args.dtype,
        )
    )
    print(f"[merge] 병합 완료 → {out}", flush=True)


if __name__ == "__main__":
    main()
