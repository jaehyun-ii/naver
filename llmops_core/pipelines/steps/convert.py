"""DAG 단계 4b — L6 서빙 포맷 변환: LoRA 어댑터를 베이스에 병합(safetensors).

승인·등록 후, 배포 전에 실행. 병합 가중치는 어댑터 없이 vLLM/transformers가 바로 로드 가능.
GPU 노드(workload=training)에서 실행(가중치 로드 메모리 필요). import는 lazy.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B")
    p.add_argument("--adapter-path", required=True)  # 학습 산출 LoRA(로컬/마운트 경로)
    p.add_argument("--output-dir", required=True)  # 병합 가중치 저장 위치
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--onnx", action="store_true")  # 추가 ONNX export(선택)
    args = p.parse_args(argv)

    from llmops_core.registry import MergeConfig, merge_lora

    out = merge_lora(
        MergeConfig(
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            output_dir=args.output_dir,
            dtype=args.dtype,
        )
    )
    print(f"LoRA merge 완료 -> {out}")

    if args.onnx:
        from llmops_core.registry import export_onnx

        onnx_dir = export_onnx(out, out.rstrip("/") + "-onnx")
        print(f"ONNX export 완료 -> {onnx_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
