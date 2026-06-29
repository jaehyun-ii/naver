"""L6 서빙 포맷 변환 — LoRA 병합(기본) + ONNX export(선택).

디코더 LLM은 vLLM이 safetensors를 직접 로드하므로 변환의 1급 작업은
**LoRA 어댑터를 베이스 가중치에 병합(merge)** 하는 것이다(공통 서빙 슬롯/배포 단순화).
ONNX는 특정 런타임이 필요할 때만(선택). import는 lazy(GPU/빌드 의존성).
"""

from __future__ import annotations

from dataclasses import dataclass

from llmops_core.common.errors import OptionalDependencyError


@dataclass
class MergeConfig:
    base_model: str
    adapter_path: str  # 학습 산출 LoRA 어댑터(로컬 경로 또는 S3에서 받은 경로)
    output_dir: str  # 병합 가중치 저장 위치(safetensors)
    dtype: str = "bfloat16"
    max_shard_size: str = "5GB"


def merge_lora(cfg: MergeConfig) -> str:
    """베이스 + LoRA → 병합 가중치(safetensors)로 저장. output_dir 반환.

    peft의 merge_and_unload로 어댑터를 본체에 흡수해 일반 HF 체크포인트를 만든다.
    결과는 vLLM/transformers가 어댑터 없이 바로 로드 가능.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("transformers/peft", "training") from exc

    dtype = getattr(torch, cfg.dtype)
    base = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, cfg.adapter_path)
    merged = model.merge_and_unload()  # 어댑터를 베이스에 병합
    merged.save_pretrained(
        cfg.output_dir, safe_serialization=True, max_shard_size=cfg.max_shard_size
    )
    AutoTokenizer.from_pretrained(cfg.base_model).save_pretrained(cfg.output_dir)
    return cfg.output_dir


def export_onnx(model_dir: str, output_dir: str) -> str:
    """병합 모델 → ONNX export(선택, optimum). 특정 런타임용. output_dir 반환."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("optimum[onnxruntime]", "serving") from exc

    model = ORTModelForCausalLM.from_pretrained(model_dir, export=True)
    model.save_pretrained(output_dir)
    return output_dir
