"""vLLM AsyncLLMEngine 임베드 — 엔진을 직접 제어 (OpenAI 표면은 게이트웨이가 제공).

vLLM Proxy/서버 대신 엔진 모듈만 가져와, 테넌트 LoRA 주입·텔레메트리 훅을 자체 코드가 보유한다.
실제 실행은 GPU 노드(workload=serving)에서. import는 lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmops_core.common.errors import OptionalDependencyError
from llmops_core.serving.lora_manager import AdapterHandle, LoRAAdapterManager


@dataclass
class ServingConfig:
    model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B"
    served_model_name: str = "hcx-seed-3b"
    quantization: str | None = "fp8"
    enable_lora: bool = True
    max_lora_rank: int = 32
    tensor_parallel_size: int = 1
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    extra_engine_args: dict = field(default_factory=dict)


class SeedServingEngine:
    """AsyncLLMEngine 래퍼 + 테넌트 LoRA 핫스왑."""

    def __init__(self, config: ServingConfig, lora_manager: LoRAAdapterManager | None = None):
        self.config = config
        self.lora = lora_manager or LoRAAdapterManager()
        self._engine = None

    def start(self) -> None:
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("vllm", "serving") from exc

        args = AsyncEngineArgs(
            model=self.config.model,
            quantization=self.config.quantization,
            enable_lora=self.config.enable_lora,
            max_lora_rank=self.config.max_lora_rank,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            **self.config.extra_engine_args,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)

    async def generate(
        self,
        prompt: str,
        request_id: str,
        *,
        adapter: AdapterHandle | None = None,
        sampling: dict | None = None,
    ):
        """단일 생성 스트림. adapter가 있으면 테넌트 LoRA를 주입한다."""
        if self._engine is None:
            raise RuntimeError("engine 미시작 — start() 호출 필요")
        from vllm import SamplingParams

        params = SamplingParams(**(sampling or {"temperature": 0.7, "max_tokens": 512}))
        lora_req = self.lora.to_lora_request(adapter) if adapter else None
        async for output in self._engine.generate(
            prompt, params, request_id, lora_request=lora_req
        ):
            yield output
