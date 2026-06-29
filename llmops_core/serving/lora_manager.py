"""테넌트 LoRA 어댑터 수명주기 매니저 — 멀티테넌시 서빙측 핵심 (자체 구현).

베이스 SEED 모델 하나에 테넌트별 LoRA 어댑터를 핫스왑한다.
어댑터 가중치는 S3(prod-{domain}-models)에서 받아 로컬 캐시에 두고 vLLM에 등록한다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from llmops_core.common.storage import ObjectStore


@dataclass
class AdapterHandle:
    tenant_id: str
    name: str
    local_path: str
    lora_int_id: int  # vLLM LoRARequest용 정수 ID


class LoRAAdapterManager:
    """어댑터 로드/언로드/핫스왑 + 정수 ID 할당. 스레드세이프."""

    def __init__(self, cache_dir: str = "/tmp/llmops-loras", store: ObjectStore | None = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or ObjectStore()
        self._adapters: dict[str, AdapterHandle] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def ensure(self, tenant_id: str, name: str, s3_prefix: str) -> AdapterHandle:
        """S3에서 어댑터를 받아 로컬 캐시에 두고 핸들을 반환(이미 있으면 재사용)."""
        with self._lock:
            if name in self._adapters:
                return self._adapters[name]
            local = self.cache_dir / name
            local.mkdir(parents=True, exist_ok=True)
            # adapter_config.json + adapter_model.safetensors 다운로드
            for fname in ("adapter_config.json", "adapter_model.safetensors"):
                key = f"{s3_prefix.rstrip('/')}/{fname}"
                data = self.store.get_bytes("models", key)
                (local / fname).write_bytes(data)
            handle = AdapterHandle(tenant_id, name, str(local), self._next_id)
            self._next_id += 1
            self._adapters[name] = handle
            return handle

    def to_lora_request(self, handle: AdapterHandle):
        """vllm.lora.request.LoRARequest 생성 (lazy import)."""
        from vllm.lora.request import LoRARequest

        return LoRARequest(handle.name, handle.lora_int_id, handle.local_path)

    def unload(self, name: str) -> None:
        with self._lock:
            self._adapters.pop(name, None)
