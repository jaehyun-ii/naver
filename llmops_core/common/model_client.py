"""공용 모델 클라이언트 — 논리 모델명 → litellm Router 경유 텍스트 생성.

가드레일 분류 모델·벤치마크 추론 등이 공유한다. model_list.yaml의 논리명을 그대로 쓰므로
백엔드(vLLM/CLOVA/외부 API)는 게이트웨이가 흡수 — "모델 선택 = 논리명 지정"이 된다.
"""

from __future__ import annotations

from typing import Any

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError


class ModelClient:
    """litellm.Router 동기 래퍼. complete(model, messages) → 응답 텍스트."""

    def __init__(self, model_list: list[dict[str, Any]] | None = None) -> None:
        try:
            from litellm import Router
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("litellm", "gateway") from exc
        if model_list is None:
            from llmops_core.gateway.router import _load_model_list

            model_list = _load_model_list(get_settings().gateway.config_path)
        self._router = Router(model_list=model_list)

    def complete(self, model: str, messages: list[dict], *,
                 temperature: float = 0.0, max_tokens: int = 256) -> str:
        resp = self._router.completion(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return (raw.get("choices") or [{}])[0].get("message", {}).get("content") or ""


_client: ModelClient | None = None


def get_model_client() -> ModelClient:
    """프로세스 단일 클라이언트(논리명 라우팅 캐시)."""
    global _client
    if _client is None:
        _client = ModelClient()
    return _client
