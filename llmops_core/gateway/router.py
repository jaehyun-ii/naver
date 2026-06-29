"""litellm.Router 임베드 — 라우팅/폴백/로드밸런싱 로직만 라이브러리로 사용.

LiteLLM Proxy 제품(서버/UI/키관리)은 띄우지 않는다. 100+ 프로바이더 라우팅 코어만 가져와
우리 FastAPI 게이트웨이가 직접 호출한다. 모델 목록은 config/model_list.yaml에서 로드.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import ChatCompletionRequest, Usage


def _load_model_list(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return data.get("model_list", [])


class GatewayRouter:
    """litellm.Router 래퍼. 논리 모델명 → 백엔드 매핑을 보유."""

    def __init__(self, model_list: list[dict[str, Any]] | None = None) -> None:
        try:
            from litellm import Router  # lazy: pip install -e ".[gateway]"
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("litellm", "gateway") from exc

        if model_list is None:
            model_list = _load_model_list(get_settings().gateway.config_path)
        self._model_list = model_list
        self._router = Router(model_list=model_list)

    @property
    def logical_models(self) -> list[str]:
        return sorted({m["model_name"] for m in self._model_list})

    async def acompletion(self, req: ChatCompletionRequest) -> tuple[dict[str, Any], Usage]:
        """채팅 완성 호출. (raw 응답 dict, 정규화된 Usage) 반환."""
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            **req.extra,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens

        resp = await self._router.acompletion(**kwargs)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        u = raw.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return raw, usage

    def estimate_cost(self, model: str, usage: Usage) -> float:
        """litellm의 비용 계산기를 차용(없으면 0)."""
        try:
            from litellm import cost_per_token

            prompt_c, completion_c = cost_per_token(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            return float(prompt_c + completion_c)
        except Exception:  # noqa: BLE001
            return 0.0
