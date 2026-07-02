"""litellm.Router 임베드 — 라우팅/폴백/로드밸런싱 로직만 라이브러리로 사용.

LiteLLM Proxy 제품(서버/UI/키관리)은 띄우지 않는다. 100+ 프로바이더 라우팅 코어만 가져와
우리 FastAPI 게이트웨이가 직접 호출한다. 모델 목록은 config/model_list.yaml에서 로드.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from llmops_core.common.config import get_settings
from llmops_core.common.errors import OptionalDependencyError
from llmops_core.common.schemas import ChatCompletionRequest, Usage

logger = logging.getLogger(__name__)


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
            import litellm
            from litellm import Router  # lazy: pip install -e ".[gateway]"
        except ImportError as exc:  # pragma: no cover
            raise OptionalDependencyError("litellm", "gateway") from exc

        if model_list is None:
            model_list = _load_model_list(get_settings().gateway.config_path)
        self._model_list = model_list

        # 응답 캐싱: litellm 네이티브(인메모리/redis/시맨틱)를 임베드. 설정으로 on/off.
        from llmops_core.gateway.cache import make_litellm_cache

        cache = make_litellm_cache()
        cache_responses = cache is not None
        if cache_responses:
            litellm.cache = cache
        self._router = Router(model_list=model_list, cache_responses=cache_responses)

    @property
    def logical_models(self) -> list[str]:
        return sorted({m["model_name"] for m in self._model_list})

    async def acompletion(
        self, req: ChatCompletionRequest
    ) -> tuple[dict[str, Any], Usage, bool]:
        """채팅 완성 호출. (raw 응답 dict, 정규화된 Usage, 캐시적중여부) 반환."""
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
        # litellm이 캐시 적중을 _hidden_params로 알려줌
        hidden = getattr(resp, "_hidden_params", None) or raw.get("_hidden_params") or {}
        cache_hit = bool(hidden.get("cache_hit"))
        u = raw.get("usage") or {}
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
        )
        return raw, usage, cache_hit

    async def astream(self, req: ChatCompletionRequest):
        """스트리밍 호출 — litellm의 async 스트림(청크 이터레이터)을 그대로 반환.

        각 청크는 OpenAI 호환 chunk(model_dump 가능). 마지막 청크의 usage(stream_options로
        요청 시)로 비용을 집계한다. 호출 측(app.py)이 SSE로 릴레이한다.
        """
        kwargs: dict[str, Any] = {
            "model": req.model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            "stream": True,
            # 스트림 종료 시 usage 청크를 받도록 요청(지원 백엔드에 한함; best-effort).
            "stream_options": {"include_usage": True},
            **req.extra,
        }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        return await self._router.acompletion(**kwargs)

    def estimate_cost(self, model: str, usage: Usage) -> float:
        """모델 비용 산정. 우선순위: 설정 pricing → litellm 가격표 → (둘 다 없으면) WARN 후 0.

        커스텀 self-host 모델(hcx-seed-*)은 litellm 가격표에 없어 예전엔 bare except가
        0.0을 삼켰다. 이제 gateway.pricing[model]을 먼저 참조하고, 없으면 litellm
        cost_per_token으로 폴백, 그래도 가격을 못 구하면 경고 로그를 남기고 0.0을 반환한다.
        """
        # 1) 설정 기반 self-host 단가 (USD/token) — 커스텀 모델의 단일 출처
        pricing = get_settings().gateway.pricing.get(model)
        if pricing:
            in_c = float(pricing.get("input", 0.0)) * usage.prompt_tokens
            out_c = float(pricing.get("output", 0.0)) * usage.completion_tokens
            return in_c + out_c

        # 2) litellm 네이티브 가격표 폴백 (model_list의 *_cost_per_token 포함)
        try:
            from litellm import cost_per_token

            prompt_c, completion_c = cost_per_token(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            total = float(prompt_c + completion_c)
            if total > 0.0:
                return total
        except Exception as exc:  # noqa: BLE001 — 가격 미상은 예외로 처리하되 삼키지 않고 경고
            logger.warning(
                "litellm 비용 산정 실패 model=%s: %s (gateway.pricing[%s] 설정 권장)",
                model, exc, model,
            )
            return 0.0

        # 3) 어느 소스에서도 가격을 못 구함 — 침묵 대신 경고
        logger.warning(
            "모델 '%s' 비용 산정 불가 (gateway.pricing 미정의·litellm 가격표 없음) → 0.0 처리. "
            "예산/과금 정확도를 위해 gateway.pricing 또는 model_list *_cost_per_token 설정 권장.",
            model,
        )
        return 0.0
