"""Jaeger 통합 뷰 — LLM 트레이스·스팬을 메인 콘솔에서 조회(읽기전용 프록시).

Jaeger UI를 따로 띄우지 않고 Query API(/api/services, /api/traces)를 프록시. 미도달 시 graceful.
LLM 스팬의 지연·토큰·비용·에러를 콘솔에서 본다(telemetry 규약 속성).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from llmops_core.common.config import get_settings
from llmops_core.console.security import require_perm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/traces", tags=["traces"],
                   dependencies=[Depends(require_perm("read"))])


def _base() -> str:
    return get_settings().telemetry.jaeger_query_url.rstrip("/")


def _get(path: str, params: dict | None = None) -> dict:
    import httpx

    r = httpx.get(_base() + path, params=params, timeout=5.0)
    r.raise_for_status()
    return r.json()


@router.get("/services")
def services() -> dict:
    try:
        data = _get("/api/services")
        return {"available": True, "services": data.get("data", [])}
    except Exception as exc:  # noqa: BLE001
        # Jaeger 미도달을 "서비스 없음"으로 착각하지 않도록 로깅 + error 사유 노출.
        logger.warning("Jaeger services 조회 실패(%s): %s", _base(), exc, exc_info=True)
        return {"available": False, "error": str(exc), "jaeger": _base()}


def _summarize(trace: dict) -> dict:
    spans = trace.get("spans", [])
    starts = [s["startTime"] for s in spans] or [0]
    ends = [s["startTime"] + s.get("duration", 0) for s in spans] or [0]
    # 태그에서 LLM 규약 속성 추출(있으면)
    def tag(s, key):
        return next((t.get("value") for t in s.get("tags", []) if t.get("key") == key), None)
    root = spans[0] if spans else {}
    has_err = any(tag(s, "error") in (True, "true") for s in spans)
    total_tokens = next((tag(s, "gen_ai.usage.total_tokens") for s in spans
                         if tag(s, "gen_ai.usage.total_tokens") is not None), None)
    cost = next((tag(s, "llmops.cost.usd") for s in spans
                 if tag(s, "llmops.cost.usd") is not None), None)
    return {
        "traceID": trace.get("traceID"),
        "root": root.get("operationName", ""),
        "spans": len(spans),
        "duration_ms": round((max(ends) - min(starts)) / 1000, 1),
        "error": has_err,
        "total_tokens": total_tokens,
        "cost_usd": cost,
    }


@router.get("")
def traces(service: str, limit: int = 20) -> dict:
    try:
        data = _get("/api/traces", {"service": service, "limit": limit})
        items = [_summarize(t) for t in data.get("data", [])]
        items.sort(key=lambda x: x["duration_ms"], reverse=True)
        return {"available": True, "traces": items}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Jaeger traces 조회 실패(service=%s): %s", service, exc, exc_info=True)
        return {"available": False, "error": str(exc), "jaeger": _base()}


@router.get("/{trace_id}")
def trace_detail(trace_id: str) -> dict:
    try:
        data = _get(f"/api/traces/{trace_id}")
        traces_ = data.get("data", [])
        if not traces_:
            return {"available": True, "spans": []}
        spans = []
        for s in sorted(traces_[0].get("spans", []), key=lambda x: x["startTime"]):
            tags = {t["key"]: t.get("value") for t in s.get("tags", [])}
            spans.append({"op": s.get("operationName"),
                          "duration_ms": round(s.get("duration", 0) / 1000, 1),
                          "tags": {k: v for k, v in tags.items()
                                   if k.startswith(("gen_ai.", "llmops.")) or k == "error"}})
        return {"available": True, "spans": spans}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Jaeger trace 상세 조회 실패(trace_id=%s): %s", trace_id, exc,
                       exc_info=True)
        return {"available": False, "error": str(exc)}
