"""서빙 안전 제어평면 — 가드레일 점검·설정 상태(콘솔 UI용).

게이트웨이가 실제 요청에 적용하는 가드레일을 콘솔에서 미리 점검하고, RAG 서빙·캐시·
가드레일의 현재 설정을 한눈에 본다. 점검은 viewer, 설정 조회는 viewer.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from llmops_core.common.config import get_settings
from llmops_core.console.security import require_perm
from llmops_core.gateway.guardrails import GuardrailEngine

router = APIRouter(prefix="/api/safety", tags=["safety"],
                   dependencies=[Depends(require_perm("read"))])


class CheckBody(BaseModel):
    input: str | None = None  # 사용자 입력(인젝션 점검)
    output: str | None = None  # 모델 출력(금칙어·PII 점검)


@router.get("/config")
def config() -> dict:
    """가드레일·RAG 서빙·캐시의 현재 설정 상태."""
    s = get_settings()
    return {
        "guardrails": {
            "enabled": s.guardrails.enabled,
            "block_on_injection": s.guardrails.block_on_injection,
            "block_on_banned": s.guardrails.block_on_banned,
            "mask_output_pii": s.guardrails.mask_output_pii,
            "banned_terms": s.guardrails.banned_terms,
        },
        "rag_serving": {"enabled": s.rag.serving_enabled, "top_k": s.rag.top_k,
                        "embedder": s.rag.embedder, "backend": s.rag.backend},
        "cache": {"enabled": s.cache.enabled, "backend": s.cache.backend, "ttl_s": s.cache.ttl_s},
    }


@router.get("/cache-stats")
def cache_stats() -> dict:
    """게이트웨이의 응답 캐시 히트율을 프록시 조회(콘솔↔게이트웨이는 별도 앱).

    게이트웨이 미가동/미도달이면 available=False로 graceful 반환(콘솔이 죽지 않음).
    """
    import httpx

    s = get_settings()
    url = s.gateway.base_url.rstrip("/") + "/admin/cache/stats"
    try:
        r = httpx.get(url, headers={"X-Master-Key": s.gateway.master_key}, timeout=2.0)
        r.raise_for_status()
        return {"available": True, **r.json()}
    except Exception as exc:  # noqa: BLE001 — 게이트웨이 미가동 등
        return {"available": False, "error": str(exc), "gateway": s.gateway.base_url}


@router.post("/guardrail/check")
def guardrail_check(body: CheckBody) -> dict:
    """입력/출력을 게이트웨이와 동일한 가드레일로 점검(차단 여부·플래그)."""
    eng = GuardrailEngine()
    out: dict = {}
    if body.input is not None:
        r = eng.check_input([{"role": "user", "content": body.input}])
        out["input"] = {"allowed": r.allowed, "reason": r.reason, "flags": r.flags}
    if body.output is not None:
        r = eng.check_output(body.output)
        out["output"] = {"allowed": r.allowed, "reason": r.reason,
                         "flags": r.flags, "text": r.text}
    return out
