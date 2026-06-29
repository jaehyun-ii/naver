"""서빙 운영 — 카나리/블루그린 롤아웃 상태·승격·롤백 (제어평면).

배포 파이프라인이 카나리로 신모델을 소량 노출하면, 운영자가 지표 확인 후 승격/롤백한다.
상태 조회는 viewer 이상, 승격/롤백·가중 조정은 operator/admin.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.config import get_settings
from llmops_core.common.security import Principal
from llmops_core.console.security import require_perm
from llmops_core.serving import canary

router = APIRouter(prefix="/api/serving", tags=["serving"],
                   dependencies=[Depends(require_perm("read"))])


def _model_list() -> list:
    path = Path(get_settings().gateway.config_path)
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return data.get("model_list", [])


@router.get("/rollout/{name}")
def rollout(name: str) -> dict:
    """논리 모델명의 stable/canary weight 현황."""
    return canary.rollout_status(_model_list(), name)


@router.post("/{name}/promote")
def promote(name: str, principal: Principal = Depends(require_perm("pipeline:run"))) -> dict:
    """카나리를 stable로 승격(100%) — 이전 stable 정리."""
    from llmops_core.console.executor import RealExecutor
    from llmops_core.console.services import services

    try:
        out = RealExecutor().promote_canary(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"승격 실패: {exc}") from exc
    services().audit.record(principal.subject, "serving.promote", target=name)
    return out


@router.post("/{name}/rollback")
def rollback(name: str, principal: Principal = Depends(require_perm("pipeline:run"))) -> dict:
    """카나리 제거 — stable 100% 복귀."""
    from llmops_core.console.executor import RealExecutor
    from llmops_core.console.services import services

    try:
        out = RealExecutor().rollback_canary(name)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"롤백 실패: {exc}") from exc
    services().audit.record(principal.subject, "serving.rollback", target=name)
    return out
