"""모델 목록 — config/model_list.yaml의 논리 모델명 (litellm 불필요)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llmops_core.console.security import require_perm
from llmops_core.console.services import load_logical_models

router = APIRouter(prefix="/api/models", tags=["models"], dependencies=[Depends(require_perm("read"))])


@router.get("")
def list_models() -> dict:
    return {"models": load_logical_models()}
