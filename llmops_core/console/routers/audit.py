"""감사로그 조회 — 권한 동작(키·승인·배포·토큰)의 기록 (admin 전용).

actor·action·target·시각을 시간 역순으로 제공. 컴플라이언스/사고대응의 감사 추적.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(prefix="/api/audit", tags=["audit"],
                   dependencies=[Depends(require_perm("*"))])


@router.get("")
def list_audit(limit: int = 100) -> list[dict]:
    return services().audit.list(limit=limit)
