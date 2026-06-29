"""릴리스 승인 — Release Gateway 2차 거버넌스 게이트 UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from llmops_core.common.errors import ReleaseRejected
from llmops_core.common.schemas import EvalResult, ReleaseRequest, ReleaseStatus
from llmops_core.common.security import Principal
from llmops_core.console.schemas import CreateReleaseBody, DecisionBody
from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(
    prefix="/api/releases", tags=["releases"],
    dependencies=[Depends(require_perm("releases:read"))],
)


@router.get("")
def list_releases(status: ReleaseStatus | None = None) -> list[ReleaseRequest]:
    return services().releases.store.list(status=status)


@router.post("")
def create_release(body: CreateReleaseBody) -> ReleaseRequest:
    result = EvalResult(
        suite=body.suite or "manual",
        model_ref=body.model_ref,
        metrics=body.metrics,
        passed=body.passed,
        data_version=body.data_version,
    )
    try:
        return services().releases.request(result, requested_by=body.requested_by)
    except ReleaseRejected as exc:  # 자동 게이트 미통과 → 요청 차단
        raise HTTPException(422, str(exc)) from exc


def _enforce_sod(svc, request_id: str, approver: str) -> None:
    """직무분리: 요청자(requested_by)와 승인자가 동일하면 차단."""
    req = svc.releases.store.get(request_id)
    if req is not None and req.requested_by and req.requested_by == approver:
        raise HTTPException(
            403, f"직무분리 위반: 요청자({approver})는 자신의 릴리스를 승인/반려할 수 없습니다")


@router.post("/{request_id}/approve")
def approve(
    request_id: str, body: DecisionBody,
    principal: Principal = Depends(require_perm("releases:approve")),
) -> ReleaseRequest:
    svc = services()
    approver = principal.subject  # 승인자는 인증 주체로 강제(클라이언트 입력 무시)
    _enforce_sod(svc, request_id, approver)
    try:
        out = svc.releases.approve(request_id, approver=approver)
    except ReleaseRejected as exc:
        raise HTTPException(404, str(exc)) from exc
    svc.audit.record(approver, "release.approve", target=request_id)
    return out


@router.post("/{request_id}/reject")
def reject(
    request_id: str, body: DecisionBody,
    principal: Principal = Depends(require_perm("releases:approve")),
) -> ReleaseRequest:
    svc = services()
    approver = principal.subject
    _enforce_sod(svc, request_id, approver)
    try:
        out = svc.releases.reject(
            request_id, approver=approver, reason=body.reason or "(사유 없음)")
    except ReleaseRejected as exc:
        raise HTTPException(404, str(exc)) from exc
    svc.audit.record(approver, "release.reject", target=request_id,
                     detail={"reason": body.reason})
    return out
