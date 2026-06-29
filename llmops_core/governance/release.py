"""Release Gateway — 배포 2차 거버넌스 게이트 (자체 구현, Build 모드, 두산 차용).

흐름: 평가 1차 자동 게이트(임계값) 통과 → 본 게이트가 승인 요청 생성 →
사람/정책이 승인/반려 → 승인된 경우에만 MLflow Registry 'Production' 승격 진행.

Argo 연동: WorkflowTemplate의 `suspend` 노드가 승인 대기 지점이 되고,
승인 액션이 Argo `resume`를 호출한다(본 모듈은 승인 상태의 단일 출처).
저장소는 인터페이스로 분리(개발=인메모리, 운영=Postgres 교체).
"""

from __future__ import annotations

import secrets
import time
from abc import ABC, abstractmethod

from llmops_core.common.errors import ReleasePending, ReleaseRejected
from llmops_core.common.schemas import (
    EvalResult,
    ReleaseRequest,
    ReleaseStatus,
)


class ReleaseStore(ABC):
    """승인 요청 저장소 계약. 운영 시 Postgres 구현으로 교체."""

    @abstractmethod
    def put(self, req: ReleaseRequest) -> None: ...

    @abstractmethod
    def get(self, request_id: str) -> ReleaseRequest | None: ...

    @abstractmethod
    def list(self, *, status: ReleaseStatus | None = None) -> list[ReleaseRequest]: ...


class InMemoryReleaseStore(ReleaseStore):
    def __init__(self) -> None:
        self._by_id: dict[str, ReleaseRequest] = {}

    def put(self, req: ReleaseRequest) -> None:
        self._by_id[req.id] = req

    def get(self, request_id: str) -> ReleaseRequest | None:
        return self._by_id.get(request_id)

    def list(self, *, status: ReleaseStatus | None = None) -> list[ReleaseRequest]:
        reqs = list(self._by_id.values())
        return [r for r in reqs if status is None or r.status == status]


class PostgresReleaseStore(ReleaseStore):
    """Postgres 영속 승인 요청 저장소 — 승인 상태의 단일 출처(다중 인스턴스 공유)."""

    def __init__(self) -> None:
        from llmops_core.common.db import init_schema

        init_schema()

    def put(self, req: ReleaseRequest) -> None:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "INSERT INTO release_requests (id, status, payload) VALUES (%s,%s,%s) "
                "ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, payload=EXCLUDED.payload",
                (req.id, req.status.value, req.model_dump_json()),
            )

    def get(self, request_id: str) -> ReleaseRequest | None:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute("SELECT payload FROM release_requests WHERE id=%s", (request_id,))
            row = cur.fetchone()
        return ReleaseRequest.model_validate(row[0]) if row else None

    def list(self, *, status: ReleaseStatus | None = None) -> list[ReleaseRequest]:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            if status is None:
                cur.execute("SELECT payload FROM release_requests ORDER BY created_at DESC")
            else:
                cur.execute(
                    "SELECT payload FROM release_requests WHERE status=%s ORDER BY created_at DESC",
                    (status.value,),
                )
            rows = cur.fetchall()
        return [ReleaseRequest.model_validate(r[0]) for r in rows]


class ReleaseGateway:
    """승인 요청·승인·반려·검증의 단일 지점."""

    def __init__(self, store: ReleaseStore | None = None) -> None:
        self.store = store or InMemoryReleaseStore()

    def request(self, result: EvalResult, *, requested_by: str | None = None) -> ReleaseRequest:
        """평가 결과로 승인 요청 생성. **1차 자동 게이트 미통과면 요청 자체를 차단**한다."""
        if not result.passed:
            raise ReleaseRejected(
                f"자동 평가 게이트 미통과 — 승인 요청 불가 (suite={result.suite})"
            )
        req = ReleaseRequest(
            id="rel-" + secrets.token_urlsafe(8),
            model_ref=result.model_ref,
            suite=result.suite,
            metrics=result.metrics,
            auto_gate_passed=True,
            data_version=result.data_version,
            status=ReleaseStatus.PENDING,
            requested_by=requested_by,
            created_at=time.time(),
        )
        self.store.put(req)
        return req

    def approve(self, request_id: str, approver: str) -> ReleaseRequest:
        req = self._require(request_id)
        if req.status != ReleaseStatus.PENDING:
            return req  # 멱등: 이미 결정됨
        updated = req.model_copy(
            update={
                "status": ReleaseStatus.APPROVED,
                "approver": approver,
                "decided_at": time.time(),
            }
        )
        self.store.put(updated)
        return updated

    def reject(self, request_id: str, approver: str, reason: str) -> ReleaseRequest:
        req = self._require(request_id)
        updated = req.model_copy(
            update={
                "status": ReleaseStatus.REJECTED,
                "approver": approver,
                "reason": reason,
                "decided_at": time.time(),
            }
        )
        self.store.put(updated)
        return updated

    def require_approved(self, request_id: str) -> ReleaseRequest:
        """승격 직전 호출. 승인 전이면 ReleasePending, 반려면 ReleaseRejected."""
        req = self._require(request_id)
        if req.status == ReleaseStatus.REJECTED:
            raise ReleaseRejected(f"{request_id} 반려됨: {req.reason}")
        if req.status != ReleaseStatus.APPROVED:
            raise ReleasePending(f"{request_id} 승인 대기 중")
        return req

    def _require(self, request_id: str) -> ReleaseRequest:
        req = self.store.get(request_id)
        if req is None:
            raise ReleaseRejected(f"승인 요청 없음: {request_id}")
        return req
