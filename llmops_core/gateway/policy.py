"""테넌트 정책 엔진 — 예산·RPM·모델 화이트리스트의 자체 구현.

게이트웨이가 모든 호출의 단일 진입점이므로, 여기서 멀티테넌시 정책을 일괄 적용한다.
스토어는 인터페이스로 분리(개발=인메모리, 운영=Redis/Postgres 교체).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from llmops_core.common.errors import (
    BudgetExceeded,
    ModelNotAllowed,
    RateLimited,
)
from llmops_core.common.schemas import TenantContext


class UsageLedger(ABC):
    """테넌트 누적 사용량(비용) 원장."""

    @abstractmethod
    def add_cost(self, tenant_id: str, cost_usd: float) -> None: ...

    @abstractmethod
    def month_to_date(self, tenant_id: str) -> float: ...


class InMemoryLedger(UsageLedger):
    def __init__(self) -> None:
        self._cost: dict[str, float] = defaultdict(float)

    def add_cost(self, tenant_id: str, cost_usd: float) -> None:
        self._cost[tenant_id] += cost_usd

    def month_to_date(self, tenant_id: str) -> float:
        return self._cost[tenant_id]


class PostgresLedger(UsageLedger):
    """Postgres 영속 사용량 원장 — 비용 항목을 append하고 당월 합산(예산 강제의 단일 출처)."""

    def __init__(self) -> None:
        from llmops_core.common.db import init_schema

        init_schema()

    def add_cost(self, tenant_id: str, cost_usd: float) -> None:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "INSERT INTO usage_ledger (tenant_id, cost_usd) VALUES (%s,%s)",
                (tenant_id, cost_usd),
            )

    def month_to_date(self, tenant_id: str) -> float:
        from llmops_core.common.db import cursor

        with cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM usage_ledger "
                "WHERE tenant_id=%s AND ts >= date_trunc('month', now())",
                (tenant_id,),
            )
            return float(cur.fetchone()[0])


class SlidingWindowRateLimiter:
    """테넌트별 RPM 슬라이딩 윈도우 (개발용 인메모리; 운영은 Redis로 교체)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, tenant_id: str, rpm_limit: int) -> None:
        now = time.time()
        window = self._hits[tenant_id]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= rpm_limit:
            raise RateLimited(f"RPM {rpm_limit} 초과 (tenant={tenant_id})")
        window.append(now)


class PolicyEngine:
    """가상키 검증 결과(TenantContext)에 정책을 적용하는 단일 지점."""

    def __init__(
        self,
        ledger: UsageLedger | None = None,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.ledger = ledger or InMemoryLedger()
        self.limiter = limiter or SlidingWindowRateLimiter()

    def authorize(self, ctx: TenantContext, model: str) -> None:
        """호출 전 검사: 화이트리스트 → RPM → 예산. 위반 시 PolicyViolation 계열."""
        # 1) 모델 화이트리스트 (빈 리스트=전체 허용)
        if ctx.allowed_models and model not in ctx.allowed_models:
            raise ModelNotAllowed(
                f"모델 '{model}' 미허용 (tenant={ctx.tenant_id})"
            )
        # 2) RPM
        if ctx.rpm_limit is not None:
            self.limiter.check(ctx.tenant_id, ctx.rpm_limit)
        # 3) 월 예산
        if ctx.monthly_budget_usd is not None:
            spent = self.ledger.month_to_date(ctx.tenant_id)
            if spent >= ctx.monthly_budget_usd:
                raise BudgetExceeded(
                    f"월 예산 ${ctx.monthly_budget_usd:.2f} 초과 "
                    f"(사용 ${spent:.2f}, tenant={ctx.tenant_id})"
                )

    def record_spend(self, ctx: TenantContext, cost_usd: float) -> None:
        """호출 후 비용 반영 (비용 추적·예산 통제)."""
        self.ledger.add_cost(ctx.tenant_id, cost_usd)
