"""준비성(readiness) 프로브 — 임계 의존성 실제 도달 확인.

liveness(/health, /api/health)는 프로세스 생존만 확인한다(정적 ok, 값싼 응답).
readiness는 실제 외부 의존성(Postgres/Redis/S3/MLflow/Qdrant)에 짧은 타임아웃으로
접속을 시도하고, 하나라도 실패하면 상위 엔드포인트가 503을 반환하게 한다.
로드밸런서/쿠버네티스 프로브가 DOWN 인스턴스를 트래픽에서 제외하는 판단 근거가 된다.

프로덕션 관측 경로: 세밀한 자원/헬스는 DCGM/Prometheus + 오케스트레이터 API가 담당한다.
여기서는 트래픽 라우팅 판단에 필요한 경량 TCP/HTTP 프로브만 수행한다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from llmops_core.common.config import get_settings

logger = logging.getLogger(__name__)

# 개별 프로브 타임아웃(초). 짧게 유지해 프로브가 매달리지 않게 한다.
_TIMEOUT_S = 2.0
# 워커 결과 수거 상한(프로브 자체 타임아웃 + 여유).
_JOIN_TIMEOUT_S = _TIMEOUT_S + 3.0


def _check_postgres() -> dict[str, Any]:
    """풀(재시도 루프 포함)을 우회한 직접 커넥션 — connect_timeout으로 매달림 방지."""
    try:
        import psycopg2

        conn = psycopg2.connect(get_settings().store.dsn, connect_timeout=int(_TIMEOUT_S))
        try:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
        finally:
            conn.close()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _check_redis(url: str) -> dict[str, Any]:
    try:
        import redis

        client = redis.from_url(
            url, socket_connect_timeout=_TIMEOUT_S, socket_timeout=_TIMEOUT_S
        )
        client.ping()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _check_s3() -> dict[str, Any]:
    """캐시된 공용 클라이언트(기본 리트라이/타임아웃)를 쓰지 않고 짧은 타임아웃 전용 클라이언트."""
    try:
        import boto3
        from botocore.client import Config

        s = get_settings().s3
        cli = boto3.client(
            "s3",
            endpoint_url=s.effective_endpoint(),
            aws_access_key_id=s.access_key,
            aws_secret_access_key=s.secret_key,
            region_name=s.region,
            config=Config(
                signature_version="s3v4",
                connect_timeout=_TIMEOUT_S,
                read_timeout=_TIMEOUT_S,
                retries={"max_attempts": 0},
            ),
        )
        cli.list_buckets()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _check_http(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        import httpx

        r = httpx.get(url, timeout=_TIMEOUT_S, headers=headers or {})
        # 5xx 는 백엔드 장애로 간주. 그 외(2xx/4xx)는 프로세스가 응답 가능 → reachable.
        if r.status_code >= 500:
            return {"ok": False, "error": f"HTTP {r.status_code}"}
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _check_mlflow() -> dict[str, Any]:
    return _check_http(get_settings().mlflow.tracking_uri.rstrip("/") + "/health")


def _check_qdrant() -> dict[str, Any]:
    q = get_settings().qdrant
    headers = {"api-key": q.api_key} if q.api_key else None
    return _check_http(q.url.rstrip("/") + "/healthz", headers=headers)


def check_readiness() -> tuple[bool, dict[str, dict[str, Any]]]:
    """설정상 활성인 임계 의존성만 병렬 프로브. (all_ok, {dep: {"ok":bool,"error"?:str}})."""
    s = get_settings()
    tasks: dict[str, Callable[[], dict[str, Any]]] = {
        # S3/MinIO·MLflow 는 산출물 저장소·실험 추적으로 항상 임계.
        "s3": _check_s3,
        "mlflow": _check_mlflow,
    }
    if s.store.backend == "postgres":
        tasks["postgres"] = _check_postgres
    if s.gateway.redis_url:
        tasks["redis"] = lambda url=s.gateway.redis_url: _check_redis(url)
    if s.rag.backend == "qdrant":
        tasks["qdrant"] = _check_qdrant

    checks: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {name: ex.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                checks[name] = fut.result(timeout=_JOIN_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001
                checks[name] = {"ok": False, "error": f"probe timeout/error: {exc}"}

    all_ok = all(c.get("ok") for c in checks.values())
    if not all_ok:
        down = [k for k, c in checks.items() if not c.get("ok")]
        logger.warning("readiness DOWN: %s", down)
    return all_ok, checks


__all__ = ["check_readiness"]
