"""트레이서 프로바이더 초기화 — 백엔드 pluggable (옵션 A: Langfuse / 옵션 B: OTel 자체).

어느 백엔드든 계측은 OTel SDK로 표준화하므로 백엔드 교체가 설정 한 줄이다.
- backend="otel"     : OTLP exporter → OTel Collector → ClickHouse/Grafana (자체 구축)
- backend="langfuse" : Langfuse OTLP 엔드포인트로 송출 (제품 UI/프롬프트/평가 활용)
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from llmops_core.common.config import get_settings

_initialized = False


def init_telemetry() -> None:
    """프로세스 시작 시 1회 호출 (게이트웨이 app startup, 파이프라인 엔트리 등)."""
    global _initialized
    if _initialized:
        return

    s = get_settings().telemetry
    resource = Resource.create(
        {
            "service.name": s.service_name,
            "deployment.environment": get_settings().env,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter = _build_exporter(s.backend)
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _initialized = True


def _build_exporter(backend: str):
    """백엔드별 OTLP exporter 구성. 미설정/오류 시 None(무송출, 로컬 개발 허용)."""
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    s = get_settings().telemetry
    if backend == "langfuse":
        # Langfuse는 OTLP/HTTP 수신을 지원 → public/secret 키를 헤더로.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPExporter,
        )
        import base64

        auth = base64.b64encode(
            f"{s.langfuse_public_key}:{s.langfuse_secret_key}".encode()
        ).decode()
        return HTTPExporter(
            endpoint=f"{s.langfuse_host}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        )

    # 기본: OTel 자체 구축 백엔드
    return OTLPSpanExporter(endpoint=s.otlp_endpoint, insecure=True)
