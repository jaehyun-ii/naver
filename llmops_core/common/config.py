"""중앙 설정 — pydantic-settings 기반. 환경변수 `LLMOPS_*` 로 주입.

중첩 설정은 `__` 구분자를 사용한다. 예) LLMOPS_S3__ENDPOINT_URL
모든 외부 종속(엔드포인트/키)을 한 곳에 모아 Phase 2 NCP 전환을 설정 변경 수준으로 만든다.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class S3Settings(BaseModel):
    """S3 호환 오브젝트 스토리지(MinIO → NCP Object Storage 교체 지점).

    온프렘=MinIO, 클라우드=NCP Object Storage. provider만 바꾸면 엔드포인트 기본값이 전환되고,
    개별 값은 LLMOPS_S3__* 로 덮어쓴다(둘 다 S3 v4 서명·boto3 동일 코드).
    """

    provider: str = "minio"  # "minio"(온프렘) | "ncp"(NAVER Cloud Object Storage)
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    region: str = "us-east-1"
    # NCP 기본 엔드포인트(provider=ncp이고 endpoint 미지정 시 사용). 리전은 kr-standard 등.
    ncp_endpoint_url: str = "https://kr.object.ncloudstorage.com"

    def effective_endpoint(self) -> str:
        if self.provider == "ncp" and self.endpoint_url in ("", "http://localhost:9000"):
            return self.ncp_endpoint_url
        return self.endpoint_url


class GatewaySettings(BaseModel):
    master_key: str = "sk-master-changeme"
    config_path: str = "./config/model_list.yaml"
    redis_url: str | None = None  # 응답 캐시(선택)


class TelemetrySettings(BaseModel):
    backend: str = "otel"  # "otel" | "langfuse"
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "llmops-core"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None


class MLflowSettings(BaseModel):
    tracking_uri: str = "http://localhost:5000"


class GpuNode(BaseModel):
    """GPU 노드 — Docker 호스트 + GPU 수. docker_host=None이면 로컬 소켓."""

    name: str = "local"
    docker_host: str | None = None  # 예: tcp://10.0.0.2:2375 (원격 노드)
    gpus: int = 1
    devices: list[int] | None = None  # 명시적 디바이스 ID(공용 서버 빈 GPU 지정). 없으면 range(gpus)


class OrchestrationSettings(BaseModel):
    """GPU 잡 오케스트레이션 — 쿠버네티스 미사용, Docker 기반 다중 GPU/노드 스케줄.

    단일 H200×8: nodes=[{name:local, gpus:8}].
    2노드 H200×4: nodes=[{name:a, docker_host:..., gpus:4}, {name:b, ..., gpus:4}].
    """

    nodes: list[GpuNode] = Field(default_factory=lambda: [GpuNode()])
    acquire_timeout_s: int = 3600


class StoreSettings(BaseModel):
    """제어평면 상태 저장소 — 개발=memory, 운영=postgres(영속·HA).

    backend=postgres면 키·승인·예산·파이프라인런이 Postgres에 영속화되어
    프로세스 재기동/다중 인스턴스에도 상태가 보존된다(단일실패점 제거).
    """

    backend: str = "memory"  # "memory" | "postgres"
    dsn: str = "postgresql://mlflow:mlflow@localhost:5432/llmops"


class PromptStoreSettings(BaseModel):
    repo_path: str = "./prompt_store"


class ArgillaSettings(BaseModel):
    """라벨링 서버(Argilla) — Service. 코어는 argilla SDK만 임베드."""

    url: str = "http://localhost:6900"
    api_key: str | None = None
    workspace: str = "default"


class DataQualitySettings(BaseModel):
    """상류 데이터 정제·검증 임계값 (L1/L2 게이트)."""

    fasttext_model_path: str | None = None  # lid.176.bin 경로(언어감지). 없으면 감지 스킵
    allowed_langs: list[str] = Field(default_factory=lambda: ["ko", "en"])
    min_chars: int = 10  # 너무 짧은 레코드 제거
    max_chars: int = 20000  # 과도하게 긴 레코드 제거
    max_dup_ratio: float = 0.2  # 근사중복 허용 상한(초과 시 게이트 실패)
    near_dup_threshold: float = 0.85  # MinHash Jaccard 임계값


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMOPS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    env: str = "dev"  # dev | staging | prod
    domain: str = "demo"

    s3: S3Settings = Field(default_factory=S3Settings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    orch: OrchestrationSettings = Field(default_factory=OrchestrationSettings)
    prompts: PromptStoreSettings = Field(default_factory=PromptStoreSettings)
    argilla: ArgillaSettings = Field(default_factory=ArgillaSettings)
    data_quality: DataQualitySettings = Field(default_factory=DataQualitySettings)

    def bucket(self, purpose: str) -> str:
        """버킷 명명 규칙 `{env}-{domain}-{purpose}` (클라우드 호환 고정)."""
        return f"{self.env}-{self.domain}-{purpose}"


@lru_cache
def get_settings() -> Settings:
    """프로세스 단일 설정 인스턴스."""
    return Settings()
