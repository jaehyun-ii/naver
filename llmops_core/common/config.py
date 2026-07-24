"""중앙 설정 — pydantic-settings 기반. 환경변수 `LLMOPS_*` 로 주입.

중첩 설정은 `__` 구분자를 사용한다. 예) LLMOPS_S3__ENDPOINT_URL
모든 외부 종속(엔드포인트/키)을 한 곳에 모아 Phase 2 NCP 전환을 설정 변경 수준으로 만든다.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 기본값으로 절대 운영 기동 금지 대상(placeholder). prod/staging에서 이 값이면 fail-fast.
PLACEHOLDER_MASTER_KEY = "sk-master-changeme"
PLACEHOLDER_S3_SECRET = "minioadmin"


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
    master_key: str = PLACEHOLDER_MASTER_KEY
    config_path: str = "./config/model_list.yaml"
    redis_url: str | None = None  # 응답 캐시·RPM 리미터 공유 백엔드(다중 인스턴스). None이면 인메모리.
    # 콘솔이 게이트웨이 관리 API(캐시 통계 등)를 조회할 베이스 URL. 컴포즈에선 서비스명 사용.
    base_url: str = "http://localhost:4000"
    # 요청 입력 상한(자원고갈 방지). 초과 시 400.
    max_messages: int = 200
    max_input_chars: int = 200_000
    # self-host 모델 토큰 단가(USD/token). litellm 가격표에 없는 커스텀 모델의 비용/예산 산정에 사용.
    # {모델명: {"input": 0.0, "output": 0.0}}. 미정의 모델은 비용 0 대신 경고 후 0 처리.
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)


class CacheSettings(BaseModel):
    """게이트웨이 응답 캐싱 — 동일 요청 재사용으로 토큰 비용 절감(비용관리 보강).

    backend=memory(프로세스 LRU) | redis(공유·다중인스턴스). 키는 모델+메시지+파라미터 해시.
    """

    enabled: bool = True
    backend: str = "memory"  # "memory" | "redis"
    ttl_s: int = 3600
    max_entries: int = 1024  # memory 백엔드 LRU 상한


class GuardrailSettings(BaseModel):
    """서빙 시점 가드레일 — 입력 프롬프트인젝션·출력 모더레이션·PII 누출 차단.

    순수 휴리스틱 기본(추가 의존성 0). mask_output_pii는 Presidio(quality extra) 필요.
    """

    enabled: bool = True
    block_on_injection: bool = True  # 인젝션 탐지 시 요청 차단(False면 통과·로깅만)
    block_on_banned: bool = True  # 금칙어 출력 차단
    mask_output_pii: bool = False  # 출력 PII 마스킹(Presidio 필요). prod 강제 on(검증기).
    mask_input_pii: bool = False  # 입력 PII 마스킹/검사
    pii_languages: list[str] = Field(default_factory=lambda: ["ko", "en"])  # PII 인식 언어
    # 마스커/분류모델 의존성 미가용 시 동작: True면 요청 차단(fail-closed), False면 통과(로깅).
    fail_closed: bool = False
    banned_terms: list[str] = Field(default_factory=list)
    # 모델 기반 가드레일 — model_list.yaml의 논리명(예: LlamaGuard 서빙). None이면 휴리스틱만.
    # 분류 모델이 'unsafe' 판정 시 입력 차단(휴리스틱과 OR 결합).
    model: str | None = None
    block_on_model_flag: bool = True
    # 분류 프롬프트 — GitPromptStore 이름(prod 라벨). 미등록 시 내장 기본값. 변수: {text}
    prompt_name: str = "guardrail-classifier"


class RagSettings(BaseModel):
    """RAG 서빙 — 임베딩→벡터스토어→검색→컨텍스트 주입. 런타임 검색 경로.

    embedder=bge-m3(sentence-transformers, 캐시됨) | hashing(순수파이썬 폴백·무의존).
    backend=memory(코사인·디스크영속) | qdrant(QdrantSettings 사용).
    """

    embedder: str = "hashing"  # "bge-m3" | "st"(임의 ST 모델) | "hashing"
    embedding_model: str = "BAAI/bge-m3"
    dim: int = 256  # hashing 임베더 차원(bge-m3는 모델 차원 사용)
    backend: str = "memory"  # "memory" | "qdrant" | "qdrant_parents"
    collection: str = "default"
    top_k: int = 4
    # ── parent-직접 검색 스택(Nemotron 벤치 채택안) ──
    # query_prompt: 비대칭 임베더의 질의측 프롬프트명(예: nemotron "query")
    # parent_db: parent 본문 사이드카(sqlite) — qdrant_parents 백엔드가 하이드레이션
    # reranker_model: cross-encoder 리랭커(미설정 시 리랭크 생략)
    query_prompt: str | None = None
    parent_db: str | None = None
    reranker_model: str | None = None
    rerank_candidates: int = 20
    # 컨텍스트 주입 위치 — "system"(범용) | "user"(RAFT 학습 형식: user에 컨텍스트+[질문]).
    # HCX 계열은 chat template이 system을 못 받으므로 user 모드가 학습·서빙 정합.
    inject_mode: str = "system"
    persist_path: str | None = "./rag_store"  # memory 백엔드 디스크 영속(없으면 휘발)
    # 서빙 시점 자동 RAG — 게이트웨이가 요청을 검색·컨텍스트 주입 후 모델 호출.
    # 기본 off(요청별 extra.rag로 override). 켜면 평가(--rag)와 서빙이 동일 경로로 정합.
    serving_enabled: bool = False


class TelemetrySettings(BaseModel):
    backend: str = "otel"  # "otel" | "langfuse"
    otlp_endpoint: str = "http://localhost:4317"
    service_name: str = "llmops-core"
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # 콘솔이 트레이스를 조회할 Jaeger Query API(메인 콘솔 통합 뷰). 컴포즈에선 서비스명.
    jaeger_query_url: str = "http://localhost:16686"


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
    # 콘솔/게이트웨이 CORS 허용 오리진. prod에서 "*"는 금지(검증기).
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    s3: S3Settings = Field(default_factory=S3Settings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
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

    @property
    def is_prod(self) -> bool:
        return self.env in ("staging", "prod")

    @model_validator(mode="after")
    def _enforce_prod_safety(self) -> "Settings":
        """운영(env=staging|prod) 기동 시 PoC 기본값을 거부(fail-fast).

        dev에서는 무동작 — 데모/로컬 편의 유지. 운영 배포는 반드시 실값을 주입해야 기동된다.
        """
        if not self.is_prod:
            return self
        errs: list[str] = []
        if self.gateway.master_key in (PLACEHOLDER_MASTER_KEY, ""):
            errs.append("gateway.master_key 가 기본/빈 값입니다 (LLMOPS_GATEWAY__MASTER_KEY 주입 필요).")
        if self.store.backend != "postgres":
            errs.append("store.backend 는 prod에서 'postgres' 여야 합니다 (인메모리는 휘발).")
        if self.s3.secret_key == PLACEHOLDER_S3_SECRET:
            errs.append("s3.secret_key 가 기본값(minioadmin)입니다 (실 크리덴셜 주입 필요).")
        if self.cors_origins == ["*"]:
            errs.append("cors_origins 에 '*' 는 prod에서 금지입니다 (오리진 화이트리스트 지정).")
        if not self.guardrails.mask_output_pii:
            errs.append("guardrails.mask_output_pii 는 prod에서 on 이어야 합니다 (출력 PII 마스킹).")
        if self.rag.serving_enabled and (
            self.rag.embedder == "hashing" or self.rag.backend == "memory"
        ):
            errs.append("rag 서빙 활성 시 embedder=bge-m3·backend=qdrant 여야 합니다 (해싱/인메모리 금지).")
        if self.cache.enabled and self.cache.backend != "redis" and not self.gateway.redis_url:
            errs.append("prod 캐시/리미터는 redis 공유 백엔드 권장 — gateway.redis_url 미설정.")
        if errs:
            raise ValueError(
                "운영 설정 검증 실패 (env=" + self.env + "):\n  - " + "\n  - ".join(errs)
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """프로세스 단일 설정 인스턴스."""
    return Settings()
