"""모듈 간 공통 데이터 모델 (통합 계약).

모든 모듈이 이 스키마로 데이터를 주고받아, 글루 코드의 결합을 명시적으로 만든다.
게이트웨이 표면은 OpenAI 호환을 유지한다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── 멀티테넌시 ──
class TenantContext(BaseModel):
    """가상키 검증 후 요청에 부여되는 테넌트 식별/정책 컨텍스트."""

    tenant_id: str
    key_id: str
    allowed_models: list[str] = Field(default_factory=list)  # 빈 리스트=전체 허용
    monthly_budget_usd: float | None = None
    rpm_limit: int | None = None


# ── OpenAI 호환 채팅 (게이트웨이 표면) ──
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    # 도구 호출(Tool/Function Calling) 학습용 — OpenAI 호환 선택 필드
    tool_calls: list[dict[str, Any]] | None = None  # assistant가 호출한 함수들
    tool_call_id: str | None = None  # tool 결과 메시지가 응답하는 호출 id


class ChatCompletionRequest(BaseModel):
    model: str  # 논리 모델명 (예: hcx-seed-3b)
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # 통과시킬 추가 파라미터
    extra: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ── 모델 레지스트리 / 거버넌스 ──
class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelVersionRef(BaseModel):
    """데이터 버전 ↔ run ↔ 모델 버전 사슬을 잇는 레지스트리 참조."""

    name: str
    version: str
    stage: ModelStage = ModelStage.NONE
    run_id: str | None = None
    data_version: str | None = None  # DVC 데이터셋 해시
    tenant: str | None = None
    s3_uri: str | None = None  # 체크포인트/어댑터 위치


# ── 평가 ──
class EvalCase(BaseModel):
    """고정 테스트셋의 한 케이스."""

    id: str
    question: str
    answer: str | None = None  # 모델 출력(평가 시점에 채움)
    expected: str | None = None  # 기대/정답(있으면)
    contexts: list[str] = Field(default_factory=list)  # RAG 검색 컨텍스트


class EvalResult(BaseModel):
    """평가 하니스 결과 — MLflow 메트릭으로 기록되는 단위."""

    suite: str
    model_ref: str  # 논리 모델명 또는 레지스트리 버전
    data_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = False
    num_cases: int = 0


# ── 데이터 파이프라인 (상류 L1~L3) ──
class TextRecord(BaseModel):
    """상류 정제 단계의 원천 텍스트 레코드 (적재→정제→검증 공통 단위)."""

    id: str
    text: str
    source: str | None = None  # 원천 출처(파일/테이블)
    lang: str | None = None  # 언어감지 결과(예: "ko")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SFTExample(BaseModel):
    """SFT 학습 1건 (chat 포맷). 학습기 데이터 어댑터가 그대로 소비."""

    messages: list[ChatMessage]
    source: str | None = None


class PreferenceExample(BaseModel):
    """선호 정렬(DPO/GRPO) 1건."""

    prompt: str
    chosen: str
    rejected: str
    source: str | None = None


class DataQualityReport(BaseModel):
    """품질 검증 결과 — 게이트 통과 여부와 위반 내역(파이프라인 차단 근거)."""

    suite: str
    num_records: int = 0
    passed: bool = False
    failures: dict[str, str] = Field(default_factory=dict)  # rule -> 위반 설명
    stats: dict[str, float] = Field(default_factory=dict)  # 분포/비율 등 메트릭


class DatasetManifest(BaseModel):
    """L3 산출물 — 버전 고정된 학습 데이터셋의 사슬 기록(→ MLflow run 연계)."""

    name: str
    kind: Literal["sft", "preference", "eval"] = "sft"
    fingerprint: str  # 내용 기반 sha256 (DVC와 독립적인 안정 식별자)
    dvc_url: str | None = None  # DVC가 가리키는 버전 URL/해시
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    s3_uri: str | None = None


# ── 배포 거버넌스 (Release Gateway, L5→L6) ──
class ReleaseStatus(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    REJECTED = "Rejected"


class ReleaseRequest(BaseModel):
    """배포 승인 요청 — 자동 평가 게이트 통과 후 사람/정책 승인을 받는 2차 거버넌스 단위."""

    id: str
    model_ref: str
    suite: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    auto_gate_passed: bool = False  # 1차 자동 게이트(임계값) 통과 여부
    data_version: str | None = None
    status: ReleaseStatus = ReleaseStatus.PENDING
    requested_by: str | None = None
    approver: str | None = None
    reason: str | None = None  # 반려 사유 등
    created_at: float | None = None
    decided_at: float | None = None
