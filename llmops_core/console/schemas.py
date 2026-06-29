"""콘솔 API 요청/응답 모델 — 프론트엔드 계약."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llmops_core.common.schemas import ChatMessage


# ── 키 관리 ──
class IssueKeyBody(BaseModel):
    tenant_id: str
    allowed_models: list[str] = Field(default_factory=list)
    monthly_budget_usd: float | None = None
    rpm_limit: int | None = None


class KeyView(BaseModel):
    """평문 키 제외. 발급 직후 응답에만 virtual_key가 포함된다."""

    key_id: str
    tenant_id: str
    allowed_models: list[str] = Field(default_factory=list)
    monthly_budget_usd: float | None = None
    rpm_limit: int | None = None
    spent_usd: float = 0.0


class IssuedKey(KeyView):
    virtual_key: str  # 발급 1회만 노출


# ── 챗 플레이그라운드 ──
class ChatBody(BaseModel):
    virtual_key: str
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    max_tokens: int | None = 256


class ChatReply(BaseModel):
    content: str
    model: str
    backend: str  # "router" | "proxy" | "echo"
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0


# ── 릴리스 승인 ──
class CreateReleaseBody(BaseModel):
    model_ref: str
    suite: str | None = "manual"
    metrics: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    data_version: str | None = None
    requested_by: str | None = "console"


class DecisionBody(BaseModel):
    approver: str
    reason: str | None = None


# ── 데이터/데이터셋 ──
class ValidateBody(BaseModel):
    records: list[dict[str, Any]]  # TextRecord 필드(id,text,lang,...)
    min_chars: int | None = None
    max_chars: int | None = None
    max_dup_ratio: float | None = None


class BuildDatasetBody(BaseModel):
    name: str = "console-sft"
    labeled: list[dict[str, Any]]  # {text, response}
    system: str | None = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1


# ── 파이프라인(학습·배포 DAG) ──
class PipelineStage(BaseModel):
    name: str
    title: str
    kind: str  # "real"(이 환경에서 실제 실행) | "sim"(GPU/Argo 필요 → 모의)
    status: str = "pending"  # pending|running|succeeded|failed|waiting|skipped
    detail: str = ""


class PipelineRun(BaseModel):
    id: str
    name: str
    status: str = "running"  # running|waiting|succeeded|failed
    mode: str = "sim"  # "sim" | "real"
    stages: list[PipelineStage] = Field(default_factory=list)
    release_id: str | None = None  # release-gate가 생성한 승인 요청 ID
    artifacts: dict[str, str] = Field(default_factory=dict)  # adapter/merged/serve_url 등
    created_at: float | None = None


class RunPipelineBody(BaseModel):
    name: str = "sft-train-eval-deploy"
    # mode=real → finetune/convert/deploy를 docker로 실제 실행(GB10). sim → 모의.
    mode: str = "sim"
    # 학습 방식(조선 도메인 데이터 유형·PET 변형)
    method: str = "sft"  # "sft"(Instruction) | "dpo"(Preference)
    pet: str = "lora"  # "lora" | "dora"
    # 비우면 내장 샘플 데이터로 1-클릭 데모 실행
    records: list[dict[str, Any]] | None = None  # TextRecord(L1/L2 입력)
    labeled: list[dict[str, Any]] | None = None  # {text,response}(SFT 입력; DB 조회분 가정)
    preference: list[dict[str, Any]] | None = None  # {prompt,chosen,rejected}(DPO 입력)
    metrics: dict[str, float] | None = None  # 평가 메트릭(미지정 시 샘플)
    model_ref: str = "hcx-seed-0_5b"
    served_name: str = "hcx-seed-tuned"  # 배포 시 게이트웨이 논리모델명
    train_max_steps: int = 30  # real 학습 step 상한(PoC 속도용; -1이면 epochs 사용)
    train_epochs: float = 1.0
