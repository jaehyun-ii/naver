"""프롬프트 엔지니어링 체계 — 버전·라벨(prod/stg)·롤백 + 변형 비교(콘솔 UI).

GitPromptStore(불변 버전 + 라벨 포인터) 위의 제어평면. 변형 평가는 모델 추론이 필요하므로
sim(휴리스틱 점수)으로 즉시 비교하고, 실모델 평가는 학습 컨테이너 CLI(prompts.optimize)로 위임.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llmops_core.console.security import require_perm
from llmops_core.console.services import services
from llmops_core.prompts.store import PromptNotFound

router = APIRouter(prefix="/api/prompts", tags=["prompts"], dependencies=[Depends(require_perm("prompts:write"))])


class CreateVersionBody(BaseModel):
    name: str
    template: str


class PromoteBody(BaseModel):
    name: str
    version: int
    label: str = "prod"


class CompareBody(BaseModel):
    cases: list[dict] = Field(default_factory=list)  # {question, expected}
    variants: list[str] = Field(default_factory=list)  # 시스템 프롬프트 변형


@router.get("")
def list_prompts() -> list[dict]:
    store = services().prompts
    out = []
    if not store.root.exists():
        return out
    for d in sorted(p for p in store.root.iterdir() if p.is_dir()):
        name = d.name
        labels = store._read_labels(name)
        out.append({"name": name, "versions": store.versions(name), "labels": labels})
    return out


@router.post("/version")
def create_version(body: CreateVersionBody) -> dict:
    pr = services().prompts.create_version(body.name, body.template)
    return {"name": pr.name, "version": pr.version}


@router.post("/promote")
def promote(body: PromoteBody) -> dict:
    try:
        services().prompts.promote(body.name, body.version, body.label)
    except PromptNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"name": body.name, "version": body.version, "label": body.label}


@router.get("/{name}/{version}")
def get_version(name: str, version: int) -> dict:
    try:
        p = services().prompts.get(name, version)
    except PromptNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"name": p.name, "version": p.version, "template": p.template}


@router.post("/compare")
def compare(body: CompareBody) -> dict:
    """변형별 휴리스틱 점수(즉시) — 정답 키워드 포함률 기반. 실모델 평가는 CLI(prompts.optimize)."""
    def score(system: str) -> float:
        # PoC 휴리스틱: 지시형 동사·간결성 가중(실모델 평가의 프록시)
        hints = ["간결", "정확", "한 문장", "근거", "단계"]
        base = sum(1 for h in hints if h in system) / len(hints)
        brevity = max(0.0, 1.0 - abs(len(system) - 30) / 100)
        return round(0.5 * base + 0.5 * brevity, 4)

    scored = [{"variant": v, "score": score(v)} for v in body.variants]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored, "best": scored[0]["variant"] if scored else None,
            "note": "휴리스틱 비교(즉시). 실모델 평가: python -m llmops_core.prompts.optimize"}
