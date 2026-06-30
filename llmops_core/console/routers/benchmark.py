"""벤치마크 평가 제어평면 — 평가 데이터셋 등록·실행·리더보드.

명명된 평가셋(벤치마크)에 서빙 모델(논리명)을 돌려 표준 메트릭을 산출하고, 같은 벤치마크로
여러 모델을 비교한다. 모델 추론은 게이트웨이(ModelClient) 경유 → 백엔드 무관. judge 불필요.
"""

from __future__ import annotations

import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from llmops_core.common.security import Principal
from llmops_core.console.security import require_perm
from llmops_core.console.services import services
from llmops_core.evaluation.benchmark import run_benchmark

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"],
                   dependencies=[Depends(require_perm("read"))])


class RegisterBody(BaseModel):
    name: str
    # reference: {question/text, expected/response} / preference: {prompt, chosen, rejected}
    task: str = "reference"  # "reference" | "preference"
    cases: list[dict] = Field(default_factory=list)


class RunBody(BaseModel):
    name: str   # 벤치마크 이름
    # reference: 논리 모델명(게이트웨이) / preference: base 모델 id(컨테이너 logprob 평가)
    model: str
    max_tokens: int = 128


@router.get("")
def list_benchmarks() -> list[dict]:
    return [{"name": b["name"], "task": b.get("task", "reference"),
             "num_cases": len(b.get("cases", [])), "created_at": b.get("created_at")}
            for b in services().benchmarks.list()]


@router.post("")
def register(
    body: RegisterBody, principal: Principal = Depends(require_perm("data:write")),
) -> dict:
    if not body.cases:
        raise HTTPException(422, "벤치마크 케이스가 비었습니다")
    if body.task not in ("reference", "preference"):
        raise HTTPException(422, "task는 reference|preference")
    svc = services()
    svc.benchmarks.add({"name": body.name, "task": body.task, "cases": body.cases,
                        "created_at": time.time()})
    svc.audit.record(principal.subject, "benchmark.register", target=body.name,
                     detail={"task": body.task, "n": len(body.cases)})
    return {"name": body.name, "task": body.task, "num_cases": len(body.cases)}


@router.post("/run")
def run(body: RunBody, principal: Principal = Depends(require_perm("pipeline:run"))) -> dict:
    """벤치마크를 모델에 실행 → 메트릭 산출·저장.

    reference: 게이트웨이(ModelClient) 생성 채점. preference: 컨테이너 logprob 평가(executor).
    """
    svc = services()
    bench = svc.benchmarks.get(body.name)
    if bench is None:
        raise HTTPException(404, f"벤치마크 없음: {body.name}")
    task = bench.get("task", "reference")
    rid = "bench-" + secrets.token_urlsafe(5)

    if task == "preference":
        from llmops_core.console.pipeline_engine import _executor

        try:
            res = _executor(svc).benchmark_preference(body.model, bench["cases"], run_id=rid)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"선호 벤치마크 실패(모델/컨테이너 미가용): {exc}") from exc
    else:
        from llmops_core.common.model_client import get_model_client

        client = get_model_client()

        def generate(question: str) -> str:
            return client.complete(body.model, [{"role": "user", "content": question}],
                                   max_tokens=body.max_tokens)

        try:
            res = run_benchmark(generate, bench["cases"])
        except Exception as exc:  # noqa: BLE001 — 게이트웨이/모델 미가용
            raise HTTPException(503, f"벤치마크 실행 실패(모델 미가용): {exc}") from exc

    record = {"id": rid, "benchmark": body.name, "task": task, "model": body.model,
              "metrics": res["metrics"], "num_cases": res.get("num_cases", 0),
              "errors": res.get("errors", 0), "created_at": time.time()}
    svc.benchmark_results.add(record)
    svc.audit.record(principal.subject, "benchmark.run", target=body.name,
                     detail={"model": body.model, "metrics": res["metrics"]})
    return record


@router.get("/results")
def results(name: str | None = None) -> dict:
    """결과 목록 + 벤치마크별 리더보드(answer_match→reference_f1 순)."""
    rows = [r for r in services().benchmark_results.list()
            if name is None or r["benchmark"] == name]

    def score(r: dict) -> float:
        m = r.get("metrics", {})
        return (m.get("preference_accuracy") or m.get("answer_match")
                or m.get("reference_f1") or 0.0)

    boards: dict[str, list] = {}
    for r in rows:
        boards.setdefault(r["benchmark"], []).append(r)
    leaderboard = {
        b: sorted(rs, key=score, reverse=True)
        for b, rs in boards.items()
    }
    return {"results": rows, "leaderboard": leaderboard}
