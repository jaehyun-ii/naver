"""학습 데이터 버전별 모델 성능평가 — 파이프라인 run의 fingerprint↔메트릭 비교 (콘솔 UI).

각 파이프라인 run은 데이터셋 fingerprint(데이터 버전)와 평가 메트릭을 남긴다. 이를 데이터
버전 기준으로 모아 '같은/다른 데이터로 학습한 모델의 성능 비교'를 제공한다(design 6.5).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends

from llmops_core.console.security import require_perm
from llmops_core.console.services import services

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"],
                   dependencies=[Depends(require_perm("read"))])


def _parse_metrics(detail: str) -> dict:
    """구버전 'k=v, k=v' 문자열 메트릭 파서(하위호환)."""
    out = {}
    for kv in (detail or "").split(", "):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def _run_metrics(run) -> dict:
    """run의 구조화 메트릭(metrics_json) 우선, 없으면 구버전 문자열 파싱(하위호환)."""
    raw = run.artifacts.get("metrics_json")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, (int, float))}
        except (ValueError, TypeError):
            pass
    return _parse_metrics(run.artifacts.get("metrics", ""))


@router.get("/by-version")
def by_version() -> list[dict]:
    """데이터 버전(fingerprint)별로 run·메트릭을 묶어 반환(최신순)."""
    groups: dict[str, dict] = {}
    for run in services().pipelines.list():
        fp = run.artifacts.get("fingerprint")
        if not fp:
            continue
        metrics = _run_metrics(run)
        entry = {
            "run_id": run.id, "name": run.name,
            "status": run.status, "method": run.artifacts.get("served_name", ""),
            "metrics": metrics, "created_at": run.created_at,
        }
        g = groups.setdefault(fp, {"fingerprint": fp, "runs": []})
        g["runs"].append(entry)
    # 버전별 최고 점수 요약
    out = []
    for fp, g in groups.items():
        best = None
        for r in g["runs"]:
            m = r["metrics"]
            # 게이트 신호 우선순위, 없으면 임의 메트릭
            score = (m.get("answer_match") or m.get("preference_accuracy")
                     or m.get("reference_f1") or m.get("faithfulness")
                     or (next(iter(m.values()), None)))
            if score is not None and (best is None or score > best):
                best = score
        out.append({**g, "best_score": best, "num_runs": len(g["runs"])})
    out.sort(key=lambda x: max((r["created_at"] or 0) for r in x["runs"]), reverse=True)
    return out
