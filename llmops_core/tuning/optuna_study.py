"""HPO — optuna 임베드 (TPE 샘플러 + Pruner). 각 trial을 MLflow에 기록.

Optuna 탐색 잡 자체를 Argo Workflows의 한 단계로 실행 → 파이프라인 내 자동 HPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from llmops_core.common.errors import OptionalDependencyError


@dataclass
class HPOConfig:
    study_name: str = "sft-hpo"
    direction: str = "maximize"  # 평가 메트릭 최대화
    n_trials: int = 20


def create_study(cfg: HPOConfig):
    """TPE 샘플러 + Median Pruner 스터디 생성."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("optuna", "tuning") from exc

    return optuna.create_study(
        study_name=cfg.study_name,
        direction=cfg.direction,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
    )


def optimize(
    cfg: HPOConfig,
    objective: Callable[["object"], float],
    *,
    tracker=None,
):
    """objective(trial)->score 를 n_trials 동안 탐색. tracker로 trial 기록(선택).

    반환: (best_params, best_value, trials) — trials는 per-trial 파라미터+점수 기록.
    """
    study = create_study(cfg)

    def wrapped(trial):
        score = objective(trial)
        if tracker is not None:
            with tracker.run({"trial": trial.number, **trial.params}):
                tracker.mlflow.log_metric("hpo.score", score)
        return score

    study.optimize(wrapped, n_trials=cfg.n_trials)
    trials = [
        {"number": t.number, "params": dict(t.params),
         "value": t.value, "state": str(t.state)}
        for t in study.trials
    ]
    return study.best_params, study.best_value, trials
