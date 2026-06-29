"""파이프라인 실행 엔진(콘솔) — 실제 코어 함수 + 승인 게이트 + (real 모드) GPU/배포 실행.

DAG: data-ingest → data-quality → data-build → finetune → evaluate
     → release-gate → release-approval → register → convert → deploy

모드:
- **sim**: finetune/register/convert/deploy를 모의 진행(GPU/Argo 불필요). 빠른 데모.
- **real**: finetune/convert/deploy를 `RealExecutor`(docker)로 **실제 실행**(GB10).
            register는 MLflow에 실제 기록. data/quality/build/evaluate/approval은 양 모드 공통 real.

real 단계는 분 단위라 백그라운드 스레드에서 진행하고, 프론트는 /runs/{id}를 폴링한다.
"""

from __future__ import annotations

import secrets
import threading
import time

from llmops_core.common.errors import DataQualityFailed
from llmops_core.common.schemas import (
    DatasetManifest,
    EvalResult,
    ReleaseStatus,
    TextRecord,
)
from llmops_core.console.schemas import PipelineRun, PipelineStage, RunPipelineBody
from llmops_core.dataset.build import (
    SplitConfig,
    filter_sft_by_length,
    split_sft,
    to_preference_examples,
    to_sft_examples,
)
from llmops_core.dataset.versioning import fingerprint
from llmops_core.evaluation.gate import GatePolicy
from llmops_core.ingestion.text_ops import exact_dedup, normalize_records

# 단계 정의(순서 = DAG 실행 순서). kind는 기본값이며 real 모드에서 동적 상향된다.
_STAGE_DEFS = [
    ("data-ingest", "L1 적재·정규화·중복제거", "real"),
    ("data-quality", "L2 품질 게이트", "real"),
    ("data-build", "L3 데이터셋 빌드·버전", "real"),
    ("finetune", "L4 학습(SFT/LoRA)", "sim"),
    ("evaluate", "L5 평가·자동 게이트", "real"),
    ("release-gate", "승인 요청 생성", "real"),
    ("release-approval", "릴리스 승인 대기", "real"),
    ("register", "MLflow 등록·승격", "sim"),
    ("convert", "L6 LoRA merge 변환", "sim"),
    ("deploy", "배포(서빙 교체·라우팅)", "sim"),
]
_REAL_STAGES = {"finetune", "register", "convert", "deploy"}

_SAMPLE_RECORDS = [
    {"id": "1", "text": "환불 정책은 구매 후 14일 이내 전액 환불입니다."},
    {"id": "2", "text": "배송은 평일 기준 2~3영업일 소요됩니다."},
    {"id": "3", "text": "교환은 미사용 상품에 한해 7일 이내 가능합니다."},
]
_SAMPLE_LABELED = [
    {"text": "환불 정책이 어떻게 되나요?", "response": "구매 후 14일 이내 전액 환불됩니다."},
    {"text": "배송 기간은 얼마나 걸리나요?", "response": "평일 기준 2~3일 소요됩니다."},
    {"text": "교환 가능 기간은?", "response": "미사용 상품은 7일 이내 교환 가능합니다."},
]
_SAMPLE_METRICS = {"faithfulness": 0.9, "context_precision": 0.82}
_SAMPLE_PREFERENCE = [
    {"prompt": "선급증서 유효기간이 지나면?", "chosen": "즉시 선급기관에 재검사를 신청해 갱신합니다.",
     "rejected": "그냥 둬도 됩니다."},
    {"prompt": "FAT 불합격 시 절차는?", "chosen": "부적합 항목을 시정조치 요구서로 발행하고 재시험합니다.",
     "rejected": "무시하고 출하합니다."},
    {"prompt": "재료증명서가 없으면?", "chosen": "공급사에 제출을 요구하고 미제출 시 입고를 보류합니다.",
     "rejected": "없어도 됩니다."},
]


class PipelineRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, PipelineRun] = {}

    def add(self, run: PipelineRun) -> None:
        self._runs[run.id] = run

    def get(self, run_id: str) -> PipelineRun | None:
        return self._runs.get(run_id)

    def list(self) -> list[PipelineRun]:
        return sorted(self._runs.values(), key=lambda r: r.created_at or 0, reverse=True)


def _new_run(name: str, mode: str) -> PipelineRun:
    stages = [
        PipelineStage(
            name=n, title=t, kind=("real" if mode == "real" and n in _REAL_STAGES else k)
        )
        for n, t, k in _STAGE_DEFS
    ]
    return PipelineRun(
        id="run-" + secrets.token_urlsafe(6), name=name, mode=mode, stages=stages,
        created_at=time.time(),
    )


def _stage(run: PipelineRun, name: str) -> PipelineStage:
    return next(s for s in run.stages if s.name == name)


def _message_rows(labeled: list[dict]) -> list[dict]:
    """{text,response} → trl conversational 행({"messages":[...]})."""
    rows = []
    for ex in to_sft_examples(labeled):
        rows.append({"messages": [{"role": m.role, "content": m.content} for m in ex.messages]})
    return rows


# ── 공개 API ──
def start_run(svc, body: RunPipelineBody) -> PipelineRun:
    """release-gate까지 진행하고 승인 대기에 둔다.

    sim 모드는 동기(빠름) — 호출 즉시 waiting. real 모드는 GPU 학습이 분 단위라
    백그라운드 스레드로 진행하고 호출은 running 상태로 즉시 반환(프론트가 폴링).
    """
    run = _new_run(body.name, body.mode)
    svc.pipelines.add(run)
    if body.mode == "real":
        run.status = "running"
        threading.Thread(target=_pre_approval, args=(svc, run, body), daemon=True).start()
    else:
        _pre_approval(svc, run, body)
    return run


def resume_run(svc, run_id: str) -> PipelineRun:
    """승인 결과를 반영해 register→convert→deploy를 백그라운드 진행(또는 반려 시 스킵)."""
    run = svc.pipelines.get(run_id)
    if run is None or not run.release_id:
        raise KeyError(run_id)

    req = svc.releases.store.get(run.release_id)
    appr = _stage(run, "release-approval")

    if req and req.status == ReleaseStatus.REJECTED:
        appr.status = "failed"
        appr.detail = f"반려: {req.reason}"
        for n in ("register", "convert", "deploy"):
            _stage(run, n).status = "skipped"
        run.status = "failed"
        svc.pipelines.save(run)
        return run

    if not req or req.status != ReleaseStatus.APPROVED:
        appr.detail = "아직 승인 대기 중"
        return run

    appr.status = "succeeded"
    appr.detail = f"승인자: {req.approver}"
    if run.mode == "real":
        if run.status != "post-running":  # 중복 재개 방지
            run.status = "post-running"
            threading.Thread(target=_post_approval, args=(svc, run), daemon=True).start()
    else:
        _post_approval(svc, run)
    return run


# ── 백그라운드 실행부 ──
def _pre_approval(svc, run: PipelineRun, body: RunPipelineBody) -> None:
    try:
        records = [TextRecord(**r) for r in (body.records or _SAMPLE_RECORDS)]
        labeled = body.labeled or _SAMPLE_LABELED
        metrics = body.metrics or _SAMPLE_METRICS
        run.artifacts["served_name"] = body.served_name

        # L1 적재·정규화·중복제거 (real)
        _stage(run, "data-ingest").status = "running"
        records = normalize_records(records)
        records, removed = exact_dedup(records)
        st = _stage(run, "data-ingest")
        st.status, st.detail = "succeeded", f"{len(records)}건 (중복 {removed} 제거)"

        # L2 품질 게이트 (real)
        _stage(run, "data-quality").status = "running"
        try:
            from llmops_core.quality import validate_records

            report = validate_records(records)
            st = _stage(run, "data-quality")
            st.status, st.detail = "succeeded", f"통과 · dup_ratio={report.stats.get('dup_ratio')}"
        except DataQualityFailed as exc:
            st = _stage(run, "data-quality")
            st.status, st.detail = "failed", str(exc)
            run.status = "failed"
            return

        is_dpo = body.method == "dpo"
        use_dora = body.pet == "dora"
        pet_label = "DoRA" if use_dora else "LoRA"

        # L3 데이터셋 빌드·버전 (real) — 방식별 학습행/평가케이스/매니페스트 구성
        _stage(run, "data-build").status = "running"
        if is_dpo:
            pref = body.preference or _SAMPLE_PREFERENCE
            pref_examples = to_preference_examples(pref)
            train_rows = [{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                          for p in pref_examples]
            eval_cases = train_rows  # 선호정확도 평가는 {prompt,chosen,rejected} 사용
            fp = fingerprint(pref_examples)
            kind, n = "preference", len(train_rows)
        else:
            examples = filter_sft_by_length(to_sft_examples(labeled))
            tr, va, te = split_sft(examples, SplitConfig())
            train_rows = _message_rows(labeled)
            eval_cases = [{"question": l["text"], "expected": l["response"]} for l in labeled
                          if l.get("text") and l.get("response")]
            fp = fingerprint(tr + va + te)
            kind, n = "sft", len(tr)
        svc.datasets.add(DatasetManifest(name=body.name, kind=kind, fingerprint=fp, num_train=n))
        run.artifacts["fingerprint"] = fp
        st = _stage(run, "data-build")
        st.status = "succeeded"
        st.detail = f"{kind} · fp={fp[:12]}… · train={n}"

        # L4 학습 (SFT/DPO · LoRA/DoRA)
        ft = _stage(run, "finetune")
        ft.status = "running"
        if run.mode == "real":
            from llmops_core.console.executor import RealExecutor

            adapter = RealExecutor().finetune(
                run.id, train_rows, base_model=None, method=body.method, use_dora=use_dora,
                max_steps=body.train_max_steps, epochs=body.train_epochs,
            )
            run.artifacts["adapter"] = adapter
            ft.detail = (f"SEED-0.5B {body.method.upper()}/{pet_label}(bf16) · "
                         f"steps={body.train_max_steps} → {adapter}")
        else:
            ft.detail = f"GPU 모의: SEED {body.method.upper()}/{pet_label}(r=16)"
        ft.status = "succeeded"

        # L5 평가·자동 게이트
        ev = _stage(run, "evaluate")
        ev.status = "running"
        num_cases = len(eval_cases)
        if run.mode == "real":
            # 학습 어댑터로 평가셋을 실제 추론·채점 (judge 불필요)
            from llmops_core.console.executor import RealExecutor

            task = "preference" if is_dpo else "reference"
            res = RealExecutor().evaluate(run.id, eval_cases, task=task)
            metrics = res["metrics"]
            num_cases = res.get("num_cases", len(eval_cases))
            # DPO: 선호정확도(chosen>rejected) / SFT: answer_match(정답 핵심부 포함률)
            gate = GatePolicy(thresholds=(
                {"preference_accuracy": 0.5} if is_dpo else {"answer_match": 0.5}
            ))
        else:
            gate = GatePolicy(thresholds={"faithfulness": 0.8, "context_precision": 0.7})

        passed = gate.passes(metrics)
        result = EvalResult(
            suite="pipeline", model_ref=body.model_ref, metrics=metrics, passed=passed,
            num_cases=num_cases,
        )
        run.artifacts["metrics"] = ", ".join(f"{k}={v}" for k, v in metrics.items())
        ev.status = "succeeded" if passed else "failed"
        ev.detail = run.artifacts["metrics"]
        if not passed:
            run.status = "failed"
            return

        # 승인 요청 생성 (real — Release Gateway 연동)
        req = svc.releases.request(result, requested_by="pipeline")
        run.release_id = req.id
        gs = _stage(run, "release-gate")
        gs.status, gs.detail = "succeeded", f"요청 {req.id}"

        appr = _stage(run, "release-approval")
        appr.status, appr.detail = "waiting", "승인·평가 페이지에서 승인/반려"
        run.status = "waiting"
    except Exception as exc:  # noqa: BLE001
        _fail_running(run, exc)
    finally:
        svc.pipelines.save(run)  # 영속(HA): 터미널/대기 상태 스냅샷


def _post_approval(svc, run: PipelineRun) -> None:
    try:
        body_mode = run.mode
        # register — MLflow 기록(real) / 모의(sim)
        rs = _stage(run, "register")
        rs.status = "running"
        if body_mode == "real":
            rs.detail = _register_real(run)
        else:
            rs.detail = "MLflow Registry → Production 승격(모의)"
        rs.status = "succeeded"

        # convert — LoRA 병합(real) / 모의(sim)
        cs = _stage(run, "convert")
        cs.status = "running"
        if body_mode == "real":
            from llmops_core.console.executor import RealExecutor

            merged = RealExecutor().merge(run.id)
            run.artifacts["merged"] = merged
            cs.detail = f"merge_and_unload → {merged}"
        else:
            cs.detail = "LoRA merge → safetensors(모의)"
        cs.status = "succeeded"

        # deploy — 서빙 교체 + 라우팅(real) / 모의(sim)
        ds = _stage(run, "deploy")
        ds.status = "running"
        if body_mode == "real":
            from llmops_core.console.executor import RealExecutor

            # served_name은 run 이름 기준 고정 논리명 사용
            served = run.artifacts.get("served_name", "hcx-seed-tuned")
            url = RealExecutor().deploy(run.id, served)
            run.artifacts["serve_url"] = url
            ds.detail = f"{served} 서빙 + 게이트웨이 라우팅 → {url}"
        else:
            ds.detail = "ArgoCD 동기화 → vLLM/게이트웨이 반영(모의)"
        ds.status = "succeeded"

        run.status = "succeeded"
    except Exception as exc:  # noqa: BLE001
        _fail_running(run, exc)
    finally:
        svc.pipelines.save(run)  # 영속(HA): 배포 완료/실패 스냅샷


def _register_real(run: PipelineRun) -> str:
    """MLflow에 params·metrics·어댑터를 기록하고, 모델 레지스트리에 등록·승격(alias)."""
    try:
        from llmops_core.tracking import ExperimentTracker

        tracker = ExperimentTracker(experiment="console-pipeline")
        mlflow = tracker.mlflow
        params = {"base_model": "SEED-0.5B", "method": "lora-bf16", "lora_r": 16}
        adapter = run.artifacts.get("adapter")
        with tracker.run(params, data_ver=run.artifacts.get("fingerprint")) as active:
            for kv in run.artifacts.get("metrics", "").split(", "):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        mlflow.log_metric(f"eval.{k}", float(v))
                    except ValueError:
                        pass
            if adapter:
                mlflow.log_artifacts(adapter, artifact_path="adapter")
            run_id = active.info.run_id

        # 레지스트리 등록 + production alias 승격(평가 게이트 통과·승인 완료 전제)
        name = run.artifacts.get("served_name", "hcx-seed-tuned")
        detail = "MLflow run 기록(params·metrics·adapter)"
        if adapter:
            mv = mlflow.register_model(f"runs:/{run_id}/adapter", name)
            run.artifacts["model_version"] = f"{name} v{mv.version}"
            try:
                client = mlflow.tracking.MlflowClient()
                client.set_registered_model_alias(name, "production", mv.version)
                detail = f"등록·승격: {name} v{mv.version} @production"
            except Exception:  # noqa: BLE001
                detail = f"등록: {name} v{mv.version}"
        return detail
    except Exception as exc:  # noqa: BLE001
        return f"MLflow 기록 스킵: {exc}"


def _fail_running(run: PipelineRun, exc: Exception) -> None:
    for s in run.stages:
        if s.status == "running":
            s.status, s.detail = "failed", str(exc)[:300]
    run.status = "failed"
