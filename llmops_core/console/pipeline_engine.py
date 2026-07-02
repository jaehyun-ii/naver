"""파이프라인 실행 엔진(콘솔) — 데이터·학습·평가·승인·배포를 운영 실행.

DAG: data-ingest → data-quality → data-build → finetune → evaluate
     → release-gate → release-approval → register → convert → deploy

finetune/convert/deploy는 `RealExecutor`(docker)로 실제 실행, register는 MLflow에 기록.
GPU 학습이 분 단위라 기본은 백그라운드 스레드로 진행하고 프론트가 /runs/{id}를 폴링한다
(background=False면 동기 실행 — 자동화·테스트용). 실행기는 RealExecutor를 모듈에서 가져오므로
테스트는 이를 대체(monkeypatch)해 docker 없이 배선 로직만 검증할 수 있다.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time


class PipelineCancelled(Exception):
    """실행 취소 또는 데드라인 초과로 파이프라인이 중단됨."""


# 배포 게이트 기본 임계값 — 동전던지기(0.5) 금지, 의미 있는 하한
_DEFAULT_GATE = {"answer_match": 0.7, "preference_accuracy": 0.7}

# 실행 취소 플래그(run_id) — 라우터가 cancel_run()으로 설정, 단계 사이에서 확인
_cancel_flags: set[str] = set()


def cancel_run(run_id: str) -> None:
    """진행 중인 run에 취소 요청(다음 단계 경계에서 중단)."""
    _cancel_flags.add(run_id)


def _run_deadline(body: "RunPipelineBody", created_at: float | None) -> float | None:
    """body.max_runtime_s(있으면)로 벽시계 데드라인 계산(없으면 None=무제한)."""
    max_s = getattr(body, "max_runtime_s", None)
    if max_s and created_at:
        try:
            return created_at + float(max_s)
        except (TypeError, ValueError):
            return None
    return None


def _guard(run: "PipelineRun", deadline: float | None) -> None:
    """단계 경계 가드 — 취소 요청/데드라인 초과 시 중단."""
    if run.id in _cancel_flags:
        raise PipelineCancelled("사용자 취소 요청")
    if deadline is not None and time.time() > deadline:
        raise PipelineCancelled("실행 시간 초과(deadline)")


def _bg_default() -> bool:
    """기본 백그라운드 실행 여부. 테스트는 LLMOPS_PIPELINE_BACKGROUND=0으로 동기화."""
    return os.environ.get("LLMOPS_PIPELINE_BACKGROUND", "1") != "0"

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

# 단계 정의(순서 = DAG 실행 순서)
_STAGE_DEFS = [
    ("data-ingest", "L1 적재·정규화·중복제거"),
    ("data-quality", "L2 품질 게이트"),
    ("data-build", "L3 데이터셋 빌드·버전"),
    ("finetune", "L4 학습(SFT/LoRA)"),
    ("evaluate", "L5 평가·자동 게이트"),
    ("release-gate", "승인 요청 생성"),
    ("release-approval", "릴리스 승인 대기"),
    ("register", "MLflow 등록·승격"),
    ("convert", "L6 LoRA merge 변환"),
    ("deploy", "배포(서빙 교체·라우팅)"),
]


def _executor(svc=None):
    """운영 실행기. 테스트는 `llmops_core.console.executor.RealExecutor`를 대체해 주입."""
    from llmops_core.console.executor import RealExecutor

    return RealExecutor()

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


def _new_run(name: str) -> PipelineRun:
    stages = [PipelineStage(name=n, title=t) for n, t in _STAGE_DEFS]
    return PipelineRun(
        id="run-" + secrets.token_urlsafe(6), name=name, stages=stages,
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


def _demo_mode(body: RunPipelineBody) -> bool:
    """내장 샘플(canned) 데이터 사용을 명시적으로 요청했는지 — dev/demo 플래그."""
    return bool(getattr(body, "demo", False)) or os.environ.get("LLMOPS_PIPELINE_DEMO") == "1"


def _strict_mode() -> bool:
    """운영 STRICT 모드 — 빈 페이로드에 샘플 대체를 금지(명시적 demo만 허용).

    기존 1-클릭 데모/테스트 계약을 보존하기 위해 기본은 비활성이며,
    운영에서는 LLMOPS_PIPELINE_STRICT=1로 켠다.
    """
    return os.environ.get("LLMOPS_PIPELINE_STRICT") == "1"


def _incumbent_metric(svc, served_name: str | None, metric: str,
                      *, exclude_run_id: str) -> float | None:
    """현재 프로덕션(같은 served_name) 후보의 최근 성공 run에 기록된 metric 값(없으면 None)."""
    if not served_name:
        return None
    for r in svc.pipelines.list():  # 최신순
        if r.id == exclude_run_id or r.status != "succeeded":
            continue
        if r.artifacts.get("served_name") != served_name:
            continue
        raw = r.artifacts.get("metrics_json")
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        val = data.get(metric)
        if isinstance(val, (int, float)):
            return float(val)
    return None


# ── 공개 API ──
def start_run(svc, body: RunPipelineBody, *, background: bool | None = None) -> PipelineRun:
    """release-gate까지 진행하고 승인 대기에 둔다.

    기본(background=True): GPU 학습이 분 단위라 백그라운드 스레드로 진행하고 호출은 running
    상태로 즉시 반환(프론트가 폴링). background=False면 동기 실행(자동화·테스트) — 반환 시 waiting.
    """
    if background is None:
        background = _bg_default()
    run = _new_run(body.name)
    svc.pipelines.add(run)
    if background:
        run.status = "running"
        threading.Thread(target=_pre_approval, args=(svc, run, body), daemon=True).start()
    else:
        _pre_approval(svc, run, body)
    return run


def resume_run(svc, run_id: str, *, background: bool | None = None) -> PipelineRun:
    """승인 결과를 반영해 register→convert→deploy를 진행(또는 반려 시 스킵)."""
    if background is None:
        background = _bg_default()
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
    if background:
        if run.status != "post-running":  # 중복 재개 방지
            run.status = "post-running"
            threading.Thread(target=_post_approval, args=(svc, run), daemon=True).start()
    else:
        _post_approval(svc, run)
    return run


# ── 백그라운드 실행부 ──
def _pre_approval(svc, run: PipelineRun, body: RunPipelineBody) -> None:
    deadline = _run_deadline(body, run.created_at)
    try:
        is_dpo = body.method == "dpo"
        use_dora = body.pet == "dora"
        pet_label = "DoRA" if use_dora else "LoRA"
        demo = _demo_mode(body)
        has_data = bool(body.records or body.labeled or body.preference)

        # 입력 검증 — 빈 페이로드에 canned 샘플을 '조용히' 쓰지 않는다.
        # STRICT 모드(운영)에서는 명시적 demo 플래그 없이는 거부.
        if not has_data:
            if _strict_mode() and not demo:
                raise ValueError(
                    "빈 파이프라인 페이로드: records/labeled/preference 중 하나가 필요합니다 "
                    "(데모 실행은 demo=true 또는 LLMOPS_PIPELINE_DEMO=1)")
            records_in = _SAMPLE_RECORDS
            labeled = _SAMPLE_LABELED
            pref_in = _SAMPLE_PREFERENCE
            run.artifacts["demo_data"] = "1"  # 샘플 사용을 명시적으로 표시(비-silent)
        else:
            records_in = body.records or []
            labeled = body.labeled or []
            pref_in = body.preference or []
        run.artifacts["served_name"] = body.served_name
        # 실제 학습 파라미터 스냅샷(MLflow 기록용) — 하드코딩 대신 run body 반영
        run.artifacts["params_json"] = json.dumps({
            "base_model": body.model_ref, "method": body.method, "pet": body.pet,
            "qlora": bool(getattr(body, "qlora", False)),
            "lr": getattr(body, "lr", None), "lora_r": getattr(body, "lora_r", 16),
            "max_steps": body.train_max_steps, "epochs": body.train_epochs,
        }, ensure_ascii=False)

        _guard(run, deadline)
        # L1 적재·정규화·중복제거·근사중복·PII (real). 원천 레코드가 없으면 스킵.
        records = [TextRecord(**r) for r in records_in]
        if records:
            _stage(run, "data-ingest").status = "running"
            records = normalize_records(records)
            records, removed = exact_dedup(records)
            near_removed = 0
            try:  # 근사중복 제거(datasketch) — 미설치/미가용이면 건너뜀(graceful)
                from llmops_core.quality.clean import drop_near_duplicates

                records, near_removed = drop_near_duplicates(records)
            except Exception:  # noqa: BLE001  OptionalDependency 등
                pass
            try:  # PII 마스킹 — 출력/기록 전 민감정보 제거
                from llmops_core.quality.clean import mask_records

                records = mask_records(records)
            except Exception:  # noqa: BLE001
                pass
            st = _stage(run, "data-ingest")
            st.status = "succeeded"
            st.detail = f"{len(records)}건 (exact {removed}·near {near_removed} 제거·PII 마스킹)"

            _guard(run, deadline)
            # L2 품질 게이트 (real) — 언어감지·필터 + 규칙 검증
            _stage(run, "data-quality").status = "running"
            lang_note = ""
            try:  # 언어 주석·필터(fasttext) — 모델 미설정이면 원본 유지(graceful)
                from llmops_core.quality.clean import annotate_language, filter_by_language

                records = annotate_language(records)
                records, dropped = filter_by_language(records, ["ko", "en"])
                if dropped:
                    lang_note = f" · lang제외 {dropped}"
            except Exception:  # noqa: BLE001
                pass
            try:
                from llmops_core.quality import validate_records

                report = validate_records(records)
                st = _stage(run, "data-quality")
                st.status = "succeeded"
                st.detail = f"통과 · dup_ratio={report.stats.get('dup_ratio')}{lang_note}"
            except DataQualityFailed as exc:
                st = _stage(run, "data-quality")
                st.status, st.detail = "failed", str(exc)
                run.status = "failed"
                return
        else:
            for n in ("data-ingest", "data-quality"):
                st = _stage(run, n)
                st.status, st.detail = "skipped", "원천 레코드 미제공 — 큐레이션 데이터 직접 사용"

        _guard(run, deadline)
        # L3 데이터셋 빌드·버전 (real) — 방식별 학습행/평가케이스/매니페스트 구성
        _stage(run, "data-build").status = "running"
        if is_dpo:
            pref = pref_in
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

        _guard(run, deadline)
        # L4 학습 (SFT/DPO · LoRA/DoRA) — HPO best_params가 있으면 주입
        ft = _stage(run, "finetune")
        ft.status = "running"
        ex = _executor(svc)
        ft_kwargs: dict = {}
        best_params = _resolve_hpo_params(svc, body)
        if best_params:
            ft_kwargs["hyperparams"] = best_params
        if getattr(body, "qlora", False):
            ft_kwargs["qlora"] = True
        adapter = ex.finetune(
            run.id, train_rows, base_model=None, method=body.method, use_dora=use_dora,
            max_steps=body.train_max_steps, epochs=body.train_epochs, **ft_kwargs,
        )
        run.artifacts["adapter"] = adapter
        run.loss_history = ex.last_loss  # 차트용 step별 loss
        hpo_note = " · HPO best_params 적용" if best_params else ""
        ft.detail = (f"{body.method.upper()}/{pet_label}(bf16) · "
                     f"steps={body.train_max_steps} → {adapter}{hpo_note}")
        ft.status = "succeeded"

        _guard(run, deadline)
        # L5 평가·자동 게이트 — 학습 어댑터로 평가셋을 실제 추론·채점 (judge 불필요)
        ev = _stage(run, "evaluate")
        ev.status = "running"
        task = "preference" if is_dpo else "reference"
        res = _executor(svc).evaluate(
            run.id, eval_cases, task=task,
            prompt_name=body.prompt_name, prompt_label=body.prompt_label,
            use_rag=body.use_rag, rag_top_k=body.rag_top_k,
        )
        metrics = res["metrics"]
        num_cases = res.get("num_cases", len(eval_cases))
        applied = []
        if res.get("prompt"):
            applied.append(f"프롬프트={res['prompt']}")
        if res.get("rag"):
            applied.append(f"RAG={res['rag']}")

        # 게이트 임계값: body.gate_thresholds 우선, 없으면 의미 있는 기본값(코인플립 0.5 아님)
        gate_metric = "preference_accuracy" if is_dpo else "answer_match"
        override = getattr(body, "gate_thresholds", None)
        thresholds = dict(override) if override else {gate_metric: _DEFAULT_GATE[gate_metric]}
        gate = GatePolicy(thresholds=thresholds)
        passed = gate.passes(metrics)

        # must-beat-incumbent — 현 프로덕션(같은 served_name) 대비 회귀면 승격 차단
        beat_note = ""
        incumbent = _incumbent_metric(svc, body.served_name, gate_metric, exclude_run_id=run.id)
        cand = metrics.get(gate_metric)
        if incumbent is not None:
            margin = float(getattr(body, "beat_margin", 0.0) or 0.0)
            if cand is None or cand < incumbent + margin:
                passed = False
                beat_note = (f"incumbent 미달({gate_metric} {cand} < "
                             f"{incumbent}+{margin})")
            else:
                beat_note = f"incumbent 초과({gate_metric} {cand} ≥ {incumbent}+{margin})"
        else:
            beat_note = "incumbent 없음 — 절대 임계값만 적용"

        result = EvalResult(
            suite="pipeline", model_ref=body.model_ref, metrics=metrics, passed=passed,
            num_cases=num_cases,
        )
        # 구조화 메트릭(JSON) 영속 + 하위호환 표시 문자열
        run.artifacts["metrics_json"] = json.dumps(metrics, ensure_ascii=False)
        run.artifacts["metrics"] = ", ".join(f"{k}={v}" for k, v in metrics.items())
        applied.append(beat_note)
        ev.status = "succeeded" if passed else "failed"
        ev.detail = " · ".join([*applied, run.artifacts["metrics"]])
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
    except PipelineCancelled as exc:
        _cancel_running(run, exc)
    except Exception as exc:  # noqa: BLE001
        _fail_running(run, exc)
    finally:
        _cancel_flags.discard(run.id)
        svc.pipelines.save(run)  # 영속(HA): 터미널/대기 상태 스냅샷


def _post_approval(svc, run: PipelineRun) -> None:
    try:
        _guard(run, None)
        # register — MLflow 기록·모델 레지스트리 등록·승격
        rs = _stage(run, "register")
        rs.status = "running"
        rs.detail = _register_real(run)
        rs.status = "succeeded"

        _guard(run, None)
        # convert — LoRA 병합
        cs = _stage(run, "convert")
        cs.status = "running"
        merged = _executor(svc).merge(run.id)
        run.artifacts["merged"] = merged
        cs.detail = f"merge_and_unload → {merged}"
        cs.status = "succeeded"

        _guard(run, None)
        # deploy — 서빙 교체 + 게이트웨이 라우팅. 자동(auto-retrain) 경로는 카나리로.
        ds = _stage(run, "deploy")
        ds.status = "running"
        served = run.artifacts.get("served_name", "hcx-seed-tuned")
        ex = _executor(svc)
        canary = run.artifacts.get("deploy_mode") == "canary"
        if canary and hasattr(ex, "deploy_canary"):
            out = ex.deploy_canary(run.id, served)
            url = out.get("url") if isinstance(out, dict) else out
            run.artifacts["serve_url"] = str(url)
            ds.detail = f"{served} 카나리 배포(weighted) → {url}"
        else:
            url = ex.deploy(run.id, served)
            run.artifacts["serve_url"] = str(url)
            mode = "카나리 미지원 폴백 " if canary else ""
            ds.detail = f"{mode}{served} 서빙 + 게이트웨이 라우팅 → {url}"
        ds.status = "succeeded"

        run.status = "succeeded"
    except PipelineCancelled as exc:
        _cancel_running(run, exc)
    except Exception as exc:  # noqa: BLE001
        _fail_running(run, exc)
    finally:
        _cancel_flags.discard(run.id)
        svc.pipelines.save(run)  # 영속(HA): 배포 완료/실패 스냅샷


def _resolve_hpo_params(svc, body: RunPipelineBody) -> dict | None:
    """HPO best_params 해석 — body.hpo_id로 레지스트리 조회 또는 body.hyperparams 직접."""
    hpo_id = getattr(body, "hpo_id", None)
    if hpo_id:
        try:
            rec = svc.hpo.get(hpo_id)
        except Exception:  # noqa: BLE001
            rec = None
        if rec and rec.get("best_params"):
            return dict(rec["best_params"])
    direct = getattr(body, "hyperparams", None)
    return dict(direct) if direct else None


def _register_real(run: PipelineRun) -> str:
    """MLflow에 params·metrics·어댑터를 기록하고, 모델 레지스트리에 등록·승격(alias)."""
    try:
        from llmops_core.tracking import ExperimentTracker

        tracker = ExperimentTracker(experiment="console-pipeline")
        mlflow = tracker.mlflow
        # 실제 run 파라미터 기록(하드코딩 금지) — pre_approval에서 스냅샷한 값
        try:
            params = json.loads(run.artifacts.get("params_json") or "{}")
        except (ValueError, TypeError):
            params = {}
        params = {k: v for k, v in params.items() if v is not None} or {
            "base_model": "unknown", "method": "unknown"}
        adapter = run.artifacts.get("adapter")
        with tracker.run(params, data_ver=run.artifacts.get("fingerprint")) as active:
            try:
                metrics = json.loads(run.artifacts.get("metrics_json") or "{}")
            except (ValueError, TypeError):
                metrics = {}
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(f"eval.{k}", float(v))
                except (ValueError, TypeError):
                    pass
            if adapter:
                mlflow.log_artifacts(adapter, artifact_path="adapter")
            run_id = active.info.run_id
            artifact_uri = active.info.artifact_uri

        # 모델 레지스트리 등록 — 저수준 create_model_version(source=아티팩트)로
        # MLflow 2.x/3.x 모두 호환(3.x의 logged-model 요구를 우회). + production alias 승격.
        name = run.artifacts.get("served_name", "hcx-seed-tuned")
        detail = "MLflow 기록(params·metrics·adapter)"
        if adapter:
            try:
                client = mlflow.tracking.MlflowClient()
                try:
                    client.create_registered_model(name)
                except Exception:  # noqa: BLE001  이미 존재
                    pass
                mv = client.create_model_version(
                    name=name, source=f"{artifact_uri}/adapter", run_id=run_id)
                run.artifacts["model_version"] = f"{name} v{mv.version}"
                detail = f"등록: {name} v{mv.version}"
                try:
                    client.set_registered_model_alias(name, "production", mv.version)
                    detail = f"등록·승격: {name} v{mv.version} @production"
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001  모델등록만 실패 — 기록은 유지
                detail = f"MLflow 기록(metrics) · 모델등록 스킵({str(exc)[:60]})"
        return detail
    except Exception as exc:  # noqa: BLE001
        return f"MLflow 기록 스킵: {exc}"


def _fail_running(run: PipelineRun, exc: Exception) -> None:
    marked = False
    for s in run.stages:
        if s.status == "running":
            s.status, s.detail = "failed", str(exc)[:300]
            marked = True
    if not marked:  # 단계 시작 전 실패(예: 입력 검증) — 첫 대기 단계에 사유 기록
        first = next((s for s in run.stages if s.status == "pending"), None)
        if first is not None:
            first.status, first.detail = "failed", str(exc)[:300]
    run.status = "failed"


def _cancel_running(run: PipelineRun, exc: Exception) -> None:
    """취소/데드라인 초과 — 진행 중 단계를 취소 표시하고 run을 실패로 종료."""
    for s in run.stages:
        if s.status == "running":
            s.status, s.detail = "cancelled", str(exc)[:300]
    run.status = "failed"
