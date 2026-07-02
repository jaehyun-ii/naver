"""운영 e2e 실행기 — GPU/배포 단계를 docker로 오케스트레이션 (H100/H200 단일·다중 GPU).

콘솔 파이프라인의 finetune/convert/deploy 단계를 **실제로** 수행한다:
- finetune: `llmops/train` 컨테이너에서 trl+peft LoRA SFT → 어댑터
- convert : `llmops/train` 컨테이너에서 merge_lora → 병합 safetensors
- deploy  : vLLM 컨테이너로 병합모델 서빙 + model_list.yaml 갱신 + 게이트웨이 재기동

전제: 콘솔이 **docker에 접근 가능한 호스트**에서 실행(README의 `uvicorn ... console.app`).
컨테이너로 콘솔을 띄울 경우 `/var/run/docker.sock` 마운트 필요.
설정은 환경변수 `LLMOPS_EXEC__*` 로 주입(기본값은 이 박스 기준).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _env(key: str, default: str) -> str:
    return os.environ.get(f"LLMOPS_EXEC__{key}", default)


@dataclass
class ExecutorConfig:
    """실행기 설정. env `LLMOPS_EXEC__*` 로 주입(인스턴스 생성 시점에 읽음)."""

    # 호스트 경로(컨테이너에 마운트)
    work_dir: str = field(default_factory=lambda: _env("WORK_DIR", str(Path.cwd() / ".runs")))
    hf_cache: str = field(
        default_factory=lambda: _env("HF_CACHE", str(Path.home() / ".cache/huggingface/hub"))
    )
    repo_dir: str = field(
        default_factory=lambda: _env("REPO_DIR", str(Path.cwd() / "llmops_core"))
    )
    config_path: str = field(
        default_factory=lambda: _env("CONFIG_PATH", str(Path.cwd() / "config/model_list.yaml"))
    )
    train_image: str = field(default_factory=lambda: _env("TRAIN_IMAGE", "llmops/train:latest"))
    serve_image: str = field(
        default_factory=lambda: _env("SERVE_IMAGE", "llmops/hf-serving:latest")
    )
    # 서빙 백엔드: vllm(기본·프로덕션 처리량, PagedAttention·연속배칭) | transformers(대체)
    serve_backend: str = field(default_factory=lambda: _env("SERVE_BACKEND", "vllm"))
    vllm_image: str = field(default_factory=lambda: _env("VLLM_IMAGE", "vllm/vllm-openai:latest"))
    base_model: str = field(
        default_factory=lambda: _env(
            "BASE_MODEL", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
        )
    )
    serve_port: int = field(default_factory=lambda: int(_env("SERVE_PORT", "8020")))
    serve_container: str = field(
        default_factory=lambda: _env("SERVE_CONTAINER", "llmops-tuned-serving")
    )
    gateway_container: str = field(
        default_factory=lambda: _env("GATEWAY_CONTAINER", "llmops-infra-gateway-1")
    )
    timeout_s: int = field(default_factory=lambda: int(_env("TIMEOUT_S", "1800")))


class ExecutorError(RuntimeError):
    pass


def _parse_loss_curve(stdout: str) -> list[dict]:
    """학습 컨테이너 stdout의 `{'loss': X, ... 'epoch': Y}` 줄에서 step별 loss 추출(차트용)."""
    import ast
    import re

    points: list[dict] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if "'loss'" not in line or not line.startswith("{"):
            continue
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            m = re.search(r"'loss':\s*([0-9.]+)", line)
            if not m:
                continue
            d = {"loss": float(m.group(1))}
        if isinstance(d, dict) and "loss" in d:
            points.append({"step": len(points) + 1, "loss": round(float(d["loss"]), 4),
                           "epoch": d.get("epoch")})
    return points


class RealExecutor:
    """docker CLI를 통해 실제 학습/병합/배포를 실행한다."""

    def __init__(self, cfg: ExecutorConfig | None = None) -> None:
        self.cfg = cfg or ExecutorConfig()
        self.last_loss: list[dict] = []  # 직전 finetune의 step별 loss(차트용)
        Path(self.cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # ── 내부 유틸 ──
    def _run(self, cmd: list[str], *, timeout: int | None = None) -> str:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout or self.cfg.timeout_s
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-2000:]
            raise ExecutorError(f"$ {' '.join(cmd[:6])} … → rc={proc.returncode}\n{tail}")
        return proc.stdout

    def _wait_ready(self, container: str, host_flag: list[str], *,
                    port: int = 8000, timeout: int = 300, interval: float = 3.0) -> None:
        """서빙 컨테이너가 모델 로드를 마치고 /health 200을 줄 때까지 폴링.

        컨테이너 내부 python으로 localhost:port/health를 확인(로컬·원격 docker_host 공통).
        컨테이너가 도중 종료되면 로그와 함께 즉시 실패한다. 준비 실패 시 ExecutorError.
        """
        import time

        deadline = time.time() + timeout
        probe = (
            "import sys,urllib.request;"
            f"urllib.request.urlopen('http://localhost:{port}/health',timeout=3);"
            "sys.exit(0)"
        )
        while time.time() < deadline:
            running = subprocess.run(
                ["docker", *host_flag, "inspect", "-f", "{{.State.Running}}", container],
                capture_output=True, text=True,
            )
            if running.returncode != 0 or running.stdout.strip() != "true":
                logs = subprocess.run(
                    ["docker", *host_flag, "logs", "--tail", "50", container],
                    capture_output=True, text=True,
                )
                raise ExecutorError(
                    f"서빙 컨테이너 '{container}'가 준비 전 종료됨\n"
                    f"{(logs.stderr or logs.stdout or '')[-1500:]}"
                )
            health = subprocess.run(
                ["docker", *host_flag, "exec", container, "python", "-c", probe],
                capture_output=True, text=True,
            )
            if health.returncode == 0:
                return
            time.sleep(interval)
        raise ExecutorError(
            f"서빙 컨테이너 '{container}' 준비 타임아웃({timeout}s) — /health 미응답")

    def _run_dir(self, run_id: str) -> Path:
        d = Path(self.cfg.work_dir) / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _common_mounts(self) -> list[str]:
        return [
            "-v", f"{self.cfg.hf_cache}:/hf/hub:ro",
            "-v", f"{self.cfg.repo_dir}:/app/llmops_core:ro",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
        ]

    def _gpu_run(self, tail_args: list[str], *, n_gpus: int = 1,
                 timeout: int | None = None) -> str:
        """스케줄러에서 GPU를 배정받아 `docker [-H node] run --gpus device=N ...` 실행 후 반납.

        다중 GPU/노드 환경에서 잡이 비어있는 디바이스에 핀닝되어 병렬 실행된다.
        """
        from llmops_core.orchestration.scheduler import scheduler

        sch = scheduler()
        lease = sch.acquire(n_gpus)
        try:
            base = ["docker"]
            if lease.docker_host:
                base += ["-H", lease.docker_host]
            base += ["run", "--rm", "--gpus", lease.device_arg]
            return self._run(base + tail_args, timeout=timeout)
        finally:
            sch.release(lease)

    # ── 1) 학습 (SFT/DPO · LoRA/DoRA) ──
    # 학습 CLI(llmops_core.training.run) 플래그로 변환 가능한 하이퍼파라미터 키.
    _HP_FLAGS: dict[str, str] = {
        "learning_rate": "--lr", "lr": "--lr",
        "lora_r": "--lora-r",
        "batch_size": "--batch-size", "per_device_train_batch_size": "--batch-size",
        "grad_accum": "--grad-accum", "gradient_accumulation_steps": "--grad-accum",
        "max_seq_length": "--max-seq-length",
    }

    def finetune(
        self, run_id: str, rows: list[dict], *, base_model: str | None = None,
        method: str = "sft", use_dora: bool = False,
        max_steps: int = -1, epochs: float = 1.0,
        hyperparams: dict | None = None, qlora: bool = False,
    ) -> str:
        """rows로 LoRA/DoRA 학습 → 어댑터 경로(호스트) 반환. 학습 loss 곡선은 self.last_loss에 저장.

        method="sft": rows={"messages":[...]}; method="dpo": rows={"prompt","chosen","rejected"}.
        hyperparams: HPO 등에서 나온 오버라이드(lr/lora_r/batch_size 등) — CLI 플래그로 전달.
        qlora: True면 --qlora(4bit, bitsandbytes)로 학습.
        """
        rd = self._run_dir(run_id)
        train_jsonl = rd / "train.jsonl"
        train_jsonl.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
        )
        adapter_host = rd / "adapter"
        tail = [
            *self._common_mounts(),
            "-v", f"{rd}:/work",
            self.cfg.train_image,
            "--method", method,
            "--train", "/work/train.jsonl",
            "--base-model", base_model or self.cfg.base_model,
            "--output-dir", "/work/adapter",
            "--max-steps", str(max_steps), "--epochs", str(epochs),
        ]
        if use_dora:
            tail.append("--dora")
        if qlora:
            tail.append("--qlora")
        for key, val in (hyperparams or {}).items():
            flag = self._HP_FLAGS.get(key)
            if flag is not None and val is not None:
                tail += [flag, str(val)]
        out = self._gpu_run(tail)
        self.last_loss = _parse_loss_curve(out)  # 차트용 step별 loss
        if not (adapter_host / "adapter_config.json").exists():
            raise ExecutorError("학습은 끝났으나 어댑터 산출물이 없습니다")
        return str(adapter_host)

    # ── 1.5) 평가(학습 어댑터로 실제 추론·채점) ──
    def evaluate(
        self, run_id: str, cases: list[dict], *, base_model: str | None = None,
        task: str = "reference", prompt_name: str | None = None,
        prompt_label: str = "prod", use_rag: bool = False, rag_top_k: int = 4,
    ) -> dict:
        """run_id의 어댑터로 cases를 평가 → {metrics, num_cases} 반환.

        task="reference": cases={question,expected} 생성·채점 (SFT).
        task="preference": cases={prompt,chosen,rejected} 선호정확도 (DPO).

        prompt_name 지정 시: 호스트에서 GitPromptStore의 해당 라벨 프롬프트를 해석해
        평가에 시스템 프롬프트로 주입(서빙과 동일 적용 — 평가=서빙 정합). 스토어는
        호스트에만 있으므로 컨테이너에 마운트하지 않고 텍스트로 전달한다.

        use_rag=True: 호스트 RAG 스토어를 run 디렉토리로 복사해 평가가 검색·컨텍스트
        주입을 거치게 한다(서빙이 RAG 기반일 때). 스토어가 없으면 빈 KB로 무영향.
        """
        rd = self._run_dir(run_id)
        if not (rd / "adapter" / "adapter_config.json").exists():
            raise ExecutorError("평가할 어댑터가 없습니다 (먼저 finetune)")
        eval_jsonl = rd / "eval.jsonl"
        eval_jsonl.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8"
        )
        prompt_args: list[str] = []
        if prompt_name:
            from llmops_core.prompts import GitPromptStore, PromptNotFound

            try:
                pr = GitPromptStore().by_label(prompt_name, prompt_label)
                prompt_args = ["--system-prompt", pr.template]
            except PromptNotFound:
                pass  # 프롬프트 미등록이면 프롬프트 없이 평가(기존 동작)
        rag_args: list[str] = []
        if use_rag:
            import shutil

            from llmops_core.common.config import get_settings

            host_store = Path(get_settings().rag.persist_path or "")
            rag_args = ["--rag", "--rag-top-k", str(rag_top_k)]
            if host_store.exists() and host_store.is_file():
                shutil.copy(host_store, rd / "rag_store")
                rag_args += ["--rag-store", "/work/rag_store"]
        tail = [
            *self._common_mounts(),
            "-v", f"{rd}:/work",
            "--entrypoint", "python3", self.cfg.train_image,
            "-m", "llmops_core.evaluation.local_eval",
            "--base", base_model or self.cfg.base_model,
            "--adapter", "/work/adapter",
            "--cases", "/work/eval.jsonl",
            "--task", task,
            "--out", "/work/metrics.json",
            *prompt_args,
            *rag_args,
        ]
        out = self._gpu_run(tail)
        # stdout 마지막 JSON 줄 파싱(견고하게)
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise ExecutorError(f"평가 메트릭 파싱 실패:\n{out[-500:]}")

    # ── 선호 벤치마크(어댑터 없이 base 모델 logprob 평가) ──
    def benchmark_preference(self, model: str, cases: list[dict], *,
                             run_id: str = "bench-pref") -> dict:
        """{prompt,chosen,rejected} 케이스를 base 모델로 logprob 평가 → preference_accuracy.

        선호 평가는 logprob이 필요해 게이트웨이가 아닌 컨테이너 in-process(local_eval)로 수행.
        어댑터 없이 model을 --base로 로드한다. {metrics, num_cases} 반환.
        """
        rd = self._run_dir(run_id)
        (rd / "eval.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
        tail = [
            *self._common_mounts(), "-v", f"{rd}:/work",
            "--entrypoint", "python3", self.cfg.train_image,
            "-m", "llmops_core.evaluation.local_eval",
            "--base", model, "--cases", "/work/eval.jsonl",
            "--task", "preference", "--out", "/work/metrics.json",
        ]
        out = self._gpu_run(tail)
        for line in reversed(out.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise ExecutorError(f"선호 벤치마크 파싱 실패:\n{out[-500:]}")

    # ── HPO(Optuna) ──
    def hpo(self, run_id: str, labeled: list[dict], eval_cases: list[dict], *,
            trials: int = 4, steps: int = 12, base_model: str | None = None) -> dict:
        """학습 컨테이너에서 Optuna HPO 실행 → {best_params, best_value, trials} 반환."""
        rd = self._run_dir(f"hpo-{run_id}")
        (rd / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in labeled), encoding="utf-8")
        (rd / "eval.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in eval_cases), encoding="utf-8")
        tail = [
            *self._common_mounts(), "-v", f"{rd}:/work",
            "--entrypoint", "python3", self.cfg.train_image,
            "-m", "llmops_core.tuning.run",
            "--train", "/work/train.jsonl", "--eval", "/work/eval.jsonl",
            "--base", base_model or self.cfg.base_model,
            "--trials", str(trials), "--steps", str(steps), "--out", "/work/hpo.json",
        ]
        self._gpu_run(tail)
        hpo_json = rd / "hpo.json"
        if not hpo_json.exists():
            raise ExecutorError("HPO 결과(hpo.json)가 없습니다")
        return json.loads(hpo_json.read_text())

    # ── 2) 병합(convert) ──
    def merge(self, run_id: str, *, base_model: str | None = None) -> str:
        """run_id의 어댑터를 베이스에 병합 → 병합모델 디렉토리 경로(호스트) 반환."""
        rd = self._run_dir(run_id)
        merged_host = rd / "merged"
        tail = [
            *self._common_mounts(),
            "-v", f"{rd}:/work",
            "--entrypoint", "python3", self.cfg.train_image,
            "-m", "llmops_core.registry.merge",
            "--base", base_model or self.cfg.base_model,
            "--adapter", "/work/adapter",
            "--output-dir", "/work/merged",
        ]
        self._gpu_run(tail)
        if not (merged_host / "model.safetensors").exists():
            raise ExecutorError("병합은 끝났으나 model.safetensors가 없습니다")
        return str(merged_host)

    # ── 3) 배포(deploy) ──
    def deploy(self, run_id: str, served_name: str) -> str:
        """병합모델을 hf-serving 컨테이너로 띄우고 게이트웨이 라우팅에 등록. 서빙 URL 반환."""
        rd = self._run_dir(run_id)
        merged_host = rd / "merged"
        if not (merged_host / "model.safetensors").exists():
            raise ExecutorError("배포할 병합모델이 없습니다 (먼저 convert)")

        # 서빙은 GPU를 장기 점유 → 스케줄러에서 영구 lease(재배포 시 이전 lease 해제)
        from llmops_core.orchestration.scheduler import scheduler

        sch = scheduler()
        lease = sch.hold(f"serve:{self.cfg.serve_container}", 1)
        host_flag = ["-H", lease.docker_host] if lease.docker_host else []

        # 기존 튜닝 서빙 컨테이너 정리 후 재기동
        subprocess.run(["docker", *host_flag, "rm", "-f", self.cfg.serve_container],
                       capture_output=True, text=True)
        image, serve_args = self._serving_spec(served_name)
        cmd = [
            "docker", *host_flag, "run", "-d", "--name", self.cfg.serve_container,
            "--gpus", lease.device_arg,
            "-v", f"{merged_host}:/model:ro",
            "-v", f"{self.cfg.repo_dir}:/app/llmops_core:ro",
            "-p", f"{self.cfg.serve_port}:8000",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            image, *serve_args,
        ]
        try:
            self._run(cmd, timeout=120)
            # 모델 로드 완료(/health)를 확인한 뒤에만 라우팅 등록 → 트래픽이 미준비 서버로 안 감
            self._wait_ready(self.cfg.serve_container, host_flag)
        except Exception:
            # 실패 시 컨테이너 정리 + GPU lease 반납(누수 방지)
            subprocess.run(["docker", *host_flag, "rm", "-f", self.cfg.serve_container],
                           capture_output=True, text=True)
            sch.release_tag(f"serve:{self.cfg.serve_container}")
            raise
        self._register_route(served_name)
        return f"http://host.docker.internal:{self.cfg.serve_port}/v1"

    def _serving_spec(self, served_name: str) -> tuple[str, list[str]]:
        """서빙 백엔드별 (이미지, 컨테이너 인자). vllm은 PagedAttention·연속배칭(H200 처리량)."""
        if self.cfg.serve_backend == "vllm":
            return self.cfg.vllm_image, [
                "--model", "/model", "--served-model-name", served_name,
                "--port", "8000", "--gpu-memory-utilization", "0.90",
                "--max-model-len", "4096",
            ]
        # 대체: transformers(hf_server) — vllm 미가용 환경용 레퍼런스 백엔드
        return self.cfg.serve_image, [
            "--model", "/model", "--served-name", served_name, "--port", "8000",
        ]

    def _stable_url(self) -> str:
        return f"http://host.docker.internal:{self.cfg.serve_port}/v1"

    def _canary_url(self) -> str:
        return f"http://host.docker.internal:{self.cfg.serve_port + 1}/v1"

    def _apply_routing(self, transform) -> None:
        """model_list에 transform(list)->list를 적용·기록하고 게이트웨이 재기동."""
        path = Path(self.cfg.config_path)
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        data["model_list"] = transform(data.get("model_list", []))
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        subprocess.run(["docker", "restart", self.cfg.gateway_container],
                       capture_output=True, text=True)

    def _register_route(self, served_name: str) -> None:
        """단일 stable 라우팅(weight 100)으로 등록하고 게이트웨이 재기동."""
        from llmops_core.serving.canary import set_stable

        self._apply_routing(lambda ml: set_stable(ml, served_name, self._stable_url()))

    # ── 카나리/블루그린 배포 ──
    def deploy_canary(self, run_id: str, served_name: str, *, weight: int = 10) -> dict:
        """병합모델을 카나리 서빙 컨테이너(별도 포트·GPU)로 띄우고 weight%만 라우팅.

        다중 GPU 필요(stable+canary 동시 서빙). 단일 GPU 환경에서는 GPU 부족으로 대기.
        """
        from llmops_core.orchestration.scheduler import scheduler
        from llmops_core.serving.canary import add_canary

        rd = self._run_dir(run_id)
        merged_host = rd / "merged"
        if not (merged_host / "model.safetensors").exists():
            raise ExecutorError("배포할 병합모델이 없습니다 (먼저 convert)")
        sch = scheduler()
        lease = sch.hold(f"serve:{self.cfg.serve_container}-canary", 1)
        host_flag = ["-H", lease.docker_host] if lease.docker_host else []
        cname = f"{self.cfg.serve_container}-canary"
        subprocess.run(["docker", *host_flag, "rm", "-f", cname],
                       capture_output=True, text=True)
        image, serve_args = self._serving_spec(served_name)
        cmd = [
            "docker", *host_flag, "run", "-d", "--name", cname, "--gpus", lease.device_arg,
            "-v", f"{merged_host}:/model:ro",
            "-v", f"{self.cfg.repo_dir}:/app/llmops_core:ro",
            "-p", f"{self.cfg.serve_port + 1}:8000",
            "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
            image, *serve_args,
        ]
        try:
            self._run(cmd, timeout=120)
            self._wait_ready(cname, host_flag)
        except Exception:
            subprocess.run(["docker", *host_flag, "rm", "-f", cname],
                           capture_output=True, text=True)
            sch.release_tag(f"serve:{self.cfg.serve_container}-canary")
            raise
        self._apply_routing(lambda ml: add_canary(
            ml, served_name, stable_base=self._stable_url(),
            canary_base=self._canary_url(), weight=weight))
        return {"served_name": served_name, "canary_url": self._canary_url(), "weight": weight}

    def promote_canary(self, served_name: str) -> dict:
        """카나리를 stable로 승격 — 라우팅 100% 전환, 이전 stable 컨테이너·GPU 정리."""
        from llmops_core.orchestration.scheduler import scheduler
        from llmops_core.serving.canary import promote

        self._apply_routing(lambda ml: promote(ml, served_name, canary_base=self._canary_url()))
        # 이전 stable 컨테이너 정리 + GPU 반납
        subprocess.run(["docker", "rm", "-f", self.cfg.serve_container],
                       capture_output=True, text=True)
        scheduler().release_tag(f"serve:{self.cfg.serve_container}")
        return {"served_name": served_name, "promoted": True}

    def rollback_canary(self, served_name: str) -> dict:
        """카나리 제거 — stable 100% 복귀, 카나리 컨테이너·GPU 정리.

        가드: 유효한 이전 stable이 없으면 롤백을 거부한다(rollback이 논리 모델을 통째로
        삭제해 라우팅이 사라지는 것을 방지). 이 경우 promote 또는 재배포로 stable을 세워야 한다.
        """
        from llmops_core.orchestration.scheduler import scheduler
        from llmops_core.serving.canary import rollback, rollout_status

        path = Path(self.cfg.config_path)
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        status = rollout_status(data.get("model_list", []), served_name)
        if not status.get("stable"):
            raise ExecutorError(
                f"'{served_name}' 롤백 불가 — 유효한 stable이 없습니다"
                " (promote 또는 재배포로 stable을 먼저 세우세요)")

        self._apply_routing(lambda ml: rollback(ml, served_name))
        subprocess.run(["docker", "rm", "-f", f"{self.cfg.serve_container}-canary"],
                       capture_output=True, text=True)
        scheduler().release_tag(f"serve:{self.cfg.serve_container}-canary")
        return {"served_name": served_name, "rolledback": True}


def docker_available() -> bool:
    """docker CLI 접근 가능 여부(파이프라인 실행 가능성 판정)."""
    try:
        return subprocess.run(
            ["docker", "version"], capture_output=True, text=True, timeout=10
        ).returncode == 0
    except Exception:  # noqa: BLE001
        return False
