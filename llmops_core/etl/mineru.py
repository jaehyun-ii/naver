"""MinerU 2.5 CLI 호출 — noksan_ax parser의 hybrid-http-client 실행부를 이식.

명령 형태(동일):
    mineru -p <pdf> -o <output_dir> -b hybrid-http-client -u <endpoint>
           --effort <high|medium> --image-analysis <bool> --formula <bool>
           --table <bool> --client-side-output-generation true [-m <method>]

FastAPI 서비스(진행률 heartbeat·취소·DB 아티팩트)는 벗겨내고, 순수 subprocess
실행만 남긴다. MinerU 자체는 stock이므로 이 플래그들은 정식 옵션이다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .config import EtlConfig


def mineru_cli_path() -> str | None:
    """설치된 mineru 실행파일 경로. venv/PATH 순으로 탐색."""
    candidate = Path(sys.executable).with_name("mineru")
    if candidate.exists():
        return str(candidate)
    return shutil.which("mineru")


def build_command(input_path: Path, output_dir: Path, cfg: EtlConfig, mineru_bin: str) -> list[str]:
    command = [
        mineru_bin,
        "-p", str(input_path),
        "-o", str(output_dir),
        "-b", "hybrid-http-client",
        "-u", cfg.endpoint.rstrip("/"),
        "--effort", cfg.effort,
        "--image-analysis", "true" if cfg.image_analysis else "false",
        "--formula", "true" if cfg.formula_enabled else "false",
        "--table", "true" if cfg.table_enabled else "false",
        "--client-side-output-generation", "true",
    ]
    # parse_method은 PDF에만 의미가 있다(원본 parser와 동일 조건)
    if cfg.parse_method and input_path.suffix.lower() == ".pdf":
        command.extend(["-m", cfg.parse_method])
    return command


def run_mineru(input_path: Path, output_dir: Path, cfg: EtlConfig) -> list[str]:
    """MinerU CLI를 실행해 output_dir에 산출물을 생성. 실행한 명령을 반환."""
    cfg.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    mineru_bin = mineru_cli_path()
    if not mineru_bin:
        raise RuntimeError(
            "MinerU CLI가 설치돼 있지 않습니다. `uv pip install 'mineru[pipeline]'` (또는 "
            "llmops/etl 컨테이너)로 설치하세요."
        )
    command = build_command(input_path, output_dir, cfg, mineru_bin)
    try:
        completed = subprocess.run(
            command,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=cfg.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"MinerU CLI timed out after {cfg.timeout_seconds}s") from exc
    except OSError as exc:
        raise RuntimeError(f"MinerU CLI failed to start: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip() or f"exit {completed.returncode}"
        raise RuntimeError(f"MinerU CLI failed: {detail[-2000:]}")
    return command
