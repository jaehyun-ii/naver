"""DAG 단계 5 — 배포 갱신.

Production 모델 버전을 서빙 매니페스트에 핀하고 Git에 커밋한다.
Argo CD가 변경을 감지해 vLLM 배포와 LiteLLM 라우팅을 동기화. 롤백은 git revert.
실제 커밋/푸시는 운영 자격으로 수행되므로 여기서는 매니페스트 패치만 수행한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="hcx-seed-3b")
    p.add_argument("--version", default="latest")
    p.add_argument("--manifest", default="deploy/serving/prod/values.yaml")
    args = p.parse_args(argv)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"# Argo CD가 동기화하는 서빙 값 (자동 생성)\n"
        f"model: {args.model_name}\n"
        f"modelVersion: {args.version}\n"
    )
    print(f"deploy manifest patched: {manifest} (Argo CD가 동기화)")
    print("→ git add/commit/push 후 Argo CD sync (운영 자격으로 수행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
