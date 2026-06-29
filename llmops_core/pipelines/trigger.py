"""학습 파이프라인 수동/프로그램 트리거 — Argo Events 웹훅으로 발동.

data_build 스텝 완료 후(또는 운영자가) 본 CLI로 웹훅을 쳐서 DAG를 제출한다.
선언적 트리거(argo-events.yaml)와 동일 진입점을 코드에서도 호출 가능하게 한다.
"""

from __future__ import annotations

import argparse

import httpx


def fire(
    endpoint: str,
    *,
    data_version: str,
    base_model: str,
    tenant: str,
    timeout: float = 10.0,
) -> int:
    """Argo Events 웹훅(EventSource)에 데이터셋 준비 이벤트를 전송. HTTP 상태코드 반환."""
    resp = httpx.post(
        endpoint.rstrip("/") + "/dataset-ready",
        json={"data_version": data_version, "base_model": base_model, "tenant": tenant},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.status_code


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://dataset-webhook.llmops:12000")
    p.add_argument("--data-version", required=True)
    p.add_argument("--base-model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B")
    p.add_argument("--tenant", default="shared")
    args = p.parse_args(argv)

    code = fire(
        args.endpoint,
        data_version=args.data_version,
        base_model=args.base_model,
        tenant=args.tenant,
    )
    print(f"triggered (HTTP {code}) data_version={args.data_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
