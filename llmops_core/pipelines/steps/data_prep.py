"""DAG 단계 1 — DVC로 데이터 버전 pull (S3 백본)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-version", default="HEAD")
    p.add_argument("--path", default="data/train.jsonl")
    args = p.parse_args(argv)

    from llmops_core.tracking import data_version

    url = data_version(args.path)  # DVC가 가리키는 버전 URL/해시
    print(f"data_version({args.data_version}) -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
