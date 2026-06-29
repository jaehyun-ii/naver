"""DAG 단계 0a — L1 적재/정제: 원천 정규화 + 정확중복 제거 → clean 버킷.

기본은 경량 경로(ingestion.text_ops, 의존성 없음). 진짜 빅데이터면 --engine spark.
입출력은 S3 JSONL(ObjectStore). 라벨링/검증 단계가 이 산출물을 이어받는다.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-key", default="source/raw.jsonl")  # raw 버킷 내 키
    p.add_argument("--out-key", default="clean/records.jsonl")
    p.add_argument("--purpose", default="datasets")  # 버킷 용도(env-domain-datasets)
    p.add_argument("--engine", choices=["light", "spark"], default="light")
    args = p.parse_args(argv)

    if args.engine == "spark":
        from llmops_core.ingestion.spark_jobs import IngestSparkConfig, normalize_and_dedup

        n = normalize_and_dedup(
            IngestSparkConfig(input_path=args.in_key, output_path=args.out_key)
        )
        print(f"[spark] normalized+deduped -> {args.out_key} ({n} rows)")
        return 0

    from llmops_core.common.schemas import TextRecord
    from llmops_core.ingestion import (
        exact_dedup,
        normalize_records,
        read_records,
        write_records,
    )

    records = read_records(args.purpose, args.in_key, TextRecord)
    records = normalize_records(records)
    records, removed = exact_dedup(records)
    uri = write_records(args.purpose, args.out_key, records)
    print(f"[light] {len(records)} records (정확중복 {removed}건 제거) -> {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
