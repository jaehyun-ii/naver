"""DAG 단계 0b — L2 품질 게이트: (선택)정제 + 검증. 미달 시 비0 종료로 DAG 중단.

design v3 §4.2/§7.1: PII/언어/근사중복 정제 후 네이티브 규칙(+GE)로 검증.
"쓰레기 입력, 쓰레기 모델"을 학습 진입 전에 차단한다.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-key", default="clean/records.jsonl")
    p.add_argument("--out-key", default="validated/records.jsonl")
    p.add_argument("--purpose", default="datasets")
    p.add_argument("--mask-pii", action="store_true")
    p.add_argument("--pii-lang", default="en")
    p.add_argument("--drop-near-dup", action="store_true")
    p.add_argument("--use-ge", action="store_true")  # Great Expectations 보강
    args = p.parse_args(argv)

    from llmops_core.common.errors import DataQualityFailed
    from llmops_core.common.schemas import TextRecord
    from llmops_core.ingestion import read_records, write_records
    from llmops_core.quality import validate_records

    records = read_records(args.purpose, args.in_key, TextRecord)

    # 정제(선택)
    from llmops_core.quality.clean import (
        annotate_language,
        drop_near_duplicates,
        mask_records,
    )

    records = annotate_language(records)
    if args.mask_pii:
        records = mask_records(records, lang=args.pii_lang)
    if args.drop_near_dup:
        records, removed = drop_near_duplicates(records)
        print(f"근사중복 {removed}건 제거")

    # 검증 게이트
    try:
        report = validate_records(records)
        if args.use_ge:
            from llmops_core.quality.validate import validate_with_great_expectations

            validate_with_great_expectations(records)
    except DataQualityFailed as exc:
        print(f"[QUALITY GATE FAILED] {exc}", file=sys.stderr)
        return 1

    uri = write_records(args.purpose, args.out_key, records)
    print(f"품질 통과 {report.num_records}건 (dup_ratio={report.stats.get('dup_ratio')}) -> {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
