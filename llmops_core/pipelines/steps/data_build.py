"""DAG 단계 0c — L3 데이터셋: 라벨링 결과 → SFT 포맷·필터·분할·버전 고정.

Argilla 승인 라벨(또는 검증 통과 레코드)을 chat 포맷으로 변환하고, 결정적 분할 후
핑거프린트로 버전을 고정해 S3에 적재한다. 매니페스트 fingerprint가 이후 MLflow run에 연결된다.
"""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="domain-sft")
    p.add_argument("--argilla-dataset", default=None)  # 지정 시 Argilla에서 회수
    p.add_argument("--in-key", default="validated/records.jsonl")  # 또는 S3 레코드
    p.add_argument("--purpose", default="datasets")
    p.add_argument("--system", default=None)  # SFT system 프롬프트(선택)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--dvc-path", default=None)
    args = p.parse_args(argv)

    from llmops_core.dataset import (
        SplitConfig,
        filter_sft_by_length,
        split_sft,
        to_sft_examples,
    )
    from llmops_core.dataset.versioning import publish_split

    # 라벨 소스: Argilla 승인분 우선, 없으면 S3 검증 레코드(text→user, 정답 없음 케이스)
    if args.argilla_dataset:
        from llmops_core.quality.labeling import pull_labeled

        labeled = pull_labeled(args.argilla_dataset)
    else:
        from llmops_core.common.schemas import TextRecord
        from llmops_core.ingestion import read_records

        recs = read_records(args.purpose, args.in_key, TextRecord)
        labeled = [{"text": r.text, "response": r.metadata.get("response")} for r in recs]

    examples = filter_sft_by_length(to_sft_examples(labeled, system=args.system))
    train, val, test = split_sft(
        examples, SplitConfig(val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    )
    manifest = publish_split(
        args.name, "sft", train, val, test, purpose=args.purpose, dvc_path=args.dvc_path
    )
    print(json.dumps(manifest.model_dump(exclude_none=True), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
