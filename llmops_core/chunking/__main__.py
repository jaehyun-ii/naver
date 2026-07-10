"""CLI: content_list.json → 도메인 청크 JSONL (선급 자동 판별).

    python -m llmops_core.chunking INPUT_content_list.json -o out.jsonl [--family auto]

디렉터리를 주면 하위 *_content_list.json 을 일괄 처리(각자 family 자동 판별,
data_chunks/<stem>.jsonl 로 저장).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from . import FAMILIES, chunk_document, detect_family, write_jsonl


def _one(cl: Path, out: Path, family: str) -> None:
    fam, chunks = chunk_document(cl, family=family)
    write_jsonl(chunks, out)
    lv = Counter(c["chunk_level"] for c in chunks)
    ty = Counter(c["chunk_type"] for c in chunks)
    print(f"✓ [{fam}] {cl.name} → {out}  ({len(chunks)} chunks, {lv.get('parent',0)} articles)")
    print(f"    types: {dict(ty)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="도메인 청킹: content_list.json → chunks.jsonl")
    ap.add_argument("input", type=Path, help="content_list.json 또는 그 디렉터리")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="출력 JSONL(단일 입력). 디렉터리 입력 시 무시하고 data_chunks/로 저장")
    ap.add_argument("--family", default="auto", choices=("auto", *FAMILIES))
    ap.add_argument("--detect-only", action="store_true", help="선급 판별 결과만 출력")
    args = ap.parse_args()

    if args.input.is_dir():
        cls = sorted(args.input.rglob("*content_list.json"))
        cls = [p for p in cls if "raw" not in p.parts and not p.name.endswith("v2.json")]
    else:
        cls = [args.input]
    if not cls:
        print(f"content_list 없음: {args.input}")
        return 2

    if args.detect_only:
        for cl in cls:
            print(f"{detect_family(cl):10} ← {cl}")
        return 0

    out_root = Path("data_chunks")
    for cl in cls:
        out = args.output if (args.output and len(cls) == 1) else out_root / f"{cl.stem.replace('_content_list','')}_chunks.jsonl"
        _one(cl, out, args.family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
