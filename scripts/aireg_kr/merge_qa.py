"""트랙별 *_qa.jsonl → 단일 병합 파일(all_qa.jsonl) 생성.

정렬은 (track, 파일 내 순서)로 결정적. 배치 진행 중 재실행해도 안전(읽기 전용,
원자적 교체). 검수 사이드카(suite_verdicts.jsonl)는 question_id로 조인 가능.

    AIREG_OUT_DIR=<dir> python -m scripts.aireg_kr.merge_qa
"""
from __future__ import annotations

import json

from .build_suite_qa import TRACK_FILES
from .common import OUT_DIR, load_jsonl, log


def main() -> None:
    rows = []
    for tr in sorted(TRACK_FILES):
        fname, _ = TRACK_FILES[tr]
        part = load_jsonl(OUT_DIR / fname)
        rows += part
        if part:
            log(f"{tr}: {len(part)}행")
    out = OUT_DIR / "all_qa.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    acc = sum(1 for r in rows if r.get("status") == "ACCEPT")
    log(f"→ {out} ({len(rows)}행, ACCEPT {acc})")


if __name__ == "__main__":
    main()
