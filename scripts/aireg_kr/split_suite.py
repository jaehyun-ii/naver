"""스위트 ACCEPT 산출물 → SFT 학습/검증 분할 (트랙 층화·결정적).

suite_v5 본 배치(트랙별 ACCEPT 125 × 8 = 1000) 완주 후 실행:

    PYTHONPATH=. python3 scripts/aireg_kr/split_suite.py --dir data_aireg/suite_v5

- 트랙마다 question_id 해시 정렬로 결정적 선별(재실행 시 동일 분할) 후
  앞 val-per-track건을 검증, 나머지를 학습으로 배정 → 기본 100/25 × 8트랙
  = train 800 / val 200.
- 쿼터 초과분(병렬 경쟁으로 125를 살짝 넘긴 행)은 해시 순서에서 잘리고,
  미달 트랙은 있는 만큼 4:1 비율로 축소 배정(강제 실패 대신 리포트에 기록).
- 산출: sft_train.jsonl / sft_val.jsonl (split 필드 부여) / split_report.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from scripts.aireg_kr.build_suite_qa import TRACK_FILES, stable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="스위트 산출 디렉터리")
    ap.add_argument("--per-track", type=int, default=125)
    ap.add_argument("--val-per-track", type=int, default=25)
    args = ap.parse_args()
    d = Path(args.dir)

    by_track: dict[str, list[dict]] = defaultdict(list)
    for f in glob.glob(str(d / "*_qa.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") == "ACCEPT":
                by_track[r["track"]].append(r)

    train, val, leftovers = [], [], []
    report: dict[str, dict] = {}
    for tr in sorted(TRACK_FILES):
        rows = sorted(by_track.get(tr, []), key=lambda r: stable(r["question_id"]))
        take = rows[: args.per_track]
        leftovers += rows[args.per_track:]  # 쿼터 초과분 — 부족 트랙 보충 풀
        n_val = (args.val_per_track if len(take) >= args.per_track
                 else max(1, round(len(take) * args.val_per_track
                                   / args.per_track)) if take else 0)
        val_rows, train_rows = take[:n_val], take[n_val:]
        val += val_rows
        train += train_rows
        report[tr] = {"accept": len(rows), "used": len(take),
                      "train": len(train_rows), "val": len(val_rows),
                      "shortfall": max(0, args.per_track - len(rows))}

    # 미달 트랙이 있으면 초과분 풀에서 결정적으로 보충 — 총량(기본 800/200)을
    # 우선 보장(층화가 약간 깨지는 것은 리포트에 기록).
    total_train = args.per_track * len(TRACK_FILES) - args.val_per_track * len(TRACK_FILES)
    total_val = args.val_per_track * len(TRACK_FILES)
    leftovers.sort(key=lambda r: stable(r["question_id"]))
    comp = {"val": 0, "train": 0}
    while len(val) < total_val and leftovers:
        val.append(leftovers.pop(0))
        comp["val"] += 1
    while len(train) < total_train and leftovers:
        train.append(leftovers.pop(0))
        comp["train"] += 1

    for name, rows, split in [("sft_train.jsonl", train, "train"),
                              ("sft_val.jsonl", val, "val")]:
        with open(d / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "split": split}, ensure_ascii=False)
                        + "\n")

    summary = {"train": len(train), "val": len(val),
               "compensated_from_surplus": comp, "tracks": report}
    (d / "split_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
