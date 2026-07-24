#!/usr/bin/env python3
"""RAG 검색 평가용 골든셋 생성.

각 선급에서 **실질 내용이 있는 조**를 층화 표본추출하고, 서빙 중인 소형 LLM으로
그 조를 찾을 만한 검색 질문을 생성한다. 정답 라벨 = 출처 조(parent_chunk_id).
통합본은 제외(내용 dedup 생존자가 개별편이므로 정답 조도 개별편으로 통일).

    python scripts/build_goldenset.py [n_per_soc]  →  data_chunks/goldenset.jsonl
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import requests

OUT = Path("data_chunks/goldenset.jsonl")
LLM = "http://localhost:8000/v1/chat/completions"
MODEL = "hcx-seed-0_5b"
N_PER_SOC = int(sys.argv[1]) if len(sys.argv) > 1 else 14
SOCS = ["KR", "ClassNK", "DNV", "BV", "ABS", "IACS", "LR"]


def gen_query(title: str, body: str) -> str:
    prompt = ("다음 선급규칙 조항을 실무자가 찾으려 할 때 입력할 **짧은 검색 질문** 1개를 "
              "한국어로 만들어라. 조 번호는 넣지 말고 내용 중심으로, 한 문장. 질문만 출력.\n\n"
              f"[{title}]\n{body[:400]}")
    try:
        r = requests.post(LLM, json={"model": MODEL, "temperature": 0.3, "max_tokens": 48,
                                     "messages": [{"role": "user", "content": prompt}]}, timeout=30)
        q = r.json()["choices"][0]["message"]["content"].strip()
        q = re.sub(r"^(질문[:.]?\s*|검색\s*질문[:.]?\s*|-\s*)", "", q).strip().strip('"')
        return q.split("\n")[0][:120]
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    gold = []
    for soc in SOCS:
        # 개별편(통합본 제외) 파일에서 substantial parent 조 수집
        parents = []
        for fn in sorted(glob.glob(f"data_chunks/{soc}/*.jsonl")):
            if "통합본" in fn:
                continue
            rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
            kids = {}
            for c in rows:
                if c.get("chunk_level") == "child" and c.get("chunk_type") == "text":
                    kids.setdefault(c.get("parent_chunk_id"), []).append(c.get("content") or "")
            for c in rows:
                if c.get("chunk_level") != "parent":
                    continue
                body = " ".join(kids.get(c["chunk_id"], []))
                if len(body) < 80 or "참조】" in (c.get("content") or ""):   # 스텁·초단문 제외
                    continue
                parents.append((c, body))
        # 층화: 균등 stride 표본
        if not parents:
            continue
        step = max(1, len(parents) // N_PER_SOC)
        picked = parents[::step][:N_PER_SOC]
        for c, body in picked:
            title = f"{c.get('article_no','')} {c.get('article_title','')}".strip()
            q = gen_query(title, body)
            if len(q) < 6:
                continue
            gold.append({
                "query": q, "soc": soc,
                "relevant_article_id": c["chunk_id"],
                "article_no": str(c.get("article_no") or ""),
                "article_title": c.get("article_title") or "",
                "section_path": " > ".join(c.get("section_path") or []),
            })
            print(f"  [{soc}] {q[:52]}", flush=True)
    OUT.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in gold), encoding="utf-8")
    print(f"\n✓ 골든셋 {len(gold)}개 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
