"""[레거시 단독 실행용] 스위트 태스크 ⑤ — 상호참조 해석(⑤a) + 선급 간 비교(⑤b).

파이프라인 기본 흐름에서는 build_suite_qa가 상호참조(조당 1건 라우팅)와
선급 비교(--n-compare, 본 모듈의 gen_compare 재사용)를 담당한다.
rule_cards 기반 다건 확장이 필요할 때만 단독 실행하십시오.

⑤a 상호참조: 조문이 인용하는 타 조항("202.의 1항에 적합하도록")을 해석 — 두 조항을
   결합해야 답할 수 있는 QA. 소스: rule_cards의 조항 원문에서 참조 번호를 정규식 추출,
   같은 발행처·같은 문서에서 해석(resolve).
⑤b 선급 비교: cluster_map.jsonl(bge-m3 실측)의 선급 교차 클러스터에서 동일 요건의
   두 선급 조항 쌍을 뽑아 공통/차이 비교 QA. KR 포함 쌍 우선(한국어 답변 자연).
   chunk_id가 발행처 내 문서 간 중복으로 유일 해석이 안 되는 멤버는 제외.

evidence는 조항 2개(다중 evidence) — RAG 평가에서 두 청크 모두 검색해야 만점.

    python -m scripts.aireg_kr.build_crossref_qa [--n-compare 2]
      →  data_aireg/{crossref_qa,compare_qa}.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from . import prompts
from .common import (CHUNK_DIR, OUT_DIR, _is_requirement_parent, append_jsonl,
                     chat_json, glob_docs, load_done, load_jsonl, log, nfc)

CROSSREF_OUT = OUT_DIR / "crossref_qa.jsonl"
COMPARE_OUT = OUT_DIR / "compare_qa.jsonl"

# 한국어 조문의 타 조항 참조: "202.의 1항", "902. 및" 등 (3자리 조 번호)
KR_REF = re.compile(r"(?<![\d.])(\d{3})\.(?:의\s*(\d+)\s*항)?")


def sp(chunk: dict) -> str:
    v = chunk.get("section_path") or []
    return " > ".join(v) if isinstance(v, list) else str(v)


def evidence_entry(chunk: dict, quote: str = "") -> dict:
    return {"chunk_id": chunk["chunk_id"], "source_file": chunk.get("_source_file", ""),
            "section_path": sp(chunk), "quote": quote, "article_text": chunk["content"]}


def load_all_parents() -> list[dict]:
    rows = []
    for soc_dir in sorted(CHUNK_DIR.iterdir()):
        if not soc_dir.is_dir() or soc_dir.name.startswith("_"):
            continue
        for f in glob_docs(soc_dir, "*_chunks.jsonl"):
            for l in f.open(encoding="utf-8"):
                if '"parent"' not in l:
                    continue
                r = json.loads(l)
                if _is_requirement_parent(r, soc_dir.name, 100):
                    r["publisher"] = soc_dir.name
                    r["_source_file"] = nfc(f.stem)
                    rows.append(r)
    return rows


# ── ⑤a 상호참조 ─────────────────────────────────────────────────────────
def find_refs(rule: dict, parents: list[dict], limit: int = 2) -> list[dict]:
    """조항 원문이 참조하는 같은 문서 내 타 조(parent)를 해석."""
    own = str(rule.get("article_no") or "")
    refs = []
    for m in KR_REF.finditer(rule["article_text"]):
        no = m.group(1)
        if no == own or no in refs:
            continue
        refs.append(no)
    # 같은 문서 판별: source_file 우선, 없으면(구버전 rule card) KR chunk_id의
    # 편·장 접두("RULE_P5_C7_")로 같은 장 내에서 해석
    src = rule.get("source_file", "")
    prefix = rule["chunk_id"].rsplit("_S", 1)[0] + "_S" if rule["chunk_id"].startswith("RULE_") else None
    out = []
    for no in refs:
        cand = [p for p in parents
                if p.get("publisher") == rule.get("publisher", "KR")
                and str(p.get("article_no")) == no
                and p["chunk_id"] != rule["chunk_id"]
                and ((src and p.get("_source_file", "") == src)
                     or (not src and prefix and p["chunk_id"].startswith(prefix)))]
        if len(cand) == 1:
            out.append(cand[0])
        if len(out) >= limit:
            break
    return out


def gen_crossref(rule: dict, parents: list[dict], done: set[str]) -> None:
    for i, ref in enumerate(find_refs(rule, parents), start=1):
        qid = f"{rule['id']}::xref{i}"
        if qid in done:
            continue
        obj = chat_json(prompts.CROSSREF_QA.format(
            main_path=rule["section_path"], main_text=rule["article_text"],
            ref_path=sp(ref), ref_text=ref["content"]), max_tokens=1800)
        if not obj.get("question") or not obj.get("answer"):
            continue
        main_ev = {"chunk_id": rule["chunk_id"], "source_file": rule.get("source_file", ""),
                   "section_path": rule["section_path"], "quote": obj.get("main_quote", ""),
                   "article_text": rule["article_text"]}
        append_jsonl(CROSSREF_OUT, {
            "question_id": qid,
            "task_type": "상호참조 해석형",
            "question": obj["question"],
            "gold_answer": obj["answer"],
            "evidence": [main_ev, evidence_entry(ref, obj.get("ref_quote", ""))],
            "needs_review": False,
            "metadata": {"publisher": rule.get("publisher", "KR"),
                         "synthetic": False,
                         "ref_chunk_id": ref["chunk_id"]},
        })


# ── ⑤b 선급 비교 ────────────────────────────────────────────────────────
def cross_society_pairs(parents: list[dict], n: int) -> list[tuple[dict, dict, int]]:
    """클러스터 실측에서 (KR 조항, 타 선급 조항, cluster_id) 쌍 추출."""
    cmap = load_jsonl(OUT_DIR / "cluster_map.jsonl")
    by_id = defaultdict(list)
    for p in parents:
        by_id[p["chunk_id"]].append(p)
    clusters = defaultdict(list)
    for r in cmap:
        clusters[r["cluster"]].append(r)
    pairs = []
    for cid, members in sorted(clusters.items(), key=lambda kv: len(kv[1])):
        socs = {m["soc"] for m in members}
        if not (2 <= len(members) <= 6 and "KR" in socs and len(socs) >= 2):
            continue
        kr = next((m for m in members if m["soc"] == "KR"), None)
        other = next((m for m in members if m["soc"] != "KR"), None)
        a = by_id.get(kr["chunk_id"], [])
        b = by_id.get(other["chunk_id"], [])
        if len(a) == 1 and len(b) == 1:  # 유일 해석 가능한 멤버만
            pairs.append((a[0], b[0], cid))
        if len(pairs) >= n:
            break
    return pairs


def gen_compare(parents: list[dict], n: int, done: set[str]) -> None:
    for a, b, cid in cross_society_pairs(parents, n * 3):
        if len(done) >= n:
            break
        qid = f"COMPARE__{a['chunk_id']}__{b['publisher']}__{b['chunk_id']}"
        if qid in done:
            continue
        obj = chat_json(prompts.COMPARE_QA.format(
            soc_a=a["publisher"], path_a=sp(a), text_a=a["content"],
            soc_b=b["publisher"], path_b=sp(b), text_b=b["content"]), max_tokens=2200)
        if not obj.get("question") or not obj.get("answer"):
            continue
        append_jsonl(COMPARE_OUT, {
            "question_id": qid,
            "task_type": "선급 비교형",
            "question": obj["question"],
            "gold_answer": obj["answer"],
            "evidence": [evidence_entry(a, obj.get("quote_a", "")),
                         evidence_entry(b, obj.get("quote_b", ""))],
            "needs_review": False,
            "metadata": {"publisher": f"{a['publisher']}+{b['publisher']}",
                         "synthetic": False, "cluster": cid,
                         "materially_different": bool(obj.get("materially_different"))},
        })
        done.add(qid)
        if len(done) >= n:
            break


# ── main ────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-compare", type=int, default=2, help="선급 비교 문항 수")
    args = ap.parse_args(argv)

    parents = load_all_parents()
    log(f"청크 풀 {len(parents)}개 로드")

    rules = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    done_x = load_done(CROSSREF_OUT, key="question_id")
    for rule in rules:
        try:
            gen_crossref(rule, parents, done_x)
        except Exception as e:  # noqa: BLE001
            log(f"!! xref {rule['id']} 실패: {e}")

    try:
        gen_compare(parents, args.n_compare, load_done(COMPARE_OUT, key="question_id"))
    except Exception as e:  # noqa: BLE001
        log(f"!! compare 실패: {e}")

    log(f"상호참조 {len(load_jsonl(CROSSREF_OUT))}건 | 선급 비교 {len(load_jsonl(COMPARE_OUT))}건")


if __name__ == "__main__":
    main()
