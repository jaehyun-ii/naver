"""AIReg-Bench 선급 변형 — 최종 벤치마크 QA 조립 (LLM 불필요, 결정적).

rule_cards/excerpts/annotations를 조인해 RAG 평가용 benchmark_qa.jsonl을 만든다.
- evidence는 실제 chunk_id로 앵커 → eval_retrieval.py 방식의 검색 평가와 호환.
- expected_judgment는 생성 시 target_label(gold), LLM 어노테이션은 초안으로 동봉.
- label_consistent=False 항목은 needs_review로 표시 → 전문가 우선 검수 대상.

    python -m scripts.aireg_kr.build_qa   →  data_aireg/benchmark_qa.jsonl
"""
from __future__ import annotations

import json

from .common import OUT_DIR, load_jsonl, log

QA_OUT = OUT_DIR / "benchmark_qa.jsonl"

# boundary_c/exception_nm: 결정적 슬롯 대체 종류(run_all.plan_for) — 라벨은 각각 compliant/non_compliant
TASK_TYPE = {"compliant": "준수/부적합 판정형", "subtle_nc": "준수/부적합 판정형",
             "clear_nc": "준수/부적합 판정형", "insufficient": "부족 정보 식별형",
             "boundary_c": "준수/부적합 판정형", "exception_nm": "준수/부적합 판정형"}
DIFFICULTY = {"compliant": "medium", "subtle_nc": "hard",
              "clear_nc": "easy", "insufficient": "hard",
              "boundary_c": "hard", "exception_nm": "hard"}

UNACCEPTABLE = [
    "근거 조항 없이 적합/부적합만 말하는 답변",
    "제공되지 않은 외부 규정(IMO/IACS/타 선급)을 근거로 판단하는 답변",
    "문서의 자기 선언을 그대로 믿는 답변",
    "정보가 부족한데 적합/부적합으로 단정하는 답변",
]


def build_question(rule: dict, excerpt: dict) -> str:
    loc = rule["section_path"]
    if excerpt["kind"] == "insufficient":
        return (f"아래 기술문서 발췌만으로 「{loc}」의 요건 충족 여부를 판정할 수 있는가? "
                "판정이 가능하면 근거 조항을 인용해 판정하고, 불가능하면 어떤 정보가 더 필요한지 명시하라.\n\n"
                f"[기술문서 발췌]\n{excerpt['text']}")
    return (f"아래 기술문서 발췌의 설계·구성은 「{loc}」의 요건을 충족하는가? "
            "근거 조항을 인용해 판정하라.\n\n[기술문서 발췌]\n{0}".format(excerpt["text"]))


def main(argv: list[str] | None = None) -> None:  # noqa: ARG001 — 파이프라인 시그니처 통일
    rules = {r["id"]: r for r in load_jsonl(OUT_DIR / "rule_cards.jsonl")}
    annots = {a["id"]: a for a in load_jsonl(OUT_DIR / "annotations.jsonl")}
    # oneshot 모드: 어노테이션 없음 — verify verdict가 초안·검수 판단을 대신한다
    verdicts = {v["id"]: v for v in load_jsonl(OUT_DIR / "verdicts.jsonl")}
    excerpts = load_jsonl(OUT_DIR / "excerpts.jsonl")

    if QA_OUT.exists():
        QA_OUT.unlink()

    n = 0
    with QA_OUT.open("a", encoding="utf-8") as f:
        for ex in excerpts:
            rule = rules.get(ex["rule_id"])
            ann = annots.get(ex["id"])
            if ann is None and ex["id"] in verdicts:
                v = verdicts[ex["id"]]
                ann = {"annotation": v.get("verification"),
                       "label_consistent": v.get("verdict") == "ACCEPT"}
            if not rule:
                continue
            card = rule["card"] or {}
            row = {
                "question_id": ex["id"],
                "task_type": TASK_TYPE[ex["kind"]],
                "question": build_question(rule, ex),
                "expected_judgment": ex["target_label"],
                "generation_kind": ex["kind"],
                "evidence": [{
                    "chunk_id": rule["chunk_id"],
                    "source_file": rule.get("source_file", ""),
                    "section_path": rule["section_path"],
                    "requirement_ids": [q.get("requirement_id") for q in card.get("requirements", [])],
                    # 판정 전용(조항 제공) 평가 모드용 원문 — RAG 모드에서는 chunk_id로 검색해 검증
                    "article_text": rule["article_text"],
                }],
                "llm_annotation_draft": (ann or {}).get("annotation"),
                "needs_review": not (ann or {}).get("label_consistent", False),
                "unacceptable_answers": UNACCEPTABLE,
                "metadata": {
                    "publisher": rule.get("publisher", "KR"),
                    "rule_language": rule.get("language", "ko"),
                    "doc_title": rule["doc_title"],
                    "article_no": rule["article_no"],
                    "article_title": rule["article_title"],
                    "difficulty": DIFFICULTY[ex["kind"]],
                    "case_no": ex["case_no"],
                    # 기술문서는 전량 LLM 합성 — 선박명·회사명·도면번호는 모두 가상
                    "synthetic": True,
                    "quality_flags": ex.get("quality_flags", []),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    labels = {}
    for ex in excerpts:
        labels[ex["target_label"]] = labels.get(ex["target_label"], 0) + 1
    review = sum(1 for a in annots.values() if not a.get("label_consistent"))
    log(f"benchmark_qa.jsonl: {n}문항 | 라벨 분포 {labels} | 우선검수(needs_review) {review}건")


if __name__ == "__main__":
    main()
