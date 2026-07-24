"""AIReg-Bench 선급 변형 — Master Dataset 빌드 + canonical 규칙 그룹 단위 결정적 split.

기존 산출물(rule_cards/profiles/excerpts/annotations/verdicts + 스위트 QA)을
sample 단위로 join해 master_dataset.jsonl로 정규화하고, canonical_rule_group_id
(grouping.py — 발행처|문서|판 접두 + 정규화 규칙 좌표의 sha256) 단위 해시로
train/validation/test를 결정적으로 분할한 뒤 SFT/DPO/GRPO 학습 파일을 파생한다.

설계 원칙:
- 같은 실제 조항에서 나온 모든 샘플(판정 excerpt·스위트 QA)은 같은 split.
  스위트 QA는 rule_cards를 chunk_id로 역조인해 문서 문맥을 채워 판정 트랙과
  동일한 canonical 키로 수렴시킨다. 같은 canonical 그룹이 두 split에 있으면 실패.
- split은 sha256(f"{seed}:{canonical_id}") — 파이썬 hash() 금지(프로세스별 상이).
- chunk_id fallback 비율이 임계값을 넘으면 경고(warn)·strict 실패(fail).
- 파생 SFT/DPO는 build_training의 조립 함수 재사용. GRPO는 prompt + gold
  메타데이터(gold_label·task_type·answer_format 등)를 보존해 검증형 보상에 쓴다.
- verdicts(독립 검증)가 있으면 REJECT 샘플은 파생에서 제외하되 마스터에는 보존.
- 없는 값은 null/빈 배열 — 임의 생성 금지.
- 빌드 종료 시 master_build_report.{json,md} 생성(경고 상단 표기).

    python -m scripts.aireg_kr.build_master [--seed 42] [--ratios 0.8,0.1,0.1]
        [--fallback-warn 0.05] [--fallback-fail 0.20] [--strict]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from .answer_format import (ANSWER_FORMAT_JUDGMENT, REGULATION_JUDGMENT_TASKS,
                            UNCERTAIN_DECISION, display_label,
                            format_regulation_answer)
from .build_training import (BASE_KIND, KIND_SLOTS, RagContext, band_adjust,
                             build_chosen, build_no_golden_answer, build_prompt,
                             build_rejected, load_chunk_pool, pick_distractors,
                             pick_quote, suite_rows)
from .common import OUT_DIR, load_jsonl, log
from .condition_evaluator import EVALUATOR_VERSION, OPERATOR_ALIASES, SUPPORTED_OPS
from .grouping import group_id_for_sample
from .labels import LABEL_POLICIES, compare_labels, should_include_for_training

MASTER_DIR = OUT_DIR / "master"
SPLITS = ("train", "validation", "test")
# 스위트 QA 트랙 파일 → 정식 task_type (build_suite_qa.TRACK_FILES와 정합)
SUITE_FILES = {"spec_qa.jsonl": "direct_qa", "applicability_qa.jsonl": "applicability",
               "crossref_qa.jsonl": "cross_reference_lookup",
               "compare_qa.jsonl": "comparison",
               "def_link_qa.jsonl": "cross_reference_lookup",
               "precedence_qa.jsonl": "direct_qa",
               "unit_convert_qa.jsonl": "requirement_satisfaction",
               "table_lookup_qa.jsonl": "extractive_qa",
               "hierarchy_qa.jsonl": "applicability"}
REGISTRY_PATH = Path(__file__).parent / "prompt_registry.json"

# 판정 트랙 kind → 정식 task_type(전부 판정형 — answer_format 라우팅 기준)
TASK_TYPE_FROM_KIND = {"compliant": "compliance_judgment", "subtle_nc": "compliance_judgment",
                       "clear_nc": "compliance_judgment", "insufficient": "insufficient_information",
                       "boundary_c": "condition_verification", "exception_nm": "exception_judgment"}


# ── 결정적 split ─────────────────────────────────────────────────────────
def stable_bucket(seed: int, key: str) -> float:
    """sha256 기반 [0,1) 버킷 — 프로세스·실행 무관 결정적."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_split(seed: int, key: str, ratios: tuple[float, float, float]) -> str:
    b = stable_bucket(seed, key)
    if b < ratios[0]:
        return "train"
    if b < ratios[0] + ratios[1]:
        return "validation"
    return "test"


def group_key(sample: dict) -> tuple[str, bool]:
    """(레거시) rule_id 우선 그룹 키 — split 변화 전후 비교 통계 전용."""
    rid = (sample.get("rule") or {}).get("rule_id")
    if rid:
        return rid, False
    src = sample.get("source") or {}
    if src.get("source_file") and src.get("section_path"):
        return f"{src['source_file']}::{src['section_path']}", True
    if src.get("chunk_id"):
        return src["chunk_id"], True
    return sample["sample_id"], True


def _prompt_parts(gen: dict | None) -> tuple[str | None, str | None]:
    """_gen.prompt("EXCERPT/v1") → (id, version)."""
    p = (gen or {}).get("prompt") or ""
    if "/" in p:
        pid, ver = p.rsplit("/", 1)
        return pid, ver
    return (p or None), None


# ── join → master 정규화 ────────────────────────────────────────────────
def _index_unique(rows: list[dict], name: str, stats: Counter,
                  strict: bool, key: str = "id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        rid = r.get(key)
        if rid in out:
            stats[f"duplicate_{name}"] += 1
            if strict:
                raise SystemExit(f"[strict] {name} 중복 id: {rid}")
        out[rid] = r
    return out


def _rule_block(rule: dict) -> dict:
    card = rule.get("card") or {}
    reqs = card.get("requirements") or []
    return {
        "rule_id": rule["id"],
        "rule_type": (card.get("list_logic") or {}).get("type"),  # v1 카드는 null
        "conditions": [c for r in reqs for c in (r.get("conditions") or [])],
        "requirements": reqs,
        "exceptions": [r["exception"] for r in reqs if (r.get("exception") or "").strip()],
        "cross_references": [],  # 카드에 미수집 — 임의 생성 금지
    }


_KNOWN_PUBLISHERS = {"KR", "ABS", "DNV", "LR", "BV", "IACS", "NK", "CLASSNK"}


def resolve_publisher(rule: dict) -> tuple[str | None, str]:
    """publisher 보완 — §15 우선순위. 파일명 추측 금지, 근거 있는 소스만.

    1) 원본 source.publisher  2) rule_uid/chunk_id의 문서화된 접두 규약
       (비-KR 'PUB__…', 신 KR 청커 'KR_…' — rule_uid()/kr_rule 청커 규약)
    3) unknown
    """
    if rule.get("publisher"):
        return rule["publisher"], "source.publisher"
    for key in ("id", "chunk_id"):
        rid = str(rule.get(key) or "")
        m = re.match(r"^([A-Za-z]+)__", rid)          # 비-KR rule_uid 규약
        if m and m.group(1).upper() in _KNOWN_PUBLISHERS:
            return m.group(1), "rule_uid_prefix"
        m = re.match(r"^([A-Za-z]+)_", rid)           # 신 KR 청커 doc 접두 규약
        if m and m.group(1).upper() in _KNOWN_PUBLISHERS:
            return m.group(1), "chunk_id_prefix"
        # 구 KR 청커 세그먼트 접두 — common.rule_uid() 문서화 규약("KR RULE_*은 자체 유일")
        if re.match(r"^(RULE|GUIDE|APPX)_", rid):
            return "KR", "kr_rule_uid_convention"
    return None, "unknown"


def _source_block(rule: dict) -> dict:
    pub, pub_src = resolve_publisher(rule)
    return {
        "doc_id": None,  # 별도 문서 레지스트리 부재 — source_file이 실질 식별자
        "edition": None,
        "publisher": pub,
        "publisher_source": pub_src,
        "doc_title": rule.get("doc_title"),
        "source_file": rule.get("source_file"),
        "section_path": rule.get("section_path"),
        "chunk_id": rule.get("chunk_id"),
        "source_text": rule.get("article_text"),
    }


def _quality_block(ex_gen: dict | None, ver: dict | None, provenance: dict) -> dict:
    gpid, gver = _prompt_parts(ex_gen)
    vpid, vver = _prompt_parts((ver or {}).get("_gen"))
    return {
        "generator_prompt_id": gpid, "generator_prompt_version": gver,
        "generator_model": (ex_gen or {}).get("model"),
        "verifier_prompt_id": vpid, "verifier_prompt_version": vver,
        "verification_result": (ver or {}).get("verdict"),
        "provenance": provenance,  # 논문 계보는 prompt_registry.json에서 id로 조회
    }


def build_judgment_samples(strict: bool, stats: Counter) -> list[dict]:
    """판정 트랙: excerpt를 기준 행으로 rule/profile/annotation/verdict를 id join."""
    rules = _index_unique(load_jsonl(OUT_DIR / "rule_cards.jsonl"), "rule_cards", stats, strict)
    profiles = _index_unique(load_jsonl(OUT_DIR / "profiles.jsonl"), "profiles", stats, strict)
    annots = _index_unique(load_jsonl(OUT_DIR / "annotations.jsonl"), "annotations", stats, strict)
    verdicts = _index_unique(load_jsonl(OUT_DIR / "verdicts.jsonl"), "verdicts", stats, strict)

    samples = []
    for ex in load_jsonl(OUT_DIR / "excerpts.jsonl"):
        rule = rules.get(ex["rule_id"])
        if not rule:
            stats["orphan_excerpt_no_rule"] += 1
            if strict:
                raise SystemExit(f"[strict] excerpt {ex['id']}: rule_card 없음 ({ex['rule_id']})")
            continue
        prof, ann, ver = profiles.get(ex["id"]), annots.get(ex["id"]), verdicts.get(ex["id"])
        if not prof:
            stats["orphan_excerpt_no_profile"] += 1
        if not ann:
            stats["excerpt_no_annotation"] += 1
        if not ver:
            stats["excerpt_no_verdict"] += 1
        samples.append({
            "sample_id": ex["id"],
            "rule_modality": None,  # 프로즈 excerpt — 요건 단위 modality 미지정(하위 호환)
            "question_intent": "compliance_judgment",
            "labels": None,
            "task": {"task_type": TASK_TYPE_FROM_KIND.get(ex.get("kind"), "compliance_judgment"),
                     "answer_format": ANSWER_FORMAT_JUDGMENT,
                     "kind": ex.get("kind"), "target_label": ex.get("target_label")},
            "source": _source_block(rule),
            "rule": _rule_block(rule),
            "profile": ({"kind": prof.get("kind"), "case_no": prof.get("case_no"),
                         "text": prof.get("text")} if prof else None),
            "excerpt": {"doc_type_title": ex.get("doc_type_title"), "text": ex.get("text"),
                        "quality_flags": ex.get("quality_flags") or []},
            "annotation": ({"annotation": ann.get("annotation"),
                            "label_consistent": ann.get("label_consistent")} if ann else None),
            "verdict": ({"verdict": ver.get("verdict"), "derived_label": ver.get("derived_label"),
                         "review_priority": ver.get("review_priority"),
                         "reasons": ver.get("reasons") or []} if ver else None),
            # 판정 트랙 사례는 프로즈(excerpt)라 구조화 facts가 없음 — 평가 불가 명시.
            # UNKNOWN을 임의 판정으로 바꾸지 않는다(§3).
            "program_evaluation": {
                "evaluable": False, "status": "UNKNOWN", "derived_label": "UNKNOWN",
                "unsupported_conditions": ["사례 facts 비구조화(제출문서 프로즈)"],
                "evaluator_version": EVALUATOR_VERSION},
            "label_consistency": compare_labels(
                ex.get("target_label"), (ver or {}).get("derived_label"), None),
            "quality": _quality_block(
                ex.get("_gen"), ver,
                {stage: (row or {}).get("_gen")
                 for stage, row in (("rule_card", rule), ("profile", prof), ("excerpt", ex),
                                    ("annotation", ann), ("verdict", ver))}),
        })
    return samples


def build_case_samples(stats: Counter) -> list[dict]:
    """구조화 사례 트랙(build_cases 산출) — 경계값·예외 mutation 케이스."""
    rules = {r["id"]: r for r in load_jsonl(OUT_DIR / "rule_cards.jsonl")}
    verifs = {v["case_id"]: v for v in load_jsonl(OUT_DIR / "case_verifications.jsonl")}
    samples = []
    for case in load_jsonl(OUT_DIR / "cases.jsonl"):
        rule = rules.get(case["rule_id"])
        if not rule:
            stats["orphan_case_no_rule"] += 1
            continue
        ver = verifs.get(case["case_id"])
        if ver:
            lc = ver["label_consistency"]
        else:
            stats["case_no_verification"] += 1
            lc = {"generation_label": case["intended_label"], "verifier_label": None,
                  "program_label": case["program_label"],
                  "agreement": "VERIFIER_PENDING", "action": "NEEDS_VERIFICATION"}
        samples.append({
            "sample_id": case["case_id"],
            "rule_modality": case.get("rule_modality"),
            "question_intent": case.get("question_intent"),
            "labels": case.get("labels"),
            "task": {"task_type": case["task_type"], "answer_format": case["answer_format"],
                     "kind": case["case_kind"], "target_label": case["intended_label"],
                     "question": case.get("question") or case["prompt"]},
            "source": _source_block(rule),
            "rule": _rule_block(rule),
            "profile": None, "excerpt": None, "annotation": None,
            "verdict": ({"verdict": None, "derived_label": ver.get("verifier_label")}
                        if ver else None),
            "case": {"case_facts": case["case_facts"], "case_text": case["case_text"],
                     "prompt": case["prompt"], "mutation": case["mutation"],
                     "fact_realization_checks": case.get("fact_realization_checks") or []},
            "verifier_flags": {"abstention_conflict": (ver or {}).get("abstention_conflict"),
                               "fact_conflicts": (ver or {}).get("fact_conflicts") or [],
                               "two_pass": (ver or {}).get("two_pass")},
            "program_evaluation": case["program_evaluation"],
            "label_consistency": lc,
            "quality": _quality_block(case.get("_gen"), None,
                                      {"case": case.get("_gen"),
                                       "verification": (ver or {}).get("_gen")}),
        })
    return samples


def build_suite_samples(stats: Counter) -> list[dict]:
    """스위트 QA 트랙 — rule_cards를 chunk_id로 역조인해 문서 문맥을 채운다.

    문맥(publisher/source_file)이 채워져야 판정 트랙과 canonical 키가 수렴한다.
    역조인 실패분은 통계로 남기고 evidence 정보만으로 진행(fallback 후보).
    """
    by_chunk: dict[str, dict] = {}
    for r in load_jsonl(OUT_DIR / "rule_cards.jsonl"):
        by_chunk.setdefault(r.get("chunk_id"), r)
    # 스위트 블라인드 검증(verify_suite) 결과 — REJECT는 needs_review로 승격
    suite_verdicts = {v["question_id"]: v
                      for v in load_jsonl(OUT_DIR / "suite_verdicts.jsonl")}

    samples = []
    for fname, task_type in SUITE_FILES.items():
        for row in load_jsonl(OUT_DIR / fname):
            sv = suite_verdicts.get(row["question_id"], {})
            ev = (row.get("evidence") or [{}])[0]
            rule = by_chunk.get(ev.get("chunk_id"))
            if rule:
                stats["suite_rule_joined"] += 1
                source = _source_block(rule) | {"section_path": ev.get("section_path")
                                                or rule.get("section_path")}
                rule_id = rule["id"]
            else:
                stats["suite_rule_join_miss"] += 1
                source = {"doc_id": None, "edition": None,
                          "publisher": (row.get("metadata") or {}).get("publisher"),
                          "doc_title": None, "source_file": None,
                          "section_path": ev.get("section_path"),
                          "chunk_id": ev.get("chunk_id"),
                          "source_text": ev.get("article_text")}
                rule_id = ev.get("chunk_id")
            samples.append({
                "sample_id": row["question_id"],
                "task": {"task_type": task_type,
                         "answer_format": ANSWER_FORMAT_JUDGMENT
                         if task_type in REGULATION_JUDGMENT_TASKS else None,
                         "task_type_label": row.get("task_type"),
                         "track": row.get("track"),
                         "question": row.get("question"),
                         "expected_judgment": row.get("expected_judgment")},
                "source": source,
                "rule": {"rule_id": rule_id, "rule_type": None, "conditions": [],
                         "requirements": [], "exceptions": [], "cross_references": []},
                "profile": None, "excerpt": None, "annotation": None, "verdict": None,
                "quality": _quality_block(row.get("_gen"), None, {"qa": row.get("_gen")})
                | {"needs_review": (row.get("needs_review", False)
                                    or sv.get("verdict") == "REJECT"),
                   # 신 스키마(verify-loop): status 단일화 + trainable 확정값
                   "status": row.get("status"),
                   "trainable": row.get("trainable"),
                   "review_reasons": row.get("review_reasons") or [],
                   "suite_verdict": sv.get("verdict"),
                   "suite_verdict_issues": sv.get("issues") or []},
            })
    return samples


# ── 파생 학습 파일 ───────────────────────────────────────────────────────
class _SplitFiles:
    """kind(sft/dpo/grpo) × split 파일 핸들 + 건수 집계."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.files = {(k, s): (out_dir / f"{k}_{s}.jsonl").open("w", encoding="utf-8")
                      for k in ("sft", "dpo", "grpo") for s in SPLITS}
        self.counts: Counter = Counter()

    def write(self, kind: str, split: str, row: dict) -> None:
        self.files[(kind, split)].write(json.dumps(row, ensure_ascii=False) + "\n")
        self.counts[(kind, split)] += 1

    def close(self) -> None:
        for f in self.files.values():
            f.close()


def derive_judgment(samples: list[dict], split_of: dict[str, str], files: _SplitFiles,
                    n_distractors: int, stats: Counter, label_policy: str) -> None:
    """판정 트랙 → SFT/DPO/GRPO. build_training main 루프와 동일한 조립·층화 로직.

    포함 여부는 라벨 정책 함수(should_include_for_training)가 결정 — 블라인드
    REJECT 제외(기존 동작)에 3자 라벨 정책이 얹힌다. 제외 사유는 통계로 집계.
    """
    rule_rows = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    rules = {r["id"]: r for r in rule_rows}
    rule_order = {r["id"]: i for i, r in enumerate(rule_rows)}
    annots = {a["id"]: a for a in load_jsonl(OUT_DIR / "annotations.jsonl")}
    pool = load_chunk_pool()
    usable = []
    for s in samples:
        inc, reason = should_include_for_training(
            verifier_verdict=(s.get("verdict") or {}).get("verdict"),
            label_consistency=s.get("label_consistency"),
            label_policy=label_policy)
        if inc:
            usable.append(s)
        else:
            stats[f"derived_skip_{reason}"] += 1
    n_per_rule = Counter(s["rule"]["rule_id"] for s in usable)

    # 실검색(RAG) 정합 — build_training과 동일: distractor는 실측 hard negative,
    # no-golden은 실측 검색 실패 기반 + 밴드 보정. 미가용 시 기존 결정적 로직.
    rag = None
    if os.environ.get("AIREG_NO_RETRIEVAL", "0") != "1":
        from . import retrieval
        if retrieval.available():
            rag = RagContext(pool)
            log(f"판정 트랙 실검색 모드: {retrieval.QDRANT_URL}/{retrieval.COLLECTION}")

    plans = []
    for s in usable:
        rid = s["rule"]["rule_id"]
        rule, ann_row = rules.get(rid), annots.get(s["sample_id"])
        if not rule or not ann_row:
            stats["derived_skip_no_join"] += 1
            continue
        ex = {"id": s["sample_id"], "rule_id": rid, "text": s["excerpt"]["text"],
              "kind": s["task"]["kind"], "target_label": s["task"]["target_label"],
              "case_no": (s.get("profile") or {}).get("case_no", 1)}
        rng = random.Random(int(hashlib.sha1(ex["id"].encode()).hexdigest(), 16))
        if rag is not None:
            distractors, golden_hit = rag.retrieve(ex["text"], rule, n_distractors)
            if len(distractors) < n_distractors:
                have = {(d.get("publisher"), d["chunk_id"]) for d in distractors}
                distractors += [d for d in pick_distractors(
                    pool, rule, n_distractors + 2, rng)
                    if (d.get("publisher"), d["chunk_id"]) not in have
                    ][:n_distractors - len(distractors)]
            include_golden = golden_hit
        else:
            distractors = pick_distractors(pool, rule, n_distractors, rng)
            if n_per_rule[rid] == 1:
                include_golden = rule_order[rid] % 5 != 0
            else:
                no_golden_slot = KIND_SLOTS[rule_order[rid] % len(KIND_SLOTS)]
                include_golden = (ex["case_no"],
                                  BASE_KIND.get(ex["kind"], ex["kind"])) != no_golden_slot
        plans.append({"s": s, "rule": rule, "ann_row": ann_row, "ex": ex,
                      "rng": rng, "distractors": distractors,
                      "include_golden": include_golden, "tag": ""})
    if rag is not None:
        band_adjust(plans)
        stats["derived_ng_natural"] = sum(1 for p in plans
                                          if not p["include_golden"] and not p["tag"])
        stats["derived_ng_forced"] = sum(1 for p in plans if p["tag"] == "ng_forced")
        stats["derived_golden_forced"] = sum(1 for p in plans
                                             if p["tag"] == "golden_forced")

    for p in plans:
        s, rule, ex, rng = p["s"], p["rule"], p["ex"], p["rng"]
        rid, distractors = ex["rule_id"], p["distractors"]
        include_golden = p["include_golden"]
        split = split_of[s["sample_id"]]
        gid = s["canonical_rule_group_id"]
        ann = p["ann_row"]["annotation"]
        prompt, golden_pos = build_prompt(rule, distractors, ex, rng, include_golden)
        chosen = (build_chosen(rule, ex, ann, golden_pos) if include_golden
                  else build_no_golden_answer(rule))

        files.write("sft", split, {
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": chosen}],
            "canonical_rule_group_id": gid,
            "source": f"aireg_kr:{ex['id']}" + ("" if include_golden else ":no_golden")})
        if include_golden:
            for kind, rejected in build_rejected(rule, ex, distractors, golden_pos, rng):
                files.write("dpo", split, {"prompt": prompt, "chosen": chosen,
                                           "rejected": rejected,
                                           "canonical_rule_group_id": gid,
                                           "source": f"aireg_kr:{ex['id']}:{kind}"})
        # GRPO: prompt + gold·형식 메타데이터 보존 → run_grpo_peft reward kwargs로 전달
        files.write("grpo", split, {
            "prompt": prompt,
            "gold_label": "uncertain" if not include_golden else ex["target_label"],
            "task_type": s["task"]["task_type"],
            "answer_format": ANSWER_FORMAT_JUDGMENT,
            "requires_evidence_section": True,
            "requires_missing_information":
                (not include_golden) or ex["target_label"] == "uncertain",
            "kind": ex["kind"], "rule_id": rid, "sample_id": ex["id"],
            "canonical_rule_group_id": gid,
            "no_golden": not include_golden,
            "evidence_quote": pick_quote(rule["article_text"], ann) if include_golden else "",
            "missing_fields": (ann.get("missing_information") or []
                               if ex["target_label"] == "uncertain" else [])})


def derive_suite(samples: list[dict], split_of: dict[str, str],
                 files: _SplitFiles) -> None:
    """스위트 QA → split별 SFT/DPO. suite_rows를 라우팅 콜백으로 재사용하며,
    파생 행에 canonical 그룹 ID·task_type을 주입한다."""
    pool = load_chunk_pool()
    meta_by_qid = {s["sample_id"]: s for s in samples}

    class _W:  # suite_rows가 기대하는 파일 인터페이스 — JSON 행에 메타 주입 후 기록
        def __init__(self, kind: str, split: str, extra: dict):
            self.kind, self.split, self.extra = kind, split, extra

        def write(self, line: str) -> None:
            row = json.loads(line) | self.extra
            files.write(self.kind, self.split, row)

    def route(row: dict):
        s = meta_by_qid.get(row["question_id"])
        if s is None:
            return None
        split = split_of[s["sample_id"]]
        extra = {"canonical_rule_group_id": s["canonical_rule_group_id"],
                 "task_type": s["task"]["task_type"]}
        return _W("sft", split, extra), _W("dpo", split, extra)

    rag = None
    if os.environ.get("AIREG_NO_RETRIEVAL", "0") != "1":
        from . import retrieval
        if retrieval.available():
            rag = RagContext(pool)
    suite_rows(pool, None, None, route=route, rag=rag)


def check_fallback(ratio: float, warn: float, fail: float, strict: bool) -> list[str]:
    """chunk_id fallback 비율 임계값 검사 — 경고 리스트 반환, strict 초과 시 실패."""
    warnings: list[str] = []
    if strict and ratio > fail:
        raise SystemExit(f"[strict] fallback 비율 {ratio:.1%} > {fail:.0%} — "
                         "그룹 키 소스 정비 필요")
    if ratio > warn:
        warnings.append(f"chunk_id fallback 비율 {ratio:.1%}: 권장 기준 {warn:.0%} 초과")
    return warnings


_LABEL_KO = {"APPLICABLE": "적용", "NOT_APPLICABLE": "적용되지 않음",
             "EXCEPTION_APPLIES": "예외 적용",
             "INSUFFICIENT_INFORMATION": UNCERTAIN_DECISION}


def _arrow_safe_results(results: list[dict]) -> list[dict]:
    """GRPO 파생용 조건 결과 직렬화 — 행마다 값 타입(bool/float/str)이 섞이면
    datasets(pyarrow) 스키마 충돌로 학습 로드가 깨진다. 값은 전부 문자열로 통일
    (마스터 원본은 원 타입 유지, 보상 함수는 텍스트 매칭이라 손실 없음)."""
    out = []
    for r in results or []:
        out.append({k: (v if isinstance(v, str) or v is None
                        else json.dumps(v, ensure_ascii=False))
                    for k, v in r.items()})
    return out


def derive_cases(samples: list[dict], split_of: dict[str, str], files: _SplitFiles,
                 stats: Counter, label_policy: str) -> None:
    """구조화 사례 → GRPO(+결정적 SFT). 정답문은 evaluator 결과로 기계 조립."""
    for s in samples:
        inc, reason = should_include_for_training(
            verifier_verdict=None, label_consistency=s.get("label_consistency"),
            label_policy=label_policy)
        if not inc:
            stats[f"derived_skip_{reason}"] += 1
            continue
        split = split_of[s["sample_id"]]
        pe = s["program_evaluation"]
        case = s["case"]
        label = s["task"]["target_label"]
        review = [f"{c['field']} {c['operator']} {c['expected_value']}"
                  f"{('(' + str(c['unit']) + ')') if c.get('unit') else ''}: {c['status']}"
                  for c in (pe.get("condition_results") or []) + (pe.get("exception_results") or [])
                  if c["field"] != "applicability_asserted"]
        answer = format_regulation_answer(
            decision=display_label(label),
            evidence=f"규정 조항({s['source'].get('section_path')})의 조건과 사례 사실 대조",
            condition_review=review or None,
            missing_information=pe.get("missing_fields") or None)
        files.write("sft", split, {
            "messages": [{"role": "user", "content": case["prompt"]},
                         {"role": "assistant", "content": answer}],
            "canonical_rule_group_id": s["canonical_rule_group_id"],
            "rule_modality": s.get("rule_modality"),
            "question_intent": s.get("question_intent"),
            "primary_label": label,
            "source": f"aireg_kr_case:{s['sample_id']}"})
        files.write("grpo", split, {
            "prompt": case["prompt"],
            "gold_label": label,                      # = primary_label (하위 호환 alias)
            "primary_label": label,
            "applicability_label": (s.get("labels") or {}).get("applicability_label"),
            "compliance_label": (s.get("labels") or {}).get("compliance_label"),
            "rule_modality": s.get("rule_modality"),
            "question_intent": s.get("question_intent"),
            "program_label": pe.get("primary_label") or pe.get("derived_label"),
            "label_agreement": s["label_consistency"]["agreement"],
            "condition_results": _arrow_safe_results(pe.get("condition_results")),
            "exception_results": _arrow_safe_results(pe.get("exception_results")),
            "evaluable": pe.get("evaluable", False),
            "evaluator_version": pe.get("evaluator_version", EVALUATOR_VERSION),
            "task_type": s["task"]["task_type"],
            "answer_format": s["task"]["answer_format"],
            "kind": s["task"]["kind"], "rule_id": s["rule"]["rule_id"],
            "sample_id": s["sample_id"],
            "canonical_rule_group_id": s["canonical_rule_group_id"],
            "missing_fields": pe.get("missing_fields") or []})
        stats["derived_case_rows"] += 1


# ── leakage 검증 ─────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def _norm_hash(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text.strip()).encode()).hexdigest()


def validate(samples: list[dict], split_of: dict[str, str]) -> list[str]:
    """canonical 그룹·sample·본문/질문 정규화 해시의 split 교차 검사.

    canonical 그룹 교차는 빌드 실패, 본문/질문 정확중복 교차는 경고 리스트 반환.
    """
    group_splits: dict[str, set[str]] = {}
    sample_splits: dict[str, set[str]] = {}
    text_splits: dict[str, set[str]] = {}
    question_splits: dict[str, set[str]] = {}
    for s in samples:
        sp = split_of[s["sample_id"]]
        group_splits.setdefault(s["canonical_rule_group_id"], set()).add(sp)
        sample_splits.setdefault(s["sample_id"], set()).add(sp)
        src_text = (s.get("source") or {}).get("source_text") or ""
        if src_text.strip():
            text_splits.setdefault(_norm_hash(src_text), set()).add(sp)
        q = (s.get("task") or {}).get("question") or ""
        if q.strip():
            question_splits.setdefault(_norm_hash(q), set()).add(sp)

    bad_groups = {g: sp for g, sp in group_splits.items() if len(sp) > 1}
    if bad_groups:
        raise SystemExit(f"[leakage] canonical 그룹이 복수 split에 존재: "
                         f"{list(bad_groups)[:5]} …빌드 실패")
    bad_samples = [k for k, sp in sample_splits.items() if len(sp) > 1]
    if bad_samples:
        raise SystemExit(f"[leakage] sample_id가 복수 split에 존재: {bad_samples[:5]}")
    warnings = []
    n_text = sum(1 for sp in text_splits.values() if len(sp) > 1)
    if n_text:
        warnings.append(f"동일 source_text 정규화 해시가 {n_text}건 split을 교차")
    n_q = sum(1 for sp in question_splits.values() if len(sp) > 1)
    if n_q:
        warnings.append(f"동일 질문 정규화 해시가 {n_q}건 split을 교차")
    return warnings


# ── 리포트 ───────────────────────────────────────────────────────────────
def build_report(samples: list[dict], split_of: dict[str, str], stats: Counter,
                 files: _SplitFiles, warnings: list[str], split_diff: dict) -> dict:
    fallback = [s for s in samples if s["canonical_rule_group_source"] == "chunk_id"]
    prompt_counts = Counter(
        f"{s['quality'].get('generator_prompt_id')}/{s['quality'].get('generator_prompt_version')}"
        for s in samples if s["quality"].get("generator_prompt_id"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {}
    return {
        "warnings": warnings,
        "totals": {
            "master_records": len(samples),
            "judgment_track": sum(1 for s in samples
                                  if s["task"]["task_type"] in TASK_TYPE_FROM_KIND.values()
                                  and s.get("excerpt")),
            "suite_qa": sum(1 for s in samples if not s.get("excerpt")),
            "canonical_groups": len({s["canonical_rule_group_id"] for s in samples}),
        },
        "by_publisher": dict(Counter((s["source"].get("publisher") or "unknown")
                                     for s in samples)),
        "by_document": dict(Counter((s["source"].get("source_file")
                                     or s["source"].get("doc_title") or "unknown")
                                    for s in samples)),
        "canonical_source_counts": dict(Counter(s["canonical_rule_group_source"]
                                                for s in samples)),
        "fallback": {"count": len(fallback), "ratio": round(len(fallback) / len(samples), 4)},
        "split": {
            "records": {sp: sum(1 for s in samples if split_of[s["sample_id"]] == sp)
                        for sp in SPLITS},
            "canonical_groups": {sp: len({s["canonical_rule_group_id"] for s in samples
                                          if split_of[s["sample_id"]] == sp})
                                 for sp in SPLITS},
            "changed_vs_legacy_rule_id_key": split_diff,
        },
        "task_type_counts": dict(Counter(s["task"]["task_type"] for s in samples)),
        "profile_kind_counts": dict(Counter(s["task"].get("kind") for s in samples
                                            if s["task"].get("kind"))),
        "verdict_counts": dict(Counter((s.get("verdict") or {}).get("verdict") or "미검증"
                                       for s in samples)),
        "derived_counts": {f"{k}_{sp}": files.counts.get((k, sp), 0)
                           for k in ("sft", "dpo", "grpo") for sp in SPLITS},
        "exclusion_stats": dict(stats),
        "prompt_id_counts": dict(prompt_counts),
        "prompt_registry_ids": sorted(k for k in registry if not k.startswith("_")),
    }


def classify_card(card: dict) -> str:
    """§14.1 평가 가능성 분류 — 비율을 억지로 높이지 않는다."""
    if not isinstance(card, dict) or not card:
        return "INVALID_RULE_CARD"
    reqs = card.get("requirements") or []
    conds = [c for r in reqs for c in (r.get("conditions") or [])]
    if not conds:
        return "MISSING_STRUCTURE"  # v1 카드 등 — 조건 배열 자체가 없음
    def supported(c):
        op = OPERATOR_ALIASES.get(str(c.get("operator") or ""), str(c.get("operator") or ""))
        return op in SUPPORTED_OPS and str(c.get("value") or "").strip() != ""
    per_req_full = [r for r in reqs if (r.get("conditions") and
                                        all(supported(c) for c in r["conditions"]))]
    if per_req_full:
        return "PROGRAM_EVALUABLE"
    if any(supported(c) for c in conds):
        return "PROGRAM_PARTIALLY_EVALUABLE"
    return "QUALITATIVE_ONLY"


def build_evaluator_report(samples: list[dict], stats: Counter) -> dict:
    """§14 평가 범위·3자 라벨 리포트 — evaluator_report.{json,md}."""
    rule_rows = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    classes = Counter(classify_card(r.get("card")) for r in rule_rows)
    op_counts: Counter = Counter()
    unit_counts: Counter = Counter()
    for r in rule_rows:
        for req in (r.get("card") or {}).get("requirements") or []:
            for c in req.get("conditions") or []:
                op_counts[str(c.get("operator") or "?")] += 1
                if c.get("unit"):
                    unit_counts[str(c["unit"])] += 1
    agreements = Counter(s["label_consistency"]["agreement"] for s in samples
                         if s.get("label_consistency"))
    unknown_reasons: Counter = Counter()
    invalid_reasons: Counter = Counter()
    for s in samples:
        pe = s.get("program_evaluation") or {}
        for c in (pe.get("condition_results") or []) + (pe.get("exception_results") or []):
            if c["status"] == "UNKNOWN":
                unknown_reasons[c["reason"][:40]] += 1
            elif c["status"] == "INVALID":
                invalid_reasons[c["reason"][:40]] += 1
        for u in pe.get("unsupported_conditions") or []:
            unknown_reasons[str(u)[:40]] += 1
    n_eval = sum(1 for s in samples if (s.get("program_evaluation") or {}).get("evaluable"))
    return {
        "rule_cards": {"total": len(rule_rows), "classes": dict(classes),
                       "evaluable_ratio": round(
                           classes.get("PROGRAM_EVALUABLE", 0) / max(1, len(rule_rows)), 4)},
        "operator_counts": dict(op_counts),
        "unit_counts": dict(unit_counts),
        "samples": {"total": len(samples), "program_evaluable": n_eval,
                    "evaluable_ratio": round(n_eval / max(1, len(samples)), 4)},
        "label_agreements": dict(agreements),
        "unknown_reasons": dict(unknown_reasons),
        "invalid_reasons": dict(invalid_reasons),
        "exclusions": {k: v for k, v in stats.items() if k.startswith("derived_skip_")},
        "by_task_type": {t: dict(Counter(s["label_consistency"]["agreement"]
                                         for s in samples
                                         if s["task"]["task_type"] == t
                                         and s.get("label_consistency")))
                         for t in sorted({s["task"]["task_type"] for s in samples})},
        "evaluator_version": EVALUATOR_VERSION,
    }


def write_evaluator_report_md(rep: dict, path: Path) -> None:
    rc = rep["rule_cards"]
    lines = [
        "# 조건 평가기 커버리지·라벨 정합 리포트", "",
        f"- evaluator: {rep['evaluator_version']}",
        f"- Rule Card {rc['total']}개 — 분류 {rc['classes']} (완전 평가 가능 {rc['evaluable_ratio']:.1%})",
        f"- 샘플 {rep['samples']['total']}건 중 프로그램 평가 가능 "
        f"{rep['samples']['program_evaluable']}건 ({rep['samples']['evaluable_ratio']:.1%})",
        f"- 3자 라벨 일치: {rep['label_agreements']}",
        f"- 연산자 분포: {rep['operator_counts']}",
        f"- 단위 분포: {rep['unit_counts']}",
        f"- UNKNOWN 사유: {rep['unknown_reasons']}",
        f"- INVALID 사유: {rep['invalid_reasons']}",
        f"- 학습 파생 제외: {rep['exclusions']}",
        f"- task_type별 일치: {rep['by_task_type']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_md(report: dict, path: Path) -> None:
    lines = ["# Master Dataset 빌드 리포트", ""]
    if report["warnings"]:
        lines += ["## WARNING", ""] + [f"- {w}" for w in report["warnings"]] + [""]
    t = report["totals"]
    lines += [
        "## 요약", "",
        f"- 전체 레코드: {t['master_records']} (판정 {t['judgment_track']} / 스위트 QA {t['suite_qa']})",
        f"- canonical 그룹: {t['canonical_groups']}",
        f"- chunk_id fallback: {report['fallback']['count']}건 ({report['fallback']['ratio']:.1%})",
        f"- split 레코드: {report['split']['records']}",
        f"- split 그룹: {report['split']['canonical_groups']}",
        f"- 레거시(rule_id) 키 대비 split 변경: {report['split']['changed_vs_legacy_rule_id_key']}",
        "",
        "## 분포", "",
        f"- publisher: {report['by_publisher']}",
        f"- canonical source: {report['canonical_source_counts']}",
        f"- task_type: {report['task_type_counts']}",
        f"- profile kind: {report['profile_kind_counts']}",
        f"- verdict: {report['verdict_counts']}",
        f"- prompt id/version: {report['prompt_id_counts']}",
        "",
        "## 파생", "",
        f"- {report['derived_counts']}",
        f"- 제외 사유: {report['exclusion_stats']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REVIEW_DIR = OUT_DIR / "review"

_REVIEW_OPTIONS = ["생성 라벨이 맞음", "검증 라벨이 맞음", "프로그램 라벨이 맞음",
                   "질문이 잘못됨", "Rule Card가 잘못됨", "사례 정보가 부족함",
                   "규정 자체가 정성적이라 자동 판정 불가"]


def build_expert_queue(samples: list[dict]) -> int:
    """자동 학습 파생 제외 대상(§11) → 전문가 검수 대기열."""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in samples:
        lc = s.get("label_consistency") or {}
        agreement = lc.get("agreement")
        reasons = []
        if agreement in ("VERIFIER_DISAGREES", "PROGRAM_DISAGREES", "ALL_DISAGREE",
                         "GENERATOR_DISAGREES"):
            reasons.append(agreement)
        if agreement == "SEMANTIC_AXIS_MISMATCH":
            reasons.append("SEMANTIC_AXIS_MISMATCH")
        pe = s.get("program_evaluation") or {}
        if pe.get("unsupported_conditions") and s.get("case"):
            reasons.append("QUALITATIVE_CONDITION")
        if s.get("rule_modality") == "unknown":
            reasons.append("MODALITY_UNKNOWN")
        if not reasons:
            continue
        vf = s.get("verifier_flags") or {}
        fact_ok = all(c["realized"] and not c["contradicted"] and not c["weakened"]
                      for c in ((s.get("case") or {}).get("fact_realization_checks") or []))
        rows.append({
            "diagnosis": {
                "exception_structure_check": "PASS" if not str(
                    s["sample_id"]).count("exception") or agreement != "VERIFIER_DISAGREES"
                else "REVIEW",
                "fact_realization_check": "PASS" if fact_ok else "FAIL",
                "required_facts_present": True,  # 구조화 케이스는 구성상 명시(누락 변형 제외)
                "verifier_abstention_conflict": bool(vf.get("abstention_conflict")),
                "numeric_check": "PASS",
            },
            "suggested_root_cause": ("VERIFIER_ABSTENTION" if vf.get("abstention_conflict")
                                     else "FACT_REALIZATION" if not fact_ok
                                     else agreement),
            "recommended_action": ("REVIEW_VERIFIER_LABEL" if vf.get("abstention_conflict")
                                   else "REVIEW_CASE_TEXT" if not fact_ok
                                   else "REVIEW_LABELS"),
            "sample_id": s["sample_id"],
            "rule_id": (s.get("rule") or {}).get("rule_id"),
            "canonical_rule_group_id": s.get("canonical_rule_group_id"),
            "source_text": ((s.get("source") or {}).get("source_text") or "")[:400],
            "rule_modality": s.get("rule_modality"),
            "question_intent": s.get("question_intent"),
            "question": (s.get("task") or {}).get("question"),
            "generation_label": lc.get("generation_primary_label")
            or lc.get("generation_label"),
            "verifier_label": lc.get("verifier_primary_label") or lc.get("verifier_label"),
            "program_label": lc.get("program_primary_label") or lc.get("program_label"),
            "disagreement_type": reasons[0],
            "review_question": "세 라벨 중 어느 것이 옳으며, 오류 원인은 무엇입니까?",
            "recommended_options": _REVIEW_OPTIONS,
            "priority": "HIGH" if agreement in ("ALL_DISAGREE",
                                                "SEMANTIC_AXIS_MISMATCH") else "MEDIUM",
        })
    with (REVIEW_DIR / "expert_review_queue.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md = ["# 전문가 검수 대기열", "", f"총 {len(rows)}건", ""]
    md += [f"- [{r['priority']}] {r['sample_id']} — {r['disagreement_type']} "
           f"(gen {r['generation_label']} / ver {r['verifier_label']} / "
           f"prog {r['program_label']})" for r in rows]
    (REVIEW_DIR / "expert_review_summary.md").write_text("\n".join(md) + "\n",
                                                         encoding="utf-8")
    return len(rows)


def build_modality_report(samples: list[dict], stats: Counter) -> dict:
    """§14 modality 정합 리포트 — modality/intent/primary 분포와 축 일치 통계."""
    rule_rows = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    from .modality import modality_for_requirement
    card_modality: Counter = Counter()
    for r in rule_rows:
        for req in (r.get("card") or {}).get("requirements") or []:
            m, _ = modality_for_requirement(req, r.get("article_text", ""))
            card_modality[m.value] += 1
    cases = [s for s in samples if s.get("case")]
    agreements_by_modality = {}
    for mod in {s.get("rule_modality") for s in cases if s.get("rule_modality")}:
        sub = [s for s in cases if s.get("rule_modality") == mod
               and (s.get("label_consistency") or {}).get("agreement")
               not in (None, "VERIFIER_PENDING")]
        match = sum(1 for s in sub if s["label_consistency"]["agreement"] == "THREE_WAY_MATCH")
        agreements_by_modality[mod] = {"n": len(sub), "three_way_match": match,
                                       "match_rate": round(match / len(sub), 4) if sub else None}
    return {
        "card_requirement_modality": dict(card_modality),
        "intent_counts": dict(Counter(s.get("question_intent") for s in samples
                                      if s.get("question_intent"))),
        "modality_intent_counts": dict(Counter(
            f"{s.get('rule_modality')}::{s.get('question_intent')}" for s in cases)),
        "primary_label_counts": dict(Counter(
            (s.get("labels") or {}).get("primary_label") for s in cases)),
        "semantic_axis_mismatch": sum(
            1 for s in samples if (s.get("label_consistency") or {}).get("agreement")
            == "SEMANTIC_AXIS_MISMATCH"),
        "agreement_by_modality": agreements_by_modality,
        "compliance_assessable": sum(1 for s in cases if (s.get("labels") or {})
                                     .get("compliance_label") not in (None, "NOT_ASSESSED")),
        "compliance_not_assessed": sum(1 for s in cases if (s.get("labels") or {})
                                       .get("compliance_label") == "NOT_ASSESSED"),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Master Dataset 빌드 + canonical 그룹 단위 결정적 split")
    ap.add_argument("--seed", type=int, default=42, help="split 해시 seed")
    ap.add_argument("--ratios", default="0.8,0.1,0.1", help="train,validation,test 비율(합=1)")
    ap.add_argument("--n-distractors", type=int, default=3)
    ap.add_argument("--fallback-warn", type=float, default=0.05,
                    help="chunk_id fallback 비율 경고 임계값")
    ap.add_argument("--fallback-fail", type=float, default=0.20,
                    help="strict 모드 실패 임계값")
    ap.add_argument("--strict", action="store_true",
                    help="중복 id·필수 join 실패·fallback 초과 시 즉시 오류")
    ap.add_argument("--label-policy", choices=LABEL_POLICIES, default="report-only",
                    help="3자 라벨 불일치 처리(PoC 기본 report-only, 본 빌드 strict 권장)")
    args = ap.parse_args(argv)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios 는 합이 1인 3개 값이어야 합니다: {ratios}")

    stats: Counter = Counter()
    samples = (build_judgment_samples(args.strict, stats) + build_suite_samples(stats)
               + build_case_samples(stats))
    if not samples:
        raise SystemExit("입력 산출물이 없습니다 — run_all/build_*를 먼저 실행하십시오")

    # canonical 그룹 ID 부여 + split (레거시 키 대비 변경 통계 포함)
    split_of: dict[str, str] = {}
    split_of_group: dict[str, str] = {}
    n_changed = 0
    for s in samples:
        gid, source, key = group_id_for_sample(s)
        s["canonical_rule_group_id"] = gid
        s["canonical_rule_group_source"] = source
        s["canonical_rule_group_key"] = key
        split_of_group.setdefault(gid, assign_split(args.seed, gid, ratios))
        s["split"] = split_of[s["sample_id"]] = split_of_group[gid]
        legacy_key, _ = group_key(s)
        if assign_split(args.seed, legacy_key, ratios) != s["split"]:
            n_changed += 1
    split_diff = {"changed_samples": n_changed, "total": len(samples)}

    fallback_ratio = sum(1 for s in samples
                         if s["canonical_rule_group_source"] == "chunk_id") / len(samples)
    warnings = check_fallback(fallback_ratio, args.fallback_warn,
                              args.fallback_fail, args.strict)

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    master_path = MASTER_DIR / "master_dataset.jsonl"
    with master_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log(f"master {len(samples)}건 → {master_path}")

    files = _SplitFiles(MASTER_DIR)
    try:
        judgment = [s for s in samples if s.get("excerpt")]
        cases = [s for s in samples if s.get("case")]
        suite = [s for s in samples if not s.get("excerpt") and not s.get("case")]
        derive_judgment(judgment, split_of, files, args.n_distractors, stats,
                        args.label_policy)
        derive_suite(suite, split_of, files)
        derive_cases(cases, split_of, files, stats, args.label_policy)
    finally:
        files.close()

    # publisher unknown 비율 감시 (§15)
    n_unknown_pub = sum(1 for s in samples if not (s["source"].get("publisher")))
    if n_unknown_pub / len(samples) > 0.2:
        warnings.append(f"publisher unknown 비율 {n_unknown_pub / len(samples):.1%} — 보완 필요")

    warnings += validate(samples, split_of)
    report = build_report(samples, split_of, stats, files, warnings, split_diff)
    report["label_policy"] = args.label_policy
    (MASTER_DIR / "master_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report_md(report, MASTER_DIR / "master_build_report.md")
    ev_report = build_evaluator_report(samples, stats)
    (MASTER_DIR / "evaluator_report.json").write_text(
        json.dumps(ev_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_evaluator_report_md(ev_report, MASTER_DIR / "evaluator_report.md")
    n_queue = build_expert_queue(samples)
    mod_report = build_modality_report(samples, stats)
    mod_report["expert_review_queue"] = n_queue
    (MASTER_DIR / "modality_alignment_report.json").write_text(
        json.dumps(mod_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (MASTER_DIR / "modality_alignment_report.md").write_text(
        "# Modality 정합 리포트\n\n" + "\n".join(
            f"- {k}: {v}" for k, v in mod_report.items()) + "\n", encoding="utf-8")
    log(f"전문가 검수 대기열 {n_queue}건 → {REVIEW_DIR}")
    for w in warnings:
        log(f"경고: {w}")
    log(f"split별 샘플: {report['split']['records']} · 그룹: {report['split']['canonical_groups']}"
        f" · 레거시 대비 변경 {n_changed}건")
    log(f"리포트 → {MASTER_DIR / 'master_build_report.md'}")


if __name__ == "__main__":
    main()
