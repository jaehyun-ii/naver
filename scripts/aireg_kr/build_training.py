"""AIReg-KR 벤치마크 산출물 → RAFT 구성 SFT/DPO 학습 데이터 변환 (프로토타입).

RAFT(Retrieval-Augmented Fine-Tuning)는 학습 기법이 아니라 데이터 구성 레시피:
실제 RAG 추론과 동일하게 입력에 [정답 조항 + distractor 조항]을 섞어 주고,
근거 인용 기반 판정을 정답으로 학습시킨다. 트레이너 입장에서는 평범한 SFT/DPO.

- 입력: data_aireg/{rule_cards,excerpts,annotations}.jsonl
- distractor 소스: data_chunks/KR 청크(같은 장의 혼동 조항 1 + 타 장 무작위 2)
- no-golden(검색 실패 내성, 정답='판단 불가')은 확률이 아니라 **결정적 층화 할당**:
  조당 5개 슬롯(케이스×종류) 중 정확히 1개를 조 순번에 따라 순환 지정 → 전체 20%가
  라벨·난이도별로 고르게 유지되고 재현 가능하다.
- DPO 생성 규칙: golden이 포함된 excerpt 1건당 rejected 2~3개(부적합/판단불가 라벨은 3개)
  → 조당 4 golden excerpt 기준 DPO 10~12쌍. 템플릿 기반 유형: 무근거 단정 / 자기선언
  신봉 / distractor 낚임. (적용대상·예외조건 누락형은 내용 의존적이라 전체 실행 시
  LLM 생성으로 추가 예정.)

    python -m scripts.aireg_kr.build_training [--n-distractors 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
import re

from .answer_format import UNCERTAIN_DECISION, format_regulation_answer
from .common import CHUNK_DIR, OUT_DIR, _is_requirement_parent, load_jsonl, log, nfc

TRAIN_DIR = OUT_DIR / "training"

JUDGMENT_KO = {"compliant": "적합", "non_compliant": "부적합", "uncertain": "판단 불가"}


# ── distractor 풀 ────────────────────────────────────────────────────────
def load_chunk_pool() -> list[dict]:
    """전 발행처의 실질 요건 조항 — 실제 RAG 인덱스(전 선급 통합)와 분포를 맞춘다.

    링크된 표를 조 본문에 병합한다(_attach_tables) — 골든 조항(라우터 경유)은
    표가 병합돼 있는데 distractor만 표가 빠지면 문서 간 비대칭이 생기고, 서빙
    parent store도 표를 병합하므로 실제 검색 문맥과도 이 쪽이 일치한다."""
    from .common import _attach_tables  # 지연 임포트(공개 API 아님)
    pool = []
    for soc_dir in sorted(CHUNK_DIR.iterdir()):
        if not soc_dir.is_dir() or soc_dir.name.startswith("_"):
            continue
        for f in soc_dir.glob("*_chunks.jsonl"):
            parents, tables = [], {}
            for l in f.open(encoding="utf-8"):
                if '"parent"' not in l and '"table"' not in l:
                    continue
                r = json.loads(l)
                if r.get("chunk_type") == "table":
                    tables[r["chunk_id"]] = r
                elif _is_requirement_parent(r, soc_dir.name, 150):
                    r["publisher"] = soc_dir.name
                    parents.append(r)
            _attach_tables(parents, tables)
            pool.extend(parents)
    return pool


# ── 실검색(RAG) 문서 구성 — 학습 입력을 서빙 검색 분포와 일치시킨다 ──────
# no-golden 비율 밴드: 실측 검색 실패율을 그대로 쓰되 이 범위로 보정
# (너무 낮으면 검색 실패 내성을 못 배우고, 너무 높으면 판정 학습량이 준다)
NO_GOLDEN_MIN, NO_GOLDEN_MAX = 0.15, 0.25


def build_doc_index(pool: list[dict]) -> dict[tuple[str, str], dict]:
    """Qdrant parent_chunk_id → 코퍼스 요건 조 행. (publisher, chunk_id) 키."""
    return {(r.get("publisher", "?"), nfc(r["chunk_id"])): r for r in pool}


class RagContext:
    """excerpt/질문 텍스트로 실검색해 distractor·골든 포함 여부를 산출.

    - distractor: 검색 상위의 비골든 조(요건 parent) — 실제 hard negative
    - 골든 포함 여부: 서빙 top-k(n+1) 안에 골든 조가 실제로 잡히는가(실측)
    - 동등 요건(타 선급·타 편 동일 조항, 유사도 ≥ 0.85)은 제외 — no-golden
      라벨 오염 방지. 조당 1회 검색해 캐시한다.
    """

    def __init__(self, pool: list[dict]):
        from . import retrieval  # 지연 임포트 — 미가용 환경에서 모듈 로드 무해
        self.rt = retrieval
        self.idx = build_doc_index(pool)
        self._eq: dict[str, set[str]] = {}

    def _equivalents(self, rule: dict) -> set[str]:
        gid = nfc(rule["chunk_id"])
        if gid not in self._eq:
            eq = {nfc(p) for p in self.rt.equivalent_parents(
                rule.get("article_text") or rule.get("content", ""), gid)}
            # 규칙↔지침 미러는 링크 메타로 결정적 제외 — 임베딩 유사도(0.80)에만
            # 의존하면 미러가 경계 아래로 빠질 때 distractor로 새어 들어온다
            row = self.idx.get((rule.get("publisher", "KR"), gid)) or rule
            for k in ("linked_guidance_chunk_id", "linked_rule_chunk_id"):
                v = row.get(k)
                if v and str(v) != "None":
                    eq.add(nfc(str(v)))
            self._eq[gid] = eq
        return self._eq[gid]

    def retrieve(self, query: str, rule: dict, n: int,
                 exclude: set[str] = frozenset()) -> tuple[list[dict], bool]:
        """(distractor 조 행 n개, 골든이 서빙 top-(n+1)에 실제 포함되는가)."""
        gid, pub = nfc(rule["chunk_id"]), rule.get("publisher", "KR")
        hits = self.rt.search_parents(query, k=n + 1 + 12)
        eq = self._equivalents(rule)
        golden_hit = any(nfc(h["parent_chunk_id"]) == gid for h in hits[:n + 1])
        docs = []
        for h in hits:
            pid = nfc(h["parent_chunk_id"])
            if pid == gid or pid in eq or pid in exclude:
                continue
            row = self.idx.get((h.get("publisher") or pub, pid)) \
                or self.idx.get((pub, pid))
            if row is None:
                continue  # 요건 parent가 아닌 조(정의·목차 등) — 라벨 안전을 위해 제외
            docs.append(row)
            if len(docs) >= n:
                break
        return docs, golden_hit


def pick_distractors(pool: list[dict], rule: dict, n: int, rng: random.Random) -> list[dict]:
    """같은 발행처·같은 장의 혼동 조항 1개 + 나머지 무작위 — 검색 노이즈 분포를 흉내낸다."""
    golden_id, pub = rule["chunk_id"], rule.get("publisher", "KR")
    key = lambda c: (c.get("publisher"), c.get("_source_file", ""), c["chunk_id"])  # noqa: E731
    same = [c for c in pool if c["chunk_id"] != golden_id
            and c.get("publisher") == pub
            and rule.get("chapter_no") is not None
            and str(c.get("chapter_no")) == str(rule.get("chapter_no"))]
    same_keys = {key(c) for c in same}
    other = [c for c in pool if c["chunk_id"] != golden_id and key(c) not in same_keys]
    picked = (rng.sample(same, min(1, len(same))) if same else []) + \
        rng.sample(other, min(len(other), n - min(1, len(same))))
    return picked[:n]


# ── 프롬프트/정답 조립 ──────────────────────────────────────────────────
def fmt_doc(i: int, section_path, content: str) -> str:
    path = " > ".join(section_path) if isinstance(section_path, list) else section_path
    return f"[문서 {i}] {path}\n{content}"


def build_prompt(rule: dict, distractors: list[dict], excerpt: dict,
                 rng: random.Random, include_golden: bool) -> tuple[str, int]:
    docs = [(d.get("section_path"), d["content"]) for d in distractors]
    golden_pos = -1
    if include_golden:
        docs.append((rule["section_path"], rule["article_text"]))
    rng.shuffle(docs)
    if include_golden:
        golden_pos = next(i for i, (sp, c) in enumerate(docs, 1) if c == rule["article_text"])
    doc_block = "\n\n".join(fmt_doc(i, sp, c) for i, (sp, c) in enumerate(docs, 1))
    prompt = (
        "당신은 선급 규정 검토 전문가입니다. 아래 검색된 규정 조항들과 기술문서 발췌를 보고, "
        "기술문서의 설계·구성이 관련 조항의 요건을 충족하는지 판정하십시오.\n"
        "반드시 관련 조항을 골라 원문을 인용해 근거를 제시하고, 제공된 조항만으로 판단이 "
        "불가능하면 '판단 불가'와 함께 필요한 정보를 명시하십시오.\n\n"
        f"[검색된 규정 조항]\n{doc_block}\n\n"
        f"[기술문서 발췌]\n{excerpt['text']}"
    )
    return prompt, golden_pos


def pick_quote(article_text: str, ann: dict) -> str:
    """조항 원문에서 어노테이션 근거와 가장 겹치는 문장을 인용문으로 선택."""
    sents = [s.strip() for s in re.split(r"(?<=[다음])\.\s*", article_text) if len(s.strip()) > 20]
    basis = " ".join(e.get("evidence", "") for e in ann.get("critical_evidence", []))
    basis += ann.get("compliance_reason", "")
    basis_tokens = set(re.findall(r"[가-힣]{2,}", basis))

    def score(s: str) -> int:
        return len(basis_tokens & set(re.findall(r"[가-힣]{2,}", s)))

    best = max(sents, key=score) if sents else article_text[:120]
    return best[:160]


def build_chosen(rule: dict, excerpt: dict, ann: dict, golden_pos: int) -> str:
    """공통 답변 형식(answer_format)으로 정답 조립 — GRPO 형식 보상과 같은 기준."""
    label = excerpt["target_label"]
    quote = pick_quote(rule["article_text"], ann)
    evidence = [f"[문서 {golden_pos}] {rule['section_path']}", f"인용: 「{quote}」"]
    review = ann.get("compliance_reason", "")
    if label == "uncertain":
        missing = ann.get("missing_information") or ["판정에 필요한 세부 정보"]
        return format_regulation_answer(
            decision=UNCERTAIN_DECISION, evidence=evidence,
            condition_review=review, missing_information=missing)
    return format_regulation_answer(
        decision=JUDGMENT_KO[label], evidence=evidence, condition_review=review)


def build_no_golden_answer(rule: dict) -> str:
    topic = rule["section_path"].split(">")[-1].strip()
    return format_regulation_answer(
        decision=UNCERTAIN_DECISION,
        evidence=(f"제공된 조항 중에 이 기술문서의 판정 기준이 되는 조항"
                  f"({topic} 관련 요건)이 없습니다."),
        missing_information=[f"{topic} 요건 조항 추가 검색"])


def build_rejected(rule: dict, excerpt: dict, distractors: list[dict],
                   golden_pos: int, rng: random.Random) -> list[tuple[str, str]]:
    """(유형, rejected 텍스트) — unacceptable_answers 유형을 그대로 실현.

    rejected도 chosen과 같은 외형(공통 형식)을 쓴다 — DPO가 형식이 아니라
    내용(무근거·자기선언·오인용)으로 선호를 학습하게 하기 위함.
    """
    label = excerpt["target_label"]
    flipped = "부적합" if label == "compliant" else "적합"
    out = [
        ("무근거 단정", format_regulation_answer(
            decision=flipped, evidence="문서의 전반적인 구성을 볼 때 그렇게 판단됩니다.")),
    ]
    if label in ("non_compliant", "uncertain"):
        out.append(("자기선언 신봉", format_regulation_answer(
            decision="적합",
            evidence="문서에 설계가 적절히 구성되어 시험을 통과했다고 기술되어 있으므로 "
                     "요건을 충족하는 것으로 판단됩니다.")))
    if distractors:
        d = rng.choice(distractors)
        path = " > ".join(d.get("section_path") or [])
        out.append(("distractor 낚임", format_regulation_answer(
            decision=JUDGMENT_KO[label] if label != "uncertain" else "적합",
            evidence=f"근거 조항: {path}\n해당 조항에 따르면 본 설계는 요건을 충족합니다.")))
    return out


# ── 스위트 태스크(전 트랙) → SFT/DPO 변환 ──────────────────────────────
APPLIC_KO = {"applicable": "적용", "not_applicable": "미적용", "conditional": "조건부 적용"}
# 스위트 내부 키 → 정식 task_type (answer_format의 판정형/정보형 라우팅 기준).
# 전부 answer_format의 기존 등록 task_type에 매핑한다(신규 등록 없이 라우팅 성립).
SUITE_TASK_TYPE = {"spec": "direct_qa", "applic": "applicability",
                   "xref": "cross_reference_lookup", "compare": "comparison",
                   "def_link": "cross_reference_lookup", "precedence": "direct_qa",
                   "unit_convert": "requirement_satisfaction",
                   "table_lookup": "extractive_qa", "hierarchy": "applicability"}
# 스위트 소스 파일 → 내부 태스크 키 (build_suite_qa.TRACK_FILES와 정합)
SUITE_SOURCES = [
    ("spec_qa.jsonl", "spec"), ("applicability_qa.jsonl", "applic"),
    ("crossref_qa.jsonl", "xref"), ("compare_qa.jsonl", "compare"),
    ("def_link_qa.jsonl", "def_link"), ("precedence_qa.jsonl", "precedence"),
    ("unit_convert_qa.jsonl", "unit_convert"),
    ("table_lookup_qa.jsonl", "table_lookup"), ("hierarchy_qa.jsonl", "hierarchy"),
]
# 판정형 트랙의 decision 한국어 — chosen/rejected 라벨 표기
JUDGE_DECISION_KO = {"unit_convert": JUDGMENT_KO, "hierarchy": APPLIC_KO}

SUITE_INSTR = ("당신은 선급 규정 전문가입니다. 아래 검색된 규정 조항들을 근거로 질문에 답하십시오.\n"
               "반드시 근거 조항을 지목하고 원문을 인용하십시오. 제공된 조항으로 답할 수 없으면 "
               "'제공된 조항으로는 답할 수 없다'고 답하십시오.")


def build_suite_prompt(row: dict, distractors: list[dict],
                       rng: random.Random) -> tuple[str, list[int]]:
    """다중 evidence(상호참조·비교는 golden 조항 2개) 지원 — golden 위치 목록 반환."""
    golden = [(e["section_path"], e["article_text"]) for e in row["evidence"]]
    docs = [(d.get("section_path"), d["content"]) for d in distractors] + golden
    rng.shuffle(docs)
    golden_texts = {t for _, t in golden}
    positions = [i for i, (sp, c) in enumerate(docs, 1) if c in golden_texts]
    doc_block = "\n\n".join(fmt_doc(i, sp, c) for i, (sp, c) in enumerate(docs, 1))
    prompt = f"{SUITE_INSTR}\n\n[검색된 규정 조항]\n{doc_block}\n\n[질문]\n{row['question']}"
    return prompt, positions


def suite_rows(pool: list[dict], sft_f, dpo_f, route=None,
               rag: "RagContext | None" = None) -> tuple[int, int]:
    """스위트 QA(전 트랙) → RAFT SFT(+판정형 DPO). 존재하는 파일만 처리.

    route: callable(row) -> (sft_f, dpo_f) | None — build_master가 split별 파일로
    라우팅할 때 사용. None을 반환하면 해당 행은 건너뛴다(미배정 split 등).
    rag: RagContext — 있으면 distractor를 실검색 hard negative로 뽑는다.
    """
    from .verify_suite import load_rejected  # 지연 임포트(순환 방지)
    rejected = load_rejected()
    n_sft = n_dpo = 0
    for fname, task in SUITE_SOURCES:
        for row in load_jsonl(OUT_DIR / fname):
            if "trainable" in row:
                # 신 스키마(verify-loop 산출): status 단일화 — ACCEPT만 투입
                if not row["trainable"]:
                    continue
            elif row.get("needs_review") or row["question_id"] in rejected:
                continue  # 구 스키마: 인용 불일치·블라인드 검증 REJECT 제외
            if route is not None:
                dest = route(row)
                if dest is None:
                    continue
                sft_f, dpo_f = dest
            # 학습 투입 질문은 자연 포장 — [발췌]/[상황] 대괄호 라벨은 벤치마크
            # 관례이지 사용자 언어가 아니다(실서빙=채팅 붙여넣기). 원본 파일은
            # 라벨 유지(검수·게이트 파싱), 여기서만 프로그램 재포장(본문 축자).
            from .style_augment import naturalize_question
            row = {**row, "question": naturalize_question(
                row["question"], row["question_id"])}
            ev = row["evidence"][0]
            rng = random.Random(int(hashlib.sha1(row["question_id"].encode()).hexdigest(), 16))
            # distractor의 '같은 장' 휴리스틱용 최소 rule 정보
            pseudo_rule = {"chunk_id": ev["chunk_id"],
                           "publisher": row["metadata"].get("publisher", "KR"),
                           "article_text": ev.get("article_text", ""),
                           "chapter_no": None}
            n_dis = max(2, 4 - len(row["evidence"]))
            if rag is not None:
                # 실검색 hard negative — 모든 evidence 조는 제외(다중 근거 트랙)
                ev_ids = {nfc(e["chunk_id"]) for e in row["evidence"]}
                distractors, _ = rag.retrieve(row["question"], pseudo_rule,
                                              n_dis, exclude=ev_ids)
                if len(distractors) < n_dis:  # 검색 부족분은 휴리스틱 보충
                    have = {(d.get("publisher"), nfc(d["chunk_id"])) for d in distractors}
                    distractors += [d for d in pick_distractors(
                        pool, pseudo_rule, n_dis + 2, rng)
                        if (d.get("publisher"), nfc(d["chunk_id"])) not in have
                        ][:n_dis - len(distractors)]
            else:
                distractors = pick_distractors(pool, pseudo_rule, n_dis, rng)
            prompt, positions = build_suite_prompt(row, distractors, rng)
            cite = ", ".join(f"[문서 {p}]" for p in positions)
            paths = " / ".join(e["section_path"] for e in row["evidence"])
            if task in ("spec", "table_lookup"):
                chosen = (f"근거 조항: {cite} {paths}\n"
                          f"인용: 「{ev.get('quote', '')}」\n답변: {row['gold_answer']}")
            elif task in ("applic", "hierarchy", "unit_convert"):
                # 판정형 → 공통 규정 답변 형식(select_answer_format 라우팅과 일치)
                ko = JUDGE_DECISION_KO.get(task, APPLIC_KO)
                chosen = format_regulation_answer(
                    decision=ko[row["expected_judgment"]],
                    evidence=[f"{cite} {paths}"] + [f"인용: 「{ev['quote']}」"] * bool(ev.get("quote")),
                    condition_review=row["gold_answer"])
            else:  # xref/compare/def_link/precedence — 정보형: golden 조항을 모두 인용해 결합 답변
                quotes = "\n".join(f"인용: 「{e.get('quote', '')}」"
                                   for e in row["evidence"] if e.get("quote"))
                chosen = f"근거 조항: {cite} {paths}\n{quotes}\n답변: {row['gold_answer']}"
            sft_f.write(json.dumps({
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": chosen}],
                "source": f"aireg_kr_{task}:{row['question_id']}",
            }, ensure_ascii=False) + "\n")
            n_sft += 1
            if task in ("applic", "hierarchy", "unit_convert"):
                # 라벨이 gold이므로 반대 판정은 구성상 오답 — 무근거 단정형 rejected.
                # chosen과 같은 formatter를 써서 형식이 아닌 내용으로 선호를 학습.
                # (unit_convert 라벨은 프로그램 계산이라 flip의 오답성이 보장된다)
                ko = JUDGE_DECISION_KO.get(task, APPLIC_KO)
                wrong = [k for k in ko if k != row["expected_judgment"]]
                flipped = ko[rng.choice(wrong)]
                basis = "선박 개요를" if task == "applic" else "문서 내용을"
                dpo_f.write(json.dumps({
                    "prompt": prompt, "chosen": chosen,
                    "rejected": format_regulation_answer(
                        decision=flipped,
                        evidence=f"{basis} 볼 때 그렇게 판단됩니다."),
                    "source": f"aireg_kr_{task}:{row['question_id']}:무근거 단정",
                }, ensure_ascii=False) + "\n")
                n_dpo += 1
    return n_sft, n_dpo


# ── main ────────────────────────────────────────────────────────────────
# 조당 excerpt 슬롯 — run_all.KIND_PLAN과 동일해야 한다
KIND_SLOTS = [(1, "compliant"), (1, "subtle_nc"), (1, "insufficient"),
              (2, "compliant"), (2, "clear_nc")]
# run_all.plan_for의 결정적 대체(경계값·예외 kind)를 기본 슬롯으로 되돌리는 매핑 —
# no-golden 슬롯 매칭이 대체 여부와 무관하게 동작해야 20% 층화가 유지된다
BASE_KIND = {"boundary_c": "compliant", "exception_nm": "subtle_nc"}


def legacy_include_golden(ex: dict, rule_order: dict[str, int],
                          n_ex_per_rule: Counter) -> bool:
    """결정적 층화 할당(20% no-golden) — 검색 미가용 시의 기존 로직.

    - 조당 5 excerpt(파일럿 모드): 조 순번에 따라 5개 슬롯 중 1개 지정
    - 조당 1 excerpt(문서당 1문항 모드): 5개 조마다 1개 조 지정
    """
    if n_ex_per_rule[ex["rule_id"]] == 1:
        return rule_order[ex["rule_id"]] % 5 != 0
    no_golden_slot = KIND_SLOTS[rule_order[ex["rule_id"]] % len(KIND_SLOTS)]
    base_kind = BASE_KIND.get(ex["kind"], ex["kind"])
    return (ex["case_no"], base_kind) != no_golden_slot


def band_adjust(plans: list[dict]) -> None:
    """실측 no-golden 비율을 [NO_GOLDEN_MIN, NO_GOLDEN_MAX] 밴드로 보정.

    검색 recall이 매우 높으면 실패 내성 학습량이 부족하고, 낮으면 판정 학습량이
    준다. 초과분/부족분만 결정적(균등 간격)으로 뒤집고 태그를 남긴다."""
    n = len(plans)
    if not n:
        return
    misses = [p for p in plans if not p["include_golden"]]
    lo, hi = round(n * NO_GOLDEN_MIN), round(n * NO_GOLDEN_MAX)
    if len(misses) > hi:  # 검색 실패 과다 → 초과분은 골든 강제 포함
        flip = len(misses) - hi
        step = max(1, len(misses) // flip)
        for i, p in enumerate(misses):
            if flip and i % step == 0:
                p["include_golden"], p["tag"] = True, "golden_forced"
                flip -= 1
    elif len(misses) < lo:  # 검색 성공 과다 → 부족분은 no-golden 강제
        hits = [p for p in plans if p["include_golden"]]
        flip = lo - len(misses)
        step = max(1, len(hits) // flip)
        for i, p in enumerate(hits):
            if flip and i % step == 0:
                p["include_golden"], p["tag"] = False, "ng_forced"
                flip -= 1


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-distractors", type=int, default=3)
    ap.add_argument("--no-retrieval", action="store_true",
                    help="실검색 비활성 — 휴리스틱 distractor·결정적 20% no-golden")
    args = ap.parse_args(argv)

    rule_rows = load_jsonl(OUT_DIR / "rule_cards.jsonl")
    rules = {r["id"]: r for r in rule_rows}
    rule_order = {r["id"]: i for i, r in enumerate(rule_rows)}
    anns = {a["id"]: a for a in load_jsonl(OUT_DIR / "annotations.jsonl")}
    excerpts = load_jsonl(OUT_DIR / "excerpts.jsonl")
    # 독립 검증(verify.py) 결과가 있으면 REJECT 제외 — 미검증 행은 통과(파일럿 호환)
    verdicts = {v["id"]: v for v in load_jsonl(OUT_DIR / "verdicts.jsonl")}
    if verdicts:
        before = len(excerpts)
        excerpts = [e for e in excerpts
                    if verdicts.get(e["id"], {}).get("verdict") != "REJECT"]
        print(f"검증 필터: {before} → {len(excerpts)} (REJECT {before - len(excerpts)}건 제외)")
    n_ex_per_rule = Counter(e["rule_id"] for e in excerpts)
    pool = load_chunk_pool()

    rag = None
    if not args.no_retrieval:
        from . import retrieval
        if retrieval.available():
            rag = RagContext(pool)
            log(f"실검색 모드: {retrieval.QDRANT_URL}/{retrieval.COLLECTION}")
        else:
            log("Qdrant/bge-m3 미가용 — 휴리스틱 distractor·결정적 no-golden 폴백")

    # ── 1차 패스: excerpt별 문서 구성·골든 포함 결정 ────────────────────
    plans = []
    for ex in excerpts:
        rule, ann_row = rules.get(ex["rule_id"]), anns.get(ex["id"])
        if not rule or not ann_row:
            continue
        rng = random.Random(int(hashlib.sha1(ex["id"].encode()).hexdigest(), 16))
        if rag is not None:
            # 서빙과 동일: 기술문서 발췌를 질의로 검색 → 실측 hard negative
            distractors, golden_hit = rag.retrieve(ex["text"], rule, args.n_distractors)
            if len(distractors) < args.n_distractors:
                have = {(d.get("publisher"), nfc(d["chunk_id"])) for d in distractors}
                distractors += [d for d in pick_distractors(
                    pool, rule, args.n_distractors + 2, rng)
                    if (d.get("publisher"), nfc(d["chunk_id"])) not in have
                    ][:args.n_distractors - len(distractors)]
            include, tag = golden_hit, ("" if golden_hit else "ng_natural")
        else:
            distractors = pick_distractors(pool, rule, args.n_distractors, rng)
            include = legacy_include_golden(ex, rule_order, n_ex_per_rule)
            tag = "" if include else "ng_rotation"
        plans.append({"ex": ex, "rule": rule, "ann": ann_row["annotation"],
                      "rng": rng, "distractors": distractors,
                      "include_golden": include, "tag": tag})

    if rag is not None:
        band_adjust(plans)
        tags = Counter(p["tag"] for p in plans if p["tag"])
        log(f"no-golden 구성: {dict(tags)} / 전체 {len(plans)}")

    # ── 2차 패스: 조립·기록 ─────────────────────────────────────────────
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    sft_f = (TRAIN_DIR / "sft.jsonl").open("w", encoding="utf-8")
    dpo_f = (TRAIN_DIR / "preference.jsonl").open("w", encoding="utf-8")

    n_sft = n_dpo = 0
    for p in plans:
        ex, rule, ann, rng = p["ex"], p["rule"], p["ann"], p["rng"]
        include_golden = p["include_golden"]
        prompt, golden_pos = build_prompt(rule, p["distractors"], ex, rng, include_golden)
        chosen = (build_chosen(rule, ex, ann, golden_pos) if include_golden
                  else build_no_golden_answer(rule))

        suffix = "" if include_golden else ":no_golden"
        if p["tag"]:
            suffix += f":{p['tag']}"
        sft_f.write(json.dumps({
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": chosen}],
            "source": f"aireg_kr:{ex['id']}{suffix}",
        }, ensure_ascii=False) + "\n")
        n_sft += 1

        if include_golden:
            for kind, rejected in build_rejected(rule, ex, p["distractors"],
                                                 golden_pos, rng):
                dpo_f.write(json.dumps({
                    "prompt": prompt, "chosen": chosen, "rejected": rejected,
                    "source": f"aireg_kr:{ex['id']}:{kind}",
                }, ensure_ascii=False) + "\n")
                n_dpo += 1

    s2, d2 = suite_rows(pool, sft_f, dpo_f, rag=rag)
    sft_f.close(); dpo_f.close()
    log(f"SFT {n_sft + s2}건(판정 {n_sft} + 스위트 {s2}) → {TRAIN_DIR/'sft.jsonl'} | "
        f"DPO {n_dpo + d2}쌍(판정 {n_dpo} + 적용성 {d2}) → {TRAIN_DIR/'preference.jsonl'}")


if __name__ == "__main__":
    main()
