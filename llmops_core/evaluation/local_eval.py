"""로컬 평가 CLI — 학습 어댑터(또는 병합모델)로 test set 추론·채점 (judge 불필요).

학습 컨테이너(llmops/train) 안에서 실행한다. 게이트웨이/서빙 없이 in-process로
base+LoRA 어댑터를 로드해 각 질문에 답을 생성하고, 정답 대비 결정적 메트릭을 계산한다.

    python -m llmops_core.evaluation.local_eval \
        --base naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
        --adapter /work/adapter --cases /work/eval.jsonl --out /work/metrics.json

입력 JSONL: {"question": "...", "expected": "..."} (줄 단위).
출력: stdout 마지막 줄에 JSON {"metrics": {...}, "num_cases": n}, --out 지정 시 파일로도.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmops_core.common.schemas import EvalCase
from llmops_core.evaluation.harness import run_reference_metrics


def _load_cases(path: str) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = obj.get("question") or obj.get("text")
        exp = obj.get("expected") or obj.get("response")
        if q:
            cases.append(EvalCase(id=obj.get("id") or f"c{i}", question=q, expected=exp))
    return cases


def _load_model(base: str, adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(adapter or base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=False
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok, device


def _ct(tok, messages, **kw):
    """chat template 적용 — hyperclovax(SEED-Think) 등 추론모델은 비추론(skip_reasoning)으로
    직답하게 해 짧은 정답과 채점 정합. 해당 kwarg 미지원 토크나이저는 일반 템플릿으로 폴백."""
    try:
        return tok.apply_chat_template(messages, skip_reasoning=True, **kw)
    except TypeError:
        return tok.apply_chat_template(messages, **kw)


def _messages(question: str, system: str | None) -> list[dict]:
    """평가·서빙 공통 메시지 구성 — system 프롬프트가 있으면 prepend."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return msgs


def _seq_logprob(model, tok, device, prompt: str, completion: str,
                 system: str | None = None) -> float:
    """prompt에 이어진 completion 토큰들의 평균 로그확률(선호 비교용)."""
    import torch

    p_ids = _ct(
        tok, _messages(prompt, system),
        add_generation_prompt=True, return_tensors="pt",
    ).to(device)
    c_ids = tok(completion, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_ids = torch.cat([p_ids, c_ids], dim=-1)
    with torch.no_grad():
        logits = model(input_ids).logits
    # completion 위치의 다음토큰 예측 로그확률
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    target = input_ids[0, 1:]
    start = p_ids.shape[-1] - 1
    sel = logprobs[start:, :].gather(-1, target[start:].unsqueeze(-1)).squeeze(-1)
    return float(sel.mean())


def run_preference_eval(model, tok, device, cases: list[dict],
                        system: str | None = None, system_fn=None) -> dict:
    """각 {prompt,chosen,rejected}에서 logp(chosen)>logp(rejected) 비율 = 선호 정확도.

    system_fn(prompt)이 주어지면 행별로 시스템을 동적 구성(RAG 컨텍스트 주입용).
    """
    wins, margins = [], []
    for c in cases:
        sys_c = system_fn(c["prompt"]) if system_fn else system
        lc = _seq_logprob(model, tok, device, c["prompt"], c["chosen"], sys_c)
        lr = _seq_logprob(model, tok, device, c["prompt"], c["rejected"], sys_c)
        wins.append(1.0 if lc > lr else 0.0)
        margins.append(lc - lr)
    n = max(len(cases), 1)
    return {
        "preference_accuracy": round(sum(wins) / n, 4),
        "preference_margin": round(sum(margins) / n, 4),
    }


def _generate(model, tok, device, question: str, max_new_tokens: int = 128,
              system: str | None = None) -> str:
    import torch

    enc = _ct(
        tok, _messages(question, system),
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    prompt_len = enc["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, tokenizer=tok,  # stop_strings(generation_config) 해석에 필요
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def _compose_system(system: str | None, context: str | None) -> str | None:
    """prod 시스템 프롬프트와 RAG 컨텍스트를 합성 — 서빙과 동일 규약을 공유(평가=서빙 정합)."""
    from llmops_core.rag import compose_system

    return compose_system(system, context)


def _make_rag(args):
    """--rag 지정 시 RagPipeline과 메타데이터 반환. 미지정이면 (None, None).

    서빙이 RAG 기반일 때 평가도 동일 검색·주입을 거치게 한다(선택). 빈 KB면 컨텍스트가
    비어 사실상 무영향(graceful). 임베더/백엔드는 설정(rag.*), 스토어는 --rag-store로 override.
    """
    if not getattr(args, "rag", False):
        return None, None
    from llmops_core.common.config import get_settings
    from llmops_core.rag import RagPipeline
    from llmops_core.rag.store import InMemoryVectorStore

    store = InMemoryVectorStore(persist_path=args.rag_store) if args.rag_store else None
    rag = RagPipeline(store=store)
    meta = {"used": True, "top_k": args.rag_top_k, "documents": rag.store.count(),
            "embedder": get_settings().rag.embedder}
    return rag, meta


def _build_system_fn(base_system: str | None, rag, top_k: int):
    """질문 → 시스템 프롬프트(+RAG 컨텍스트) 함수. rag 없으면 base_system 고정."""
    if rag is None:
        return lambda _q: base_system

    def system_fn(question: str) -> str | None:
        ctx = rag.build_context(rag.retrieve(question, top_k))
        return _compose_system(base_system, ctx)

    return system_fn


def _resolve_system_prompt(args) -> tuple[str | None, dict | None]:
    """평가에 적용할 시스템 프롬프트를 해석.

    우선순위: --prompt-name(스토어 라벨 조회) > --system-prompt(리터럴) > 없음.
    반환된 meta는 eval 기록에 "어떤 프롬프트로 측정했는지" 남기기 위함(평가=서빙 정합).
    """
    if args.prompt_name:
        from llmops_core.prompts import GitPromptStore

        pr = GitPromptStore().by_label(args.prompt_name, args.prompt_label)
        return pr.template, {"name": pr.name, "version": pr.version, "label": pr.label}
    if args.system_prompt:
        return args.system_prompt, {"name": None, "version": None, "label": "inline"}
    return None, None


def main() -> None:
    p = argparse.ArgumentParser(description="로컬 평가 (judge 불필요)")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None, help="LoRA 어댑터 경로(없으면 base만)")
    p.add_argument("--cases", required=True, help="평가 JSONL")
    p.add_argument("--task", choices=["reference", "preference"], default="reference",
                   help="reference: 생성·정답대비 / preference: chosen>rejected 선호정확도")
    p.add_argument("--out", default=None, help="메트릭 JSON 저장 경로(선택)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--system-prompt", default=None,
                   help="평가에 주입할 시스템 프롬프트(서빙과 동일 적용; 평가=서빙 정합)")
    p.add_argument("--prompt-name", default=None,
                   help="GitPromptStore 프롬프트명 — 라벨로 해석해 --system-prompt 대신 사용")
    p.add_argument("--prompt-label", default="prod", help="--prompt-name 사용 시 라벨")
    p.add_argument("--rag", action="store_true",
                   help="평가에 RAG 검색·컨텍스트 주입 적용(서빙이 RAG 기반일 때 — 평가=서빙 정합)")
    p.add_argument("--rag-top-k", type=int, default=4, help="RAG 검색 문서 수")
    p.add_argument("--rag-store", default=None,
                   help="RAG 벡터스토어 경로(미지정 시 설정 rag.persist_path)")
    args = p.parse_args()

    system, prompt_meta = _resolve_system_prompt(args)
    rag, rag_meta = _make_rag(args)
    system_fn = _build_system_fn(system, rag, args.rag_top_k)
    model, tok, device = _load_model(args.base, args.adapter)

    if args.task == "preference":
        import json as _json

        rows = [_json.loads(ln) for ln in Path(args.cases).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        rows = [r for r in rows if r.get("prompt") and r.get("chosen") and r.get("rejected")]
        metrics = run_preference_eval(model, tok, device, rows, system, system_fn=system_fn)
        payload = {"metrics": metrics, "num_cases": len(rows), "prompt": prompt_meta,
                   "rag": rag_meta}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return

    cases = _load_cases(args.cases)
    scored: list[EvalCase] = []
    for c in cases:
        ans = _generate(model, tok, device, c.question, args.max_new_tokens,
                        system_fn(c.question))
        scored.append(c.model_copy(update={"answer": ans}))

    metrics = run_reference_metrics(scored)
    payload = {"metrics": metrics, "num_cases": len(scored), "prompt": prompt_meta,
               "rag": rag_meta}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # 실행기가 파싱하도록 마지막 줄에 단일 JSON
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
