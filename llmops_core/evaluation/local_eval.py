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
    tok = AutoTokenizer.from_pretrained(adapter or base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map=device
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok, device


def _seq_logprob(model, tok, device, prompt: str, completion: str) -> float:
    """prompt에 이어진 completion 토큰들의 평균 로그확률(선호 비교용)."""
    import torch

    p_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
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


def run_preference_eval(model, tok, device, cases: list[dict]) -> dict:
    """각 {prompt,chosen,rejected}에서 logp(chosen)>logp(rejected) 비율 = 선호 정확도."""
    wins, margins = [], []
    for c in cases:
        lc = _seq_logprob(model, tok, device, c["prompt"], c["chosen"])
        lr = _seq_logprob(model, tok, device, c["prompt"], c["rejected"])
        wins.append(1.0 if lc > lr else 0.0)
        margins.append(lc - lr)
    n = max(len(cases), 1)
    return {
        "preference_accuracy": round(sum(wins) / n, 4),
        "preference_margin": round(sum(margins) / n, 4),
    }


def _generate(model, tok, device, question: str, max_new_tokens: int = 128) -> str:
    import torch

    enc = tok.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    prompt_len = enc["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    p = argparse.ArgumentParser(description="로컬 평가 (judge 불필요)")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", default=None, help="LoRA 어댑터 경로(없으면 base만)")
    p.add_argument("--cases", required=True, help="평가 JSONL")
    p.add_argument("--task", choices=["reference", "preference"], default="reference",
                   help="reference: 생성·정답대비 / preference: chosen>rejected 선호정확도")
    p.add_argument("--out", default=None, help="메트릭 JSON 저장 경로(선택)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()

    model, tok, device = _load_model(args.base, args.adapter)

    if args.task == "preference":
        import json as _json

        rows = [_json.loads(ln) for ln in Path(args.cases).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        rows = [r for r in rows if r.get("prompt") and r.get("chosen") and r.get("rejected")]
        metrics = run_preference_eval(model, tok, device, rows)
        payload = {"metrics": metrics, "num_cases": len(rows)}
        if args.out:
            Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        return

    cases = _load_cases(args.cases)
    scored: list[EvalCase] = []
    for c in cases:
        ans = _generate(model, tok, device, c.question, args.max_new_tokens)
        scored.append(c.model_copy(update={"answer": ans}))

    metrics = run_reference_metrics(scored)
    payload = {"metrics": metrics, "num_cases": len(scored)}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # 실행기가 파싱하도록 마지막 줄에 단일 JSON
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
