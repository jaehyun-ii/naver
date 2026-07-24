"""Synthesize DPO/SFT training data from adaptive-chunking output.

For each chunk we ask a teacher LLM (OpenAI-compatible endpoint) to:
  1. write ONE specific question answerable from the chunk + a correct answer
     grounded only in the chunk  → this is the DPO ``chosen``;
  2. given that Q/A, write a plausible-but-WRONG answer → the DPO ``rejected``.

Outputs (matching llmops_core.common.schemas):
  preference_{train,eval}.jsonl : {prompt, chosen, rejected, source}
  sft.jsonl                     : {messages:[{role,content}...], source}

Stdlib only; runs on the host and talks to the teacher over HTTP.

    python scripts/synth_dpo.py data_chunks/adaptive.jsonl \
        --server http://localhost:8008/v1 --model teacher \
        --out-dir data_chunks/synth
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "You build high-quality question-answer training data from technical "
    "maritime/engineering standards (ABS class guidance). You are precise and "
    "never invent facts that are not in the provided passage."
)

_QA_PROMPT = """Passage{section}:
\"\"\"
{chunk}
\"\"\"

Write ONE specific, self-contained question that can be answered using ONLY the passage, and a correct concise answer grounded strictly in the passage. If the passage is a table, ask about a specific value or relationship in it.

Write the question and answer in the SAME LANGUAGE as the passage (Korean passage → Korean Q/A).

Return ONLY a JSON object: {{"question": "...", "answer": "..."}}"""

_WRONG_PROMPT = """Question: {question}
Correct answer: {answer}

Write a plausible but INCORRECT answer to the question — same language, topic, tone and length as the correct answer, but with wrong specifics (numbers, names, conditions). Do not signal that it is wrong.

Return ONLY a JSON object: {{"rejected": "..."}}"""


def _post(server: str, model: str, messages: list[dict], temperature: float,
          max_tokens: int = 512, timeout: int = 120) -> str:
    body = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(f"{server.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _chat(server, model, prompt, temperature):
    return _post(server, model,
                 [{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}], temperature)


def synth_one(chunk: dict, server: str, model: str) -> dict | None:
    section = ""
    if chunk.get("titles_context"):
        section = f" (from section: {chunk['titles_context'].splitlines()[0][:120]})"
    qa_prompt = _QA_PROMPT.format(section=section, chunk=chunk["chunk_text"][:6000])

    for attempt in range(2):
        qa = _parse_json(_chat(server, model, qa_prompt, 0.7 if attempt == 0 else 0.3))
        if qa and qa.get("question") and qa.get("answer"):
            break
    else:
        return None

    wrong_prompt = _WRONG_PROMPT.format(question=qa["question"], answer=qa["answer"])
    wrong = None
    for attempt in range(2):
        w = _parse_json(_chat(server, model, wrong_prompt, 0.8 if attempt == 0 else 0.4))
        if w and w.get("rejected"):
            wrong = w["rejected"]
            break
    if not wrong:
        return None

    pages = chunk.get("chunk_pages") or []
    source = f"{chunk['doc_name']}#p{','.join(map(str, pages))}"
    prompt = (f"Use the following context to answer the question.\n\n"
              f"Context:\n{chunk['chunk_text']}\n\nQuestion: {qa['question']}")
    return {
        "prompt": prompt,
        "chosen": qa["answer"],
        "rejected": wrong,
        "question": qa["question"],
        "source": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize DPO/SFT data from chunks")
    ap.add_argument("chunks", help="adaptive-chunking JSONL (chunk_text, ...)")
    ap.add_argument("--server", default="http://localhost:8008/v1")
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--out-dir", default="data_chunks/synth")
    ap.add_argument("--min-tokens", type=int, default=60, help="skip tiny chunks")
    ap.add_argument("--limit", type=int, default=0, help="cap chunks (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.chunks).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("chunk_len", 999) >= args.min_tokens]
    if args.limit:
        rows = rows[:args.limit]
    print(f"{len(rows)} chunks → synthesizing via {args.model} @ {args.server}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, out in enumerate(ex.map(lambda c: _safe(synth_one, c, args.server, args.model), rows)):
            if out:
                results.append(out)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)} done, {len(results)} kept")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic train/eval split by source hash
    def is_eval(r: dict) -> bool:
        return (hash(r["source"] + r["question"]) % 100) < int(args.eval_frac * 100)

    pref_train, pref_eval, sft = [], [], []
    for r in results:
        pref = {"prompt": r["prompt"], "chosen": r["chosen"],
                "rejected": r["rejected"], "source": r["source"]}
        (pref_eval if is_eval(r) else pref_train).append(pref)
        sft.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["chosen"]},
        ], "source": r["source"]})

    _dump(out_dir / "preference_train.jsonl", pref_train)
    _dump(out_dir / "preference_eval.jsonl", pref_eval)
    _dump(out_dir / "sft.jsonl", sft)
    print(f"\nkept {len(results)}/{len(rows)} chunks")
    print(f"  preference_train: {len(pref_train)}  preference_eval: {len(pref_eval)}  sft: {len(sft)}")
    print(f"  -> {out_dir}")


def _safe(fn, *a):
    try:
        return fn(*a)
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        print(f"  warn: {type(exc).__name__}: {exc}")
        return None


def _dump(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
