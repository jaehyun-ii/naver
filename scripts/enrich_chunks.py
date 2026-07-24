"""Enrich article-level chunks with RAG metadata + an LLM topic label.

Turns block_recursive (article-level) chunks into records shaped for regulatory
RAG — each chunk keeps its section path and gets a one-line topic so answers can
cite "…에 따르면" and narrow queries match the right article.

Input : adaptive.jsonl  {doc_name, titles_context, chunk_pages, chunk_text, ...}
Output: enriched.jsonl   {chunk_id, doc, section, parent_section, section_path,
                          topic, pages, content}

Topic labels come from the served LLM (OpenAI-compatible, e.g. Qwen@:8008).
Stdlib only.

    python scripts/enrich_chunks.py data_chunks/adaptive.jsonl \
        --server http://localhost:8008/v1 --model teacher -o data_chunks/enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_TOPIC_PROMPT = (
    "Read the passage from a technical/regulatory document and state its single "
    "main topic as a SHORT noun phrase (≤10 words). Answer in the SAME LANGUAGE "
    "as the passage. Return ONLY the phrase, no quotes, no punctuation at the end."
    "\n\nSection: {section}\nPassage:\n\"\"\"\n{content}\n\"\"\""
)


def _clean(text: str) -> str:
    # drop markdown heading markers but keep the heading text as the first line
    return re.sub(r"(?m)^#{1,6}\s*", "", text).strip()


def _llm_topic(server: str, model: str, section: str, content: str) -> str:
    prompt = _TOPIC_PROMPT.format(section=section or "(none)", content=content[:1600])
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": 40,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{server.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read())["choices"][0]["message"]["content"]
        return out.strip().strip('"').splitlines()[0][:120]
    except Exception:  # noqa: BLE001
        return ""


def enrich_one(row: dict, idx: int, server: str, model: str) -> dict:
    path = [p for p in str(row.get("titles_context", "")).split("\n") if p.strip()]
    section = path[-1] if path else ""
    parent = path[-2] if len(path) >= 2 else ""
    content = _clean(row["chunk_text"])
    doc = row["doc_name"]
    topic = _llm_topic(server, model, section, content) if server else ""
    return {
        "chunk_id": f"{doc}_{row.get('chunk_index', idx)}",
        "doc": doc,
        "section": section,
        "parent_section": parent,
        "section_path": " > ".join(path),
        "topic": topic,
        "pages": row.get("chunk_pages", []),
        "content": content,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich chunks with metadata + LLM topic")
    ap.add_argument("chunks", help="article-level chunks JSONL")
    ap.add_argument("-o", "--output", default="data_chunks/enriched.jsonl")
    ap.add_argument("--server", default="http://localhost:8008/v1",
                    help="LLM endpoint for topic labels ('' to skip)")
    ap.add_argument("--model", default="teacher")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.chunks).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    server = args.server or ""
    print(f"{len(rows)} chunks → enriching (topic via {args.model or 'none'})")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        out = list(ex.map(lambda a: enrich_one(a[1], a[0], server, args.model),
                          enumerate(rows)))

    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with op.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    got = sum(1 for r in out if r["topic"])
    print(f"wrote {len(out)} enriched chunks -> {op}  (topic labels: {got}/{len(out)})")


if __name__ == "__main__":
    main()
