"""OpenAI 호환 서빙 서버 — transformers 백엔드 (vLLM 미가용 환경용 레퍼런스 백엔드).

서빙 계층은 vLLM이 기본이나, vLLM 프리빌트가
즉시 가용하지 않으므로, 동일한 OpenAI seam(`/v1/chat/completions`)을 transformers로 제공한다.
게이트웨이·관측·평가 등 상위 경로는 백엔드와 무관하게 동일하게 동작한다.
→ 프로덕션 처리량(PagedAttention/Multi-LoRA)은 NGC vLLM 컨테이너/소스빌드로 교체.

실행(컨테이너 내부):
    python -m llmops_core.serving.hf_server \
        --model naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
        --served-name hcx-seed-0_5b --port 8000
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── OpenAI 호환 요청/응답 스키마 (최소) ──
class _Msg(BaseModel):
    role: str
    content: str


class _ChatReq(BaseModel):
    model: str | None = None
    messages: list[_Msg]
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95


def build_app(model_id: str, served_name: str) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    app = FastAPI(title=f"hf-serving:{served_name}")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": served_name, "device": device}

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: _ChatReq) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        prompt_tokens = int(enc["input_ids"].shape[-1])

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-5),
                top_p=req.top_p,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][prompt_tokens:]
        text = tok.decode(gen, skip_special_tokens=True)
        completion_tokens = int(gen.shape[-1])

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    p.add_argument("--served-name", default="hcx-seed-0_5b")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    app = build_app(args.model, args.served_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
