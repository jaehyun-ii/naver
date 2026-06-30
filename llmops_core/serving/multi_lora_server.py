"""멀티-LoRA OpenAI 서버 — 공통 베이스 1개 + 테넌트별 어댑터 N개 동시 서빙 (transformers 백엔드).

설계의 멀티테넌시 서빙(베이스 공유 + 테넌트 LoRA 핫스왑)을 transformers+peft로 구현한다.
요청의 model 필드(논리/서빙명)로 어댑터를 선택(set_adapter), 'base'/미지정이면 베이스로 응답.
vLLM Multi-LoRA로 교체 시에도 동일 seam(/v1/chat/completions) 유지.

실행:
    python -m llmops_core.serving.multi_lora_server --base <model> \
        --adapter tenant-a=/loras/a --adapter tenant-b=/loras/b --port 8000
"""

from __future__ import annotations

import argparse
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class _Msg(BaseModel):
    role: str
    content: str


class _ChatReq(BaseModel):
    model: str | None = None
    messages: list[_Msg]
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.95


def build_app(base_model: str, adapters: dict[str, str]) -> FastAPI:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map=device
    )

    loaded: list[str] = []
    if adapters:
        from peft import PeftModel

        names = list(adapters.items())
        first_name, first_path = names[0]
        model = PeftModel.from_pretrained(model, first_path, adapter_name=first_name)
        loaded.append(first_name)
        for name, path in names[1:]:
            model.load_adapter(path, adapter_name=name)
            loaded.append(name)
    model.eval()

    app = FastAPI(title="hf-multilora")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "base": base_model, "adapters": loaded, "device": device}

    @app.get("/v1/models")
    def models() -> dict:
        ids = ["base"] + loaded
        return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}

    def _generate(messages, req: _ChatReq) -> dict:
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        prompt_tokens = int(enc["input_ids"].shape[-1])
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
                temperature=max(req.temperature, 1e-5), top_p=req.top_p,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][prompt_tokens:]
        text = tok.decode(gen, skip_special_tokens=True)
        return {"text": text, "pt": prompt_tokens, "ct": int(gen.shape[-1])}

    @app.post("/v1/chat/completions")
    def chat(req: _ChatReq) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        target = req.model if req.model in loaded else None  # 매칭 어댑터 or 베이스

        if target is None and loaded:
            with model.disable_adapter():
                r = _generate(messages, req)
            served = "base"
        else:
            if loaded:
                model.set_adapter(target)
            r = _generate(messages, req)
            served = target or "base"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": r["text"]}}],
            "usage": {"prompt_tokens": r["pt"], "completion_tokens": r["ct"],
                      "total_tokens": r["pt"] + r["ct"]},
        }

    return app


def _parse_adapters(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--adapter 형식은 name=path: {it}")
        name, path = it.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--base", default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B")
    p.add_argument("--adapter", action="append", default=[],
                   help="테넌트 어댑터 name=path (반복 가능)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    app = build_app(args.base, _parse_adapters(args.adapter))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
