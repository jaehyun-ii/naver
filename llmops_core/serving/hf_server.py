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
    # vLLM과 동일한 확장 필드 — 최소 생성 토큰 수(조기 종료 방지)
    min_tokens: int = 0
    # vLLM과 동일한 확장 필드 — 템플릿 변수(예: HCX Think의 force_reasoning) 전달용
    chat_template_kwargs: dict | None = None


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

    SPECIAL_STRS = set(tok.all_special_tokens) | {
        "<|im_end|>", "<|endofturn|>", "<|stop|>", "<|endoftext|>"}

    def _strip_specials(text: str) -> str:
        for special in SPECIAL_STRS:
            text = text.replace(special, "")
        return text.strip()

    # 종료류 특수 토큰을 전부 eos로 등록하면 stop_strings 없이 단일 토큰에서 멈추고,
    # min_new_tokens 프로세서가 그 전까지 이들을 차단해 확률적 조기 종료를 막는다.
    # 모델 어휘에 실제로 존재하는 토큰만 포함(없는 토큰은 unk로 매핑되므로 걸러낸다).
    _vocab = tok.get_vocab()
    END_TOKEN_IDS = list({tok.eos_token_id} | {
        _vocab[t] for t in ("<|im_end|>", "<|endofturn|>", "<|stop|>", "<end_of_turn>")
        if t in _vocab
    })

    def _generate(input_ids, attention_mask, max_new: int, temperature: float,
                  top_p: float, min_new: int = 0, end_ids: list[int] | None = None):
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                min_new_tokens=min(min_new, max(0, max_new - 16)),
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
                eos_token_id=end_ids or END_TOKEN_IDS,
                # HCX Think 계열은 generation_config에 use_cache=false가 박혀 있어
                # 그대로 두면 KV 캐시 없이 매 토큰 전체 재계산한다 — 서빙에선 항상 켠다.
                use_cache=True,
                # 모델 generation_config에 stop_strings가 있어 tokenizer가 필수
                tokenizer=tok,
            )
        return out[0][input_ids.shape[-1]:]

    @app.post("/v1/chat/completions")
    def chat(req: _ChatReq) -> dict:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        template_kwargs = req.chat_template_kwargs or {}
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        ).to(device)
        prompt_tokens = int(enc["input_ids"].shape[-1])
        reasoning = None

        if template_kwargs.get("force_reasoning"):
            # HCX Think류는 사고 후 답변 헤더(<|im_start|>assistant\n)로 스스로
            # 전환하지 않고 턴을 끝내는 경우가 잦다 → 2단계 생성으로 전환을 강제한다.
            # 1단계: 사고 블록 생성 (<|im_end|> 등 종료류에서 정지)
            think_budget = max(256, int(req.max_tokens * 0.6))
            think_gen = _generate(enc["input_ids"], enc["attention_mask"], think_budget,
                                  req.temperature, req.top_p)
            keep = [t for t in think_gen.tolist() if t not in set(END_TOKEN_IDS)]
            reasoning = _strip_specials(tok.decode(keep, skip_special_tokens=False))
            # 2단계: 답변 헤더를 이어붙이고 답변만 생성 (min_tokens로 조기 종료 차단)
            header = tok("<|im_end|>\n<|im_start|>assistant\n",
                         add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
            ids2 = torch.cat([enc["input_ids"][0], torch.tensor(keep, device=device), header]).unsqueeze(0)
            mask2 = torch.ones_like(ids2)
            answer_gen = _generate(ids2, mask2, max(256, req.max_tokens - len(keep)),
                                   req.temperature, req.top_p, min_new=req.min_tokens)
            text = _strip_specials(tok.decode(answer_gen, skip_special_tokens=False))
            completion_tokens = len(keep) + int(answer_gen.shape[-1])
        else:
            # im_end는 자발적 think→답변 전환에 쓰이므로 종료 토큰에서 제외
            gen = _generate(enc["input_ids"], enc["attention_mask"], req.max_tokens,
                            req.temperature, req.top_p, min_new=req.min_tokens,
                            end_ids=[t for t in END_TOKEN_IDS
                                     if t != tok.convert_tokens_to_ids("<|im_end|>")])
            completion_tokens = int(gen.shape[-1])
            # 모델이 자발적으로 사고한 경우: 마지막 assistant 마커 뒤가 답변
            raw = tok.decode(gen, skip_special_tokens=False)
            if "<|im_start|>assistant" in raw:
                reasoning, _, text = raw.rpartition("<|im_start|>assistant")
                reasoning = _strip_specials(reasoning.removeprefix("/think"))
            else:
                text = raw.removeprefix("/think")
            text = _strip_specials(text)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": served_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text,
                                "reasoning_content": reasoning},
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
