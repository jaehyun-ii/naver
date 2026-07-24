"""SFT/정렬 학습기 — unsloth + trl + peft 모듈 조합 (Axolotl/LLaMA-Factory 통째 미채용).

동일 기능(고속 LoRA, 분산, DPO/GRPO)을 라이브러리 모듈로 직접 조립한다.
출력은 표준 HF 체크포인트 → vLLM 서빙·MLflow 등록과 일관. import는 lazy(GPU 노드).

경로 두 가지:
- `run_sft`/`run_dpo`: unsloth(고속·4bit) 경로.
- `run_sft_peft`: unsloth 없이 trl+peft LoRA(bf16 또는 QLoRA 4bit). H100/H200 운영 경로.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from llmops_core.common.errors import OptionalDependencyError


@dataclass
class SFTJobConfig:
    base_model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
    output_dir: str = "s3://dev-demo-models/run"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    extra_sft_args: dict = field(default_factory=dict)


def _build_model(cfg: SFTJobConfig):
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("unsloth", "training") from exc

    model, tok = FastLanguageModel.from_pretrained(
        cfg.base_model, max_seq_length=cfg.max_seq_length, load_in_4bit=cfg.load_in_4bit
    )
    model = FastLanguageModel.get_peft_model(model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha)
    return model, tok


def run_sft(cfg: SFTJobConfig, train_dataset, mlflow_callback=None):
    """SFTTrainer 학습 실행. mlflow_callback로 메트릭을 tracking에 연결."""
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl", "training") from exc

    model, tok = _build_model(cfg)
    args = SFTConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        **cfg.extra_sft_args,
    )
    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=train_dataset, args=args)
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    result = trainer.train()
    trainer.save_model(cfg.output_dir)  # 표준 HF 체크포인트 + 어댑터
    return result


# ── trl+peft LoRA 경로 (운영 학습기) ──
@dataclass
class PeftSFTConfig:
    """trl+peft LoRA SFT 설정 — bf16 기본, QLoRA(4bit)는 load_in_4bit로 선택."""

    base_model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
    output_dir: str = "outputs/adapter"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = False  # PET 변형: DoRA(weight-decomposed LoRA). peft use_dora=True
    load_in_4bit: bool = False  # QLoRA(4bit). bitsandbytes 필요(미설치 시 가드 에러)
    hf_token: str | None = None  # gated 모델(예: SEED 1.5B) 접근 토큰. 미지정 시 env HF_TOKEN
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    dpo_beta: float = 0.1  # DPO 선호-KL 정규화 강도(β). HPO 탐색 대상
    logging_steps: int = 1
    max_steps: int = -1  # >0이면 epoch 무시 — 빠른 루프 검증용
    extra_sft_args: dict = field(default_factory=dict)


def _load_base_for_peft(cfg: "PeftSFTConfig"):
    """bf16(기본) 또는 QLoRA(4bit) 베이스 로드 + gated 모델 토큰 처리. (model, tok) 반환.

    4bit는 bitsandbytes를 요구한다. 미설치 환경에서 load_in_4bit=True면 명확한 안내와 함께 실패한다.
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    # transformers 5.9+는 hyperclovax를 네이티브 지원 → remote code 비활성(오프라인 캐시엔 .py 없음).
    # auto_map이 있어도 trust_remote_code=False면 네이티브 HyperCLOVAXConfig/ForCausalLM로 로드된다.
    # 예외: Think-32B 등 네이티브 미지원 아키텍처는 저장소 동봉 코드가 필요 —
    # TRAIN_TRUST_REMOTE_CODE=1로만 허용(기본 비활성, 오프라인 스냅샷 코드 실행).
    auth: dict = {"trust_remote_code":
                  os.environ.get("TRAIN_TRUST_REMOTE_CODE") == "1"}
    if token:
        auth["token"] = token

    tok = AutoTokenizer.from_pretrained(cfg.base_model, **auth)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # TRAIN_DEVICE_MAP=auto: 단일 GPU에 안 실리는 베이스(예: 32B bf16)를 가시
    # GPU들에 파이프라인 분산 적재(H100 2장 페어 학습용). 기본은 단일 "cuda".
    device_map = os.environ.get("TRAIN_DEVICE_MAP",
                                "cuda" if torch.cuda.is_available() else None)
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map, **auth)
    # 일부 모델(32B)의 커스텀 코드가 flash_attn을 기본 요구 — 미설치 환경에선
    # TRAIN_ATTN_IMPL=sdpa 등으로 오버라이드.
    if os.environ.get("TRAIN_ATTN_IMPL"):
        kwargs["attn_implementation"] = os.environ["TRAIN_ATTN_IMPL"]
    if cfg.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise OptionalDependencyError(
                "bitsandbytes(QLoRA 4bit)", "training"
            ) from exc  # bitsandbytes 미설치 → 설치 후 QLoRA 사용
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **kwargs)
    if cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model, tok


def _to_message_rows(examples) -> list[dict]:
    """SFTExample(pydantic) | dict 혼용 → trl conversational 포맷 {"messages": [...]} 리스트."""
    rows: list[dict] = []
    for ex in examples:
        msgs = ex.messages if hasattr(ex, "messages") else ex["messages"]
        norm = [
            m if isinstance(m, dict) else {"role": m.role, "content": m.content}
            for m in msgs
        ]
        rows.append({"messages": norm})
    return rows


def _render_chat(tok, messages: list[dict]) -> str:
    """메시지를 모델 고유 chat template로 렌더링(완성 대화 = user+assistant 전체).

    hyperclovax(SEED-Think) 등 추론 모델은 비추론 SFT면 skip_reasoning=True로 렌더.
    해당 kwarg 미지원 모델은 일반 템플릿으로, 템플릿 자체가 없으면 단순 포맷으로 폴백한다.
    """
    for kw in ({"skip_reasoning": True}, {}):
        try:
            return tok.apply_chat_template(messages, tokenize=False, **kw)
        except TypeError:
            continue  # skip_reasoning 미지원 토크나이저 → 일반 템플릿 재시도
        except Exception:
            break  # chat_template 부재 등 → 단순 포맷 폴백
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _log_trainable(model) -> None:
    """LoRA 학습가능 파라미터 수/비율을 구조화 라인으로 출력(executor가 stdout에서 파싱)."""
    try:
        nt, tot = model.get_nb_trainable_parameters()
        print({"trainable_params": int(nt), "total_params": int(tot),
               "trainable_pct": round(100.0 * nt / max(tot, 1), 4)}, flush=True)
    except Exception:  # noqa: BLE001
        pass


def _log_adapter_norm(model) -> None:
    """학습된 LoRA 업데이트 ΔW=scale·B·A의 Frobenius norm 합 = '가중치가 얼마나 변했나' 지표."""
    try:
        total = 0.0
        for mod in model.modules():
            A, B = getattr(mod, "lora_A", None), getattr(mod, "lora_B", None)
            if A is None or B is None or not hasattr(A, "keys"):
                continue
            scaling = getattr(mod, "scaling", {})
            for name in A.keys():
                a, b = A[name].weight, B[name].weight
                scale = float(scaling.get(name, 1.0)) if hasattr(scaling, "get") else 1.0
                total += (float((b @ a).detach().float().norm().item()) * scale) ** 2
        print({"adapter_delta_norm": round(total ** 0.5, 4)}, flush=True)
    except Exception:  # noqa: BLE001
        pass


def run_sft_peft(cfg: PeftSFTConfig, examples, mlflow_callback=None) -> str:
    """trl+peft로 bf16 LoRA SFT를 실행하고 어댑터를 cfg.output_dir에 저장, 경로 반환.

    examples: list[SFTExample] 또는 [{"messages":[{role,content},...]}, ...].
    trl 버전 차이(processing_class↔tokenizer, max_seq_length↔max_length)를 런타임에 흡수한다.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl/peft/datasets", "training") from exc

    model, tok = _load_base_for_peft(cfg)
    model.config.use_cache = False

    # 모델 고유 chat template로 미리 렌더링해 text로 학습 — SEED-Think 등 ChatML/추론 모델 정합.
    # trl 기본 conversational 처리에 맡기면 특수 템플릿과 어긋나 완성 토큰 손실이 폭주(loss>ln|V|)한다.
    msg_rows = _to_message_rows(examples)
    if not msg_rows:
        raise ValueError("학습 예제가 비어 있습니다 (messages 0건)")
    dataset = Dataset.from_list([{"text": _render_chat(tok, r["messages"])} for r in msg_rows])

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )

    # SFTConfig: 학습 길이 인자명이 버전에 따라 max_seq_length↔max_length.
    sft_fields = getattr(SFTConfig, "__dataclass_fields__", {})
    len_kw = "max_seq_length" if "max_seq_length" in sft_fields else "max_length"
    # trl 1.8+ 기본 loss_type(chunked_nll)의 forward 패치는 device_map=auto의
    # partial forward와 충돌(멀티GPU 분산 학습 즉사) — TRAIN_SFT_LOSS_TYPE=nll로
    # 청크 경로를 우회(수학 동일, 메모리 최적화만 포기).
    import os as _os
    if _os.environ.get("TRAIN_SFT_LOSS_TYPE") and "loss_type" in sft_fields:
        cfg.extra_sft_args = {**cfg.extra_sft_args,
                              "loss_type": _os.environ["TRAIN_SFT_LOSS_TYPE"]}
    # TRAIN_GRAD_CKPT=1: 대형 베이스(32B 등)를 단일 GPU에 싣기 위한 활성값
    # 재계산 — trl 1.8 chunked-CE 패치가 device_map=auto의 partial forward와
    # 충돌해(멀티GPU 분산 불가) 단일 GPU + 체크포인팅 경로를 쓴다.
    import os as _os
    if _os.environ.get("TRAIN_GRAD_CKPT") == "1":
        cfg.extra_sft_args = {**cfg.extra_sft_args,
                              "gradient_checkpointing": True}
    args = SFTConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        bf16=True,
        report_to=[],
        **{len_kw: cfg.max_seq_length},
        **cfg.extra_sft_args,
    )

    # SFTTrainer: 토크나이저 인자명이 버전에 따라 processing_class↔tokenizer.
    tparams = inspect.signature(SFTTrainer.__init__).parameters
    tok_kw = "processing_class" if "processing_class" in tparams else "tokenizer"
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=dataset, peft_config=lora, **{tok_kw: tok},
    )
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)

    _log_trainable(trainer.model)
    trainer.train()
    _log_adapter_norm(trainer.model)
    trainer.save_model(cfg.output_dir)  # LoRA 어댑터(adapter_model.safetensors + config)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def run_dpo_peft(cfg: PeftSFTConfig, pref_rows, mlflow_callback=None) -> str:
    """unsloth 없는 bf16 DPO(선호 정렬) — trl DPOTrainer + peft LoRA/DoRA. 어댑터 경로 반환.

    pref_rows: PreferenceExample 또는 {"prompt","chosen","rejected"} dict 리스트.
    SFT와 동일한 trl+peft 경로. DoRA는 cfg.use_dora로.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl/peft/datasets", "training") from exc

    rows = []
    for r in pref_rows:
        d = r if isinstance(r, dict) else {
            "prompt": r.prompt, "chosen": r.chosen, "rejected": r.rejected,
        }
        if d.get("prompt") and d.get("chosen") and d.get("rejected"):
            rows.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    if not rows:
        raise ValueError("선호 학습 예제가 비어 있습니다 (prompt/chosen/rejected)")
    dataset = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=False,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )
    args = DPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        beta=cfg.dpo_beta,
        bf16=True,
        report_to=[],
    )
    tparams = inspect.signature(DPOTrainer.__init__).parameters
    tok_kw = "processing_class" if "processing_class" in tparams else "tokenizer"
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=dataset, peft_config=lora, **{tok_kw: tok},
    )
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    _log_trainable(trainer.model)
    trainer.train()
    _log_adapter_norm(trainer.model)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def grpo_rows(prompt_rows) -> list[dict]:
    """GRPO 입력 정규화 — str | dict 혼용 허용, prompt 없는 행은 제외.

    dict 행의 prompt 외 컬럼(gold_label, evidence_quote, required_conditions 등)은
    그대로 보존한다. TRL GRPOTrainer는 데이터셋의 추가 컬럼을 reward function의
    키워드 인자(list, completions와 정렬)로 전달하므로, 검증형 보상(정답 라벨 대조
    등)은 이 컬럼들로 계산한다. prompt-only 데이터는 종전과 동일하게 동작한다.
    """
    rows: list[dict] = []
    for r in prompt_rows:
        if isinstance(r, str):
            if r:
                rows.append({"prompt": r})
        elif r.get("prompt"):
            rows.append(dict(r))
    # datasets.Dataset.from_list는 앞 행에 없는 키를 스키마에서 탈락시킨다 —
    # 일부 행에만 있는 메타 컬럼(condition_results 등)이 통째로 사라지므로
    # 전 행을 키 합집합으로 정규화(누락 키는 None)해 스키마를 고정한다.
    all_keys = {k for r in rows for k in r}
    for r in rows:
        for k in all_keys:
            r.setdefault(k, None)
    return rows


def run_grpo_peft(cfg: PeftSFTConfig, prompt_rows, reward_funcs=None, mlflow_callback=None) -> str:
    """GRPO(그룹 상대 정책 최적화) — bf16 LoRA/DoRA, 보상함수 기반 강화학습. 어댑터 경로 반환.

    prompt_rows: {"prompt": ...} 리스트(정답 불필요, 보상함수가 점수화). prompt 외
                 컬럼은 보존되어 reward function의 키워드 인자로 전달된다(grpo_rows).
    reward_funcs: list[callable(prompts, completions, **kw)->list[float]]. 미지정 시
                  길이 적정성 휴리스틱 기본 보상(데모용). 실사용은 도메인 보상함수 주입.
    추론/판정 강화(4E)에 사용. 가치모델 없이 단일 GPU에서 동작.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl(GRPO)/peft", "training") from exc

    rows = grpo_rows(prompt_rows)
    if not rows:
        raise ValueError("GRPO 프롬프트가 비어 있습니다")
    dataset = Dataset.from_list(rows)

    def _default_reward(completions, **_):
        # 데모 보상: 너무 짧지/길지 않은 한국어 답변 선호(실사용은 도메인 보상으로 교체)
        out = []
        for c in completions:
            text = c if isinstance(c, str) else (c[0]["content"] if c else "")
            n = len(text.strip())
            out.append(1.0 if 20 <= n <= 400 else max(0.0, 1.0 - abs(n - 200) / 400))
        return out

    funcs = reward_funcs or [_default_reward]

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=False,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    # GRPOTrainer 내부 generate는 tokenizer를 넘기지 않으므로, tokenizer가 필요한
    # stop_strings(generation_config)를 제거한다(EOS 기반 종료로 대체).
    if getattr(model.generation_config, "stop_strings", None):
        model.generation_config.stop_strings = None
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )
    # GRPO: generation batch(=per_device_train_batch_size)는 num_generations의 배수여야 한다.
    num_gen = cfg.extra_sft_args.get("num_generations", 4)
    bs = max(cfg.per_device_train_batch_size, num_gen)
    bs = (bs // num_gen) * num_gen or num_gen
    args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        num_generations=num_gen,
        bf16=True,
        report_to=[],
        # 추가 컬럼(gold_label 등)을 reward function kwargs로 전달하려면 필수.
        # GRPOConfig 기본값이 False지만 TrainingArguments 상속 변경에 대비해 명시.
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=dataset,
        reward_funcs=funcs, peft_config=lora,
    )
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    _log_trainable(trainer.model)
    trainer.train()
    _log_adapter_norm(trainer.model)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir



def run_dpo(cfg: SFTJobConfig, pref_dataset, mlflow_callback=None):
    """선호 정렬(DPO) — unsloth 경로. 기본은 run_dpo_peft 사용."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl", "training") from exc

    model, tok = _build_model(cfg)
    args = DPOConfig(output_dir=cfg.output_dir, learning_rate=cfg.learning_rate)
    trainer = DPOTrainer(model=model, args=args, train_dataset=pref_dataset, tokenizer=tok)
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    return trainer.train()
