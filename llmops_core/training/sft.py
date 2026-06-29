"""SFT/정렬 학습기 — unsloth + trl + peft 모듈 조합 (Axolotl/LLaMA-Factory 통째 미채용).

동일 기능(고속 LoRA, 분산, DPO/GRPO)을 라이브러리 모듈로 직접 조립한다.
출력은 표준 HF 체크포인트 → vLLM 서빙·MLflow 등록과 일관. import는 lazy(GPU 노드).

경로 두 가지:
- `run_sft`/`run_dpo`: unsloth(고속·4bit) 경로. CUDA sm_80~90 등 unsloth 프리빌트 가용 환경.
- `run_sft_peft`: unsloth 없이 trl+peft bf16 LoRA. **GB10(Blackwell sm_121/aarch64)** 처럼
  unsloth/bitsandbytes 프리빌트가 없는 환경의 검증 경로. SEED-0.5B는 4bit 없이 학습 가능.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from llmops_core.common.errors import OptionalDependencyError


@dataclass
class SFTJobConfig:
    base_model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B"
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


# ── unsloth 없는 bf16 LoRA 경로 (GB10/Blackwell 검증용) ──
@dataclass
class PeftSFTConfig:
    """trl+peft bf16 LoRA SFT 설정 — bitsandbytes/unsloth 미사용.

    SEED-0.5B 기준 단일 GB10(121GB 통합메모리)에서 4bit 없이 학습 가능.
    """

    base_model: str = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    output_dir: str = "outputs/adapter"
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_dora: bool = False  # PET 변형: DoRA(weight-decomposed LoRA). peft use_dora=True
    load_in_4bit: bool = False  # QLoRA. bitsandbytes 필요(GB10 Blackwell 미지원 → 가드)
    hf_token: str | None = None  # gated 모델(예: SEED 1.5B) 접근 토큰. 미지정 시 env HF_TOKEN
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    max_steps: int = -1  # >0이면 epoch 무시 — 빠른 루프 검증용
    extra_sft_args: dict = field(default_factory=dict)


def _load_base_for_peft(cfg: "PeftSFTConfig"):
    """bf16(기본) 또는 QLoRA(4bit) 베이스 로드 + gated 모델 토큰 처리. (model, tok) 반환.

    4bit는 bitsandbytes를 요구한다. GB10(Blackwell/aarch64)에는 프리빌트가 없어
    load_in_4bit=True면 명확한 안내와 함께 실패한다(자원 충족 환경에서 동작).
    """
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = cfg.hf_token or os.environ.get("HF_TOKEN")
    auth = {"token": token} if token else {}

    tok = AutoTokenizer.from_pretrained(cfg.base_model, **auth)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=torch.bfloat16,
                  device_map="cuda" if torch.cuda.is_available() else None, **auth)
    if cfg.load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise OptionalDependencyError(
                "bitsandbytes(QLoRA 4bit)", "training"
            ) from exc  # GB10 Blackwell에는 프리빌트 없음 → CUDA sm_80~90 또는 소스빌드 필요
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

    rows = _to_message_rows(examples)
    if not rows:
        raise ValueError("학습 예제가 비어 있습니다 (messages 0건)")
    dataset = Dataset.from_list(rows)

    model, tok = _load_base_for_peft(cfg)
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )

    # SFTConfig: 학습 길이 인자명이 버전에 따라 max_seq_length↔max_length.
    sft_fields = getattr(SFTConfig, "__dataclass_fields__", {})
    len_kw = "max_seq_length" if "max_seq_length" in sft_fields else "max_length"
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

    trainer.train()
    trainer.save_model(cfg.output_dir)  # LoRA 어댑터(adapter_model.safetensors + config)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def run_dpo_peft(cfg: PeftSFTConfig, pref_rows, mlflow_callback=None) -> str:
    """unsloth 없는 bf16 DPO(선호 정렬) — trl DPOTrainer + peft LoRA/DoRA. 어댑터 경로 반환.

    pref_rows: PreferenceExample 또는 {"prompt","chosen","rejected"} dict 리스트.
    SFT와 동일한 GB10 bf16 경로(4bit/bitsandbytes 미사용). DoRA는 cfg.use_dora로.
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

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16,
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
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def run_grpo_peft(cfg: PeftSFTConfig, prompt_rows, reward_funcs=None, mlflow_callback=None) -> str:
    """GRPO(그룹 상대 정책 최적화) — bf16 LoRA/DoRA, 보상함수 기반 강화학습. 어댑터 경로 반환.

    prompt_rows: {"prompt": ...} 리스트(정답 불필요, 보상함수가 점수화).
    reward_funcs: list[callable(prompts, completions, **kw)->list[float]]. 미지정 시
                  길이 적정성 휴리스틱 기본 보상(데모용). 실사용은 도메인 보상함수 주입.
    추론/판정 강화(4E)에 사용. PPO보다 가볍고 단일 GB10에서 동작.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl(GRPO)/peft", "training") from exc

    rows = [{"prompt": (r if isinstance(r, str) else r["prompt"])} for r in prompt_rows
            if (r if isinstance(r, str) else r.get("prompt"))]
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

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )
    args = GRPOConfig(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        max_steps=cfg.max_steps,
        num_generations=cfg.extra_sft_args.get("num_generations", 4),
        bf16=True,
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model, args=args, train_dataset=dataset,
        reward_funcs=funcs, peft_config=lora,
    )
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def run_ppo_peft(cfg: PeftSFTConfig, prompt_rows, reward_model_path: str | None = None,
                 mlflow_callback=None) -> str:
    """PPO(근위 정책 최적화) — bf16 LoRA. reward·value 모델 + 정책/참조 모델. 어댑터 경로 반환.

    prompt_rows: {"prompt": ...} 리스트(토큰화하여 input_ids로 사용).
    reward_model_path: 학습된 보상 모델 경로(SequenceClassification, num_labels=1).
                       미지정 시 베이스로 초기화(랜덤 헤드) — 메커니즘 검증용. 실사용은 보상모델 주입.
    GRPO보다 무겁다(모델 4개). 추론/판정 강화에 사용.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        from trl import PPOConfig, PPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("trl(PPO)/peft/datasets", "training") from exc

    rows = [{"prompt": (r if isinstance(r, str) else r["prompt"])} for r in prompt_rows
            if (r if isinstance(r, str) else r.get("prompt"))]
    if not rows:
        raise ValueError("PPO 프롬프트가 비어 있습니다")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _tok(ex):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}], add_generation_prompt=True)
        return {"input_ids": ids, "lengths": len(ids)}

    dataset = Dataset.from_list(rows).map(_tok, remove_columns=["prompt"])

    dtype = torch.bfloat16
    rm_src = reward_model_path or cfg.base_model
    policy = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        rm_src, num_labels=1, torch_dtype=dtype)
    value_model = AutoModelForSequenceClassification.from_pretrained(
        rm_src, num_labels=1, torch_dtype=dtype)
    for m in (policy, reward_model, value_model):
        m.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules), bias="none", task_type="CAUSAL_LM",
        use_dora=cfg.use_dora,
    )
    bs = max(cfg.per_device_train_batch_size, 1)
    args = PPOConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_mini_batches=1,
        total_episodes=cfg.extra_sft_args.get("total_episodes", max(bs * 4, 8)),
        response_length=cfg.extra_sft_args.get("response_length", 48),
        learning_rate=cfg.learning_rate,
        bf16=True,
        report_to=[],
        stop_token="eos",  # 생성 응답을 EOS에서 절단(보상 인덱싱 안정화)
        missing_eos_penalty=1.0,
    )
    trainer = PPOTrainer(
        args=args, processing_class=tok, model=policy, ref_model=None,
        reward_model=reward_model, value_model=value_model,
        train_dataset=dataset, peft_config=lora,
    )
    if mlflow_callback is not None:
        trainer.add_callback(mlflow_callback)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    return cfg.output_dir


def run_dpo(cfg: SFTJobConfig, pref_dataset, mlflow_callback=None):
    """선호 정렬(DPO) — unsloth 경로. GB10에서는 run_dpo_peft 사용."""
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
