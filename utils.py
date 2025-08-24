import os
from enum import Enum
import json
from typing import Any, Dict, List

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig


# -----------------------
# Chat templates & tokens
# -----------------------
DEFAULT_CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{{'' + message['role'] + '\n' + message['content'] + '' + '\n'}}"
    "{% if loop.last and add_generation_prompt %}{{'assistant\n' }}{% endif %}"
    "{% endfor %}"
)

DEFAULT_ZEPHYR_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n{{ '\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n{{ '\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n{{ '' }}\n{% endif %}\n"
    "{% endfor %}"
)


class ZephyrSpecialTokens(str, Enum):
    user = ""
    assistant = ""
    system = ""
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    eos_token = ""
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


# -----------------------
# Dataset creation
# -----------------------
def _apply_template_from_messages(tokenizer, msgs: Any) -> str:
    """Render a list of {role, content} with the tokenizer's chat template."""
    if not isinstance(msgs, (list, tuple)):
        # some datasets store JSON strings
        msgs = json.loads(msgs)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )


def _apply_template_from_qa(tokenizer, q: str, a: str) -> str:
    conversation = [
        {"role": "user", "content": (q or "").strip()},
        {"role": "assistant", "content": (a or "").strip()},
    ]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )


def create_datasets(tokenizer, data_args, training_args, apply_chat_template: bool = False):
    """
    Builds train/validation datasets and ensures a single 'content' column for SFTTrainer.

    Priority (when apply_chat_template=True):
      1) 'messages' -> apply chat template
      2) 'dialogue' + 'soap' -> 2-turn chat template
      3) 'question' + 'answer' -> 2-turn chat template
    Otherwise (no templating): pass through training_args.dataset_text_field.
    """

    # Load splits (hub or local)
    raw = DatasetDict()
    for split in data_args.splits.split(","):
        split = split.strip()
        try:
            ds = load_dataset(data_args.dataset_name, split=split)
        except DatasetGenerationError:
            ds = load_from_disk(os.path.join(data_args.dataset_name, split))
        raw[split] = ds

    # Collect all columns across splits for safe removal later
    all_cols: List[str] = []
    for _, ds in raw.items():
        for c in ds.column_names:
            if c not in all_cols:
                all_cols.append(c)

    def preprocess(samples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        out_batch: List[str] = []

        if apply_chat_template:
            # 1) messages-style (preferred if available)
            if "messages" in samples:
                for msgs in samples["messages"]:
                    out_batch.append(_apply_template_from_messages(tokenizer, msgs))
                return {"content": out_batch}

            # 2) dialogue + soap
            if "dialogue" in samples and "soap" in samples:
                for q, a in zip(samples["dialogue"], samples["soap"]):
                    out_batch.append(_apply_template_from_qa(tokenizer, q, a))
                return {"content": out_batch}

            # 3) legacy question/answer
            if "question" in samples and "answer" in samples:
                for q, a in zip(samples["question"], samples["answer"]):
                    out_batch.append(_apply_template_from_qa(tokenizer, q, a))
                return {"content": out_batch}

            # Fall through to plain field if templating was requested but structure not found

        # No templating (or no suitable fields): pass through a chosen text field
        field = getattr(training_args, "dataset_text_field", None) or "text"
        if field in samples:
            return {"content": samples[field]}

        # Nothing matched: raise a clear error
        raise KeyError(
            f"Could not find usable columns. Available: {list(samples.keys())}. "
            f"Tried: messages | (dialogue+soap) | (question+answer) | '{field}'."
        )

    # Map to a unified 'content' column; remove all other columns to avoid Trainer confusion
    raw = raw.map(
        preprocess,
        batched=True,
        remove_columns=[c for c in all_cols if c != "content"],
    )

    train_data = raw["train"]
    valid_data = raw.get("test", None)
    if valid_data is None:
        split_ds = train_data.train_test_split(test_size=0.1, seed=training_args.seed)
        train_data, valid_data = split_ds["train"], split_ds["test"]

    return train_data, valid_data


# -----------------------
# Model/tokenizer creation
# -----------------------
def create_and_prepare_model(args, data_args, training_args):
    """
    Builds model + tokenizer + (optional) PEFT LoRA config.
    - Supports 4-bit / 8-bit (bitsandbytes), Flash-Attn toggle, Unsloth path.
    - Adds special tokens & chat template when chat_template_format != "none".
    """
    if args.use_unsloth:
        from unsloth import FastLanguageModel  # lazy import to avoid extra dep for non-users

    bnb_config = None
    quant_storage_dtype = None

    # Unsloth is not supported in multi-proc in this script
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    # Quantization config
    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,   # 'nf4' or 'fp4'
            bnb_4bit_compute_dtype=compute_dtype,           # usually torch.bfloat16
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,     # torch.uint8
        )

        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16; consider --bf16 True for faster/safer training.")
                print("=" * 80)
    elif args.use_8bit_quantization:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model (Unsloth vs standard HF)
    if args.use_unsloth:
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=getattr(training_args, "max_seq_length", 2048),
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        # If quant_storage_dtype is a float (rare), honor it; else default to fp32 and rely on bf16 autocast
        torch_dtype = (
            quant_storage_dtype if (quant_storage_dtype is not None and hasattr(quant_storage_dtype, "is_floating_point") and quant_storage_dtype.is_floating_point)
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    # LoRA config
    peft_config = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                args.lora_target_modules
                if args.lora_target_modules == "all-linear"
                else args.lora_target_modules.split(",")
            ),
        )

    # Tokenizer + special tokens + template (if requested)
    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # resize embeddings for added special tokens (pad to multiple of 8 helps Flash-Attn / tensor cores)
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        # ensure a pad token exists for batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Unsloth LoRA wrapping (if selected)
    if args.use_unsloth:
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=(
                args.lora_target_modules
                if args.lora_target_modules == "all-linear"
                else args.lora_target_modules.split(",")
            ),
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=getattr(training_args, "max_seq_length", 2048),
        )

    return model, peft_config, tokenizer