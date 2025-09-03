import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, set_seed
from trl import SFTTrainer, SFTConfig
from utils import create_and_prepare_model, create_datasets


# -----------------------
# Argument dataclasses
# -----------------------
@dataclass
class ModelArguments:
    """Which model/config/tokenizer we fine-tune from."""
    model_name_or_path: str = field(
        metadata={"help": "HF hub id or local path to the base model"}
    )
    chat_template_format: Optional[str] = field(
        default="none",  # chatml|zephyr|none
        metadata={"help": "chatml|zephyr|none. Use `none` if dataset text is already templated."},
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "Comma-separated module names for LoRA"},
    )

    # Quantization / compute
    use_nested_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8")  # storage should be uint8
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")

    # Features
    use_flash_attn: Optional[bool] = field(default=False)
    use_peft_lora: Optional[bool] = field(default=False)
    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=False)
    use_reentrant: Optional[bool] = field(default=False)   # for gradient checkpointing
    use_unsloth: Optional[bool] = field(default=False)     # handled inside utils.create_and_prepare_model


@dataclass
class DataTrainingArguments:
    """ONLY dataset identity/splits here to avoid CLI conflicts with SFTConfig."""
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "HF dataset repo id or local path (load_from_disk)"},
    )
    splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated splits"})


# -----------------------
# Main
# -----------------------
def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: SFTConfig):
    # Seed
    set_seed(training_args.seed)

    # Build model + tokenizer (+ optional PEFT config)
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    # Gradient checkpointing plumbing
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # Datasets (your utils handles hub/local & templating switch)
    train_dataset, eval_dataset = create_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        apply_chat_template=(model_args.chat_template_format != "none"),
    )

    # Build trainer – TRL 0.21: use processing_class (not tokenizer=). All dataset knobs come from SFTConfig.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,          # <- includes packing, max_seq_length, dataset_text_field, dataset_kwargs, etc.
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Logging model & trainable params (LoRA)
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # Resume support
    checkpoint = getattr(training_args, "resume_from_checkpoint", None)
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final (handle FSDP case)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    # Parse: ModelArguments + DataTrainingArguments + SFTConfig (NOT TrainingArguments)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)
