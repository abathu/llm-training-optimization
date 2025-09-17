# # import os
# # import sys
# # from dataclasses import dataclass, field
# # from typing import Optional

# # from transformers import HfArgumentParser, set_seed
# # from trl import SFTTrainer, SFTConfig
# # from utils import create_and_prepare_model, create_datasets
# # from transformers import TrainingArguments

# # # -----------------------
# # # Argument dataclasses
# # # -----------------------
# # @dataclass
# # class ModelArguments:
# #     """Which model/config/tokenizer we fine-tune from."""
# #     model_name_or_path: str = field(
# #         metadata={"help": "HF hub id or local path to the base model"}
# #     )
# #     chat_template_format: Optional[str] = field(
# #         default="none",  # chatml|zephyr|none
# #         metadata={"help": "chatml|zephyr|none. Use `none` if dataset text is already templated."},
# #     )
# #     lora_alpha: Optional[int] = field(default=16)
# #     lora_dropout: Optional[float] = field(default=0.1)
# #     lora_r: Optional[int] = field(default=64)
# #     lora_target_modules: Optional[str] = field(
# #         default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
# #         metadata={"help": "Comma-separated module names for LoRA"},
# #     )

# #     # Quantization / compute
# #     use_nested_quant: Optional[bool] = field(default=False)
# #     bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
# #     bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8")  # storage should be uint8
# #     bnb_4bit_quant_type: Optional[str] = field(default="nf4")

# #     # Features
# #     use_flash_attn: Optional[bool] = field(default=False)
# #     use_peft_lora: Optional[bool] = field(default=False)
# #     use_8bit_quantization: Optional[bool] = field(default=False)
# #     use_4bit_quantization: Optional[bool] = field(default=False)
# #     use_reentrant: Optional[bool] = field(default=False)   # for gradient checkpointing
# #     use_unsloth: Optional[bool] = field(default=False)     # handled inside utils.create_and_prepare_model


# # @dataclass
# # class DataTrainingArguments:
# #     """ONLY dataset identity/splits here to avoid CLI conflicts with SFTConfig."""
# #     dataset_name: Optional[str] = field(
# #         default="timdettmers/openassistant-guanaco",
# #         metadata={"help": "HF dataset repo id or local path (load_from_disk)"},
# #     )
# #     splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated splits"})


# # # -----------------------
# # # Main
# # # -----------------------
# # def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: SFTConfig):
# #     # Seed
# #     set_seed(training_args.seed)

# #     # Build model + tokenizer (+ optional PEFT config)
# #     model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

# #     # Gradient checkpointing plumbing
# #     model.config.use_cache = not training_args.gradient_checkpointing
# #     training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
# #     if training_args.gradient_checkpointing:
# #         training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

# #     # Datasets (your utils handles hub/local & templating switch)
# #     train_dataset, eval_dataset = create_datasets(
# #         tokenizer=tokenizer,
# #         data_args=data_args,
# #         training_args=training_args,
# #         apply_chat_template=(model_args.chat_template_format != "none"),
# #     )

# #     # Build trainer – TRL 0.21: use processing_class (not tokenizer=). All dataset knobs come from SFTConfig.
# #     trainer = SFTTrainer(
# #         model=model,
# #         processing_class=tokenizer,
# #         args=training_args,          # <- includes packing, max_seq_length, dataset_text_field, dataset_kwargs, etc.
# #         train_dataset=train_dataset,
# #         eval_dataset=eval_dataset,
# #         peft_config=peft_config,
# #     )

# #     # Logging model & trainable params (LoRA)
# #     trainer.accelerator.print(f"{trainer.model}")
# #     if hasattr(trainer.model, "print_trainable_parameters"):
# #         trainer.model.print_trainable_parameters()

# #     # Resume support
# #     checkpoint = getattr(training_args, "resume_from_checkpoint", None)
# #     trainer.train(resume_from_checkpoint=checkpoint)

# #     # Save final (handle FSDP case)
# #     if trainer.is_fsdp_enabled:
# #         trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
# #     trainer.save_model()


# # # if __name__ == "__main__":
# # #     # Parse: ModelArguments + DataTrainingArguments + SFTConfig (NOT TrainingArguments)
# # #     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))

# # #     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
# # #         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
# # #     else:
# # #         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# # #     main(model_args, data_args, training_args)

# # if __name__ == "__main__":
# #     # Parse: ModelArguments + DataTrainingArguments + TrainingArguments (改用TrainingArguments)
# #     from transformers import TrainingArguments  # 添加这行import
    
# #     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))  # 改这行

# #     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
# #         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
# #     else:
# #         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# #     main(model_args, data_args, training_args)



# import os
# import sys
# from dataclasses import dataclass, field
# from typing import Optional

# from transformers import HfArgumentParser, set_seed, TrainingArguments
# from trl import SFTTrainer, SFTConfig
# from utils import create_and_prepare_model, create_datasets


# # -----------------------
# # Argument dataclasses
# # -----------------------
# @dataclass
# class ModelArguments:
#     """Which model/config/tokenizer we fine-tune from."""
#     model_name_or_path: str = field(
#         metadata={"help": "HF hub id or local path to the base model"}
#     )
#     chat_template_format: Optional[str] = field(
#         default="none",  # chatml|zephyr|none
#         metadata={"help": "chatml|zephyr|none. Use `none` if dataset text is already templated."},
#     )
#     lora_alpha: Optional[int] = field(default=16)
#     lora_dropout: Optional[float] = field(default=0.1)
#     lora_r: Optional[int] = field(default=64)
#     lora_target_modules: Optional[str] = field(
#         default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
#         metadata={"help": "Comma-separated module names for LoRA"},
#     )

#     # Quantization / compute
#     use_nested_quant: Optional[bool] = field(default=False)
#     bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
#     bnb_4bit_quant_storage_dtype: Optional[str] = field(default="uint8")  # storage should be uint8
#     bnb_4bit_quant_type: Optional[str] = field(default="nf4")

#     # Features
#     use_flash_attn: Optional[bool] = field(default=False)
#     use_peft_lora: Optional[bool] = field(default=False)
#     use_8bit_quantization: Optional[bool] = field(default=False)
#     use_4bit_quantization: Optional[bool] = field(default=False)
#     use_reentrant: Optional[bool] = field(default=False)   # for gradient checkpointing
#     use_unsloth: Optional[bool] = field(default=False)     # handled inside utils.create_and_prepare_model


# @dataclass
# class DataTrainingArguments:
#     """Dataset and SFT-specific arguments."""
#     dataset_name: Optional[str] = field(
#         default="timdettmers/openassistant-guanaco",
#         metadata={"help": "HF dataset repo id or local path (load_from_disk)"},
#     )
#     splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated splits"})
    
#     # SFT特有参数
#     dataset_text_field: Optional[str] = field(default="content", metadata={"help": "Dataset text field name"})
#     max_seq_length: Optional[int] = field(default=1024, metadata={"help": "Maximum sequence length"})
#     packing: Optional[bool] = field(default=False, metadata={"help": "Enable packing"})


# # -----------------------
# # Main
# # -----------------------
# def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
#     # Seed
#     set_seed(training_args.seed)

#     # Build model + tokenizer (+ optional PEFT config)
#     model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

#     # Gradient checkpointing plumbing
#     model.config.use_cache = not training_args.gradient_checkpointing
#     training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
#     if training_args.gradient_checkpointing:
#         training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

#     # 创建临时的training_args对象，添加SFT特有的属性
#     # 这是一个hack，因为TrainingArguments没有这些属性，但utils.py需要它们
#     if not hasattr(training_args, 'dataset_text_field'):
#         training_args.dataset_text_field = data_args.dataset_text_field
#     if not hasattr(training_args, 'max_seq_length'):
#         training_args.max_seq_length = data_args.max_seq_length

#     # Datasets (your utils handles hub/local & templating switch)
#     train_dataset, eval_dataset = create_datasets(
#         tokenizer=tokenizer,
#         data_args=data_args,
#         training_args=training_args,
#         apply_chat_template=(model_args.chat_template_format != "none"),
#     )

#     # Build trainer - SFTTrainer会自动将TrainingArguments转换为SFTConfig
#     trainer = SFTTrainer(
#         model=model,
#         processing_class=tokenizer,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         peft_config=peft_config,
#     )

#     # Logging model & trainable params (LoRA)
#     trainer.accelerator.print(f"{trainer.model}")
#     if hasattr(trainer.model, "print_trainable_parameters"):
#         trainer.model.print_trainable_parameters()

#     # Resume support
#     checkpoint = getattr(training_args, "resume_from_checkpoint", None)
#     trainer.train(resume_from_checkpoint=checkpoint)

#     # Save final (handle FSDP case)
#     if trainer.is_fsdp_enabled:
#         trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
#     trainer.save_model()


# if __name__ == "__main__":
#     # Parse: ModelArguments + DataTrainingArguments + TrainingArguments
#     parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

#     if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
#         model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
#     else:
#         model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     main(model_args, data_args, training_args)



import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, set_seed, TrainingArguments
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
    """Dataset and SFT-specific arguments."""
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "HF dataset repo id or local path (load_from_disk)"},
    )
    splits: Optional[str] = field(default="train,test", metadata={"help": "Comma-separated splits"})
    
    # SFT特有参数 - 现在只用于传递，实际不传给SFTTrainer
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "Dataset text field name - 现在固定为text"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "Maximum sequence length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "Enable packing"})


# -----------------------
# Main
# -----------------------
def main(model_args: ModelArguments, data_args: DataTrainingArguments, training_args: TrainingArguments):
    # Seed
    set_seed(training_args.seed)

    # Build model + tokenizer (+ optional PEFT config)
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    # Gradient checkpointing plumbing
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # # 设置dataset_text_field为'text' (SFTTrainer的默认期望)
    # if not hasattr(training_args, 'dataset_text_field'):
    #     training_args.dataset_text_field = "fact"  # 固定为text
    # if not hasattr(training_args, 'max_seq_length'):
    #     training_args.max_seq_length = data_args.max_seq_length

    # Datasets (utils.py现在输出'text'字段)
    train_dataset, eval_dataset = create_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
        apply_chat_template=(model_args.chat_template_format != "none"),
    )

    ### random pick data set 

    ratio = 0.10
    train_dataset = train_dataset.shuffle(seed=training_args.seed).select(
        range(int(len(train_dataset) * ratio))
    )

    print(">>> train columns:", train_dataset.column_names[:10])
    print(">>> eval  columns:", eval_dataset.column_names[:10])
    print(">>> sample keys:", list(train_dataset[0].keys()))
    print(">>> first text:", train_dataset[0].get("text", None))

    print("train columns:", train_dataset.column_names)
    print("eval  columns:", eval_dataset.column_names)
    # assert "fact" in train_dataset.column_names, "train_dataset 没有 'fact' 列"
    # assert "fact" in eval_dataset.column_names,  "eval_dataset 没有 'fact' 列"

    # if "fact" in train_dataset.column_names:
    #     train_dataset = train_dataset.rename_column("fact", "text")
    #     eval_dataset  = eval_dataset.rename_column("fact", "text")

    # 创建SFTConfig，明确设置所有参数
    sft_config = SFTConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        bf16=training_args.bf16,
        gradient_checkpointing=training_args.gradient_checkpointing,
        optim=training_args.optim,
        seed=training_args.seed,
        dataloader_drop_last=getattr(training_args, 'dataloader_drop_last', False),
        eval_strategy=getattr(training_args, 'eval_strategy', "no"),
        save_strategy=getattr(training_args, 'save_strategy', "steps"),
        # SFT特有参数 - 使用默认值，因为我们已经在utils.py中处理了格式转换
        remove_unused_columns=False,   # 强烈建议保留，避免被 Trainer 丢列
        dataset_text_field="text",
        # max_seq_length=data_args.max_seq_length,
        packing=data_args.packing,
    )

    # Build trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,  # 使用明确配置的SFTConfig
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
    # Parse: ModelArguments + DataTrainingArguments + TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)