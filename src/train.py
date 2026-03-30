"""
Python 微调脚本 — 使用 transformers + peft 直接进行 LoRA SFT

说明：
  本脚本与 LlamaFactory YAML 配置方案功能等价，参数一一对应。
  支持 DeepSpeed ZeRO-3 多卡分片，解决 8B 模型单卡 OOM 问题。

启动方式：
  # 单卡（不推荐 8B 模型，会 OOM）
  python src/train.py

  # 多卡 + DeepSpeed ZeRO-3（推荐，4×V100 16GB）
  torchrun --nproc_per_node=4 src/train.py

  # 指定 DeepSpeed 配置（默认已在 config.py 中配置）
  torchrun --nproc_per_node=4 src/train.py --deepspeed configs/deepspeed/ds_z3_config.json

  # 后台运行（推荐，断开 SSH 不影响训练）
  bash scripts/train_background.sh --python

对应 YAML: configs/train_lora_sft.yaml

OOM 解决方案：
  DeepSpeed ZeRO-3 将模型参数分片到多张 GPU，每张 GPU 只持有 1/N 的参数，
  从根本上解决 8B 模型在 16GB V100 上无法加载的问题。
  启动时必须使用 torchrun（而非 python），否则 DeepSpeed 不会生效。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import ProjectConfig, DataConfig, load_project_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据加载与预处理
# =============================================================================


def load_jsonl_data(file_path: str) -> list[dict]:
    """加载 JSON 格式的数据文件（alpaca 格式）"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data


def preprocess_data(
    examples: list[dict],
    tokenizer,
    data_config: DataConfig,
) -> Dataset:
    """
    将 alpaca 格式数据转换为模型输入

    数据格式:
      {"instruction": "...", "input": "用户文本", "output": "标签"}

    处理逻辑:
      1. 将 instruction + input 拼接为 prompt
      2. 使用 tokenizer 的 apply_chat_template 构建完整输入
      3. 对 prompt 部分的 labels 设为 -100（不计算 loss）
    """
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for ex in examples:
        instruction = ex["instruction"]
        user_input = ex["input"]
        output = ex["output"]

        # 构建对话格式
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": output},
        ]

        # 使用 tokenizer 的 chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize 完整文本
        full_encoded = tokenizer(
            text,
            max_length=data_config.max_seq_length,
            truncation=True,
            padding=False,
        )

        # 构建 prompt 部分（不含 assistant 回复）用于计算 label mask
        prompt_messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_encoded = tokenizer(
            prompt_text,
            max_length=data_config.max_seq_length,
            truncation=True,
            padding=False,
        )

        # Labels: prompt 部分设为 -100，只对 output 部分计算 loss
        input_ids = full_encoded["input_ids"]
        labels = [-100] * len(prompt_encoded["input_ids"]) + input_ids[
            len(prompt_encoded["input_ids"]) :
        ]

        # 确保长度一致
        labels = labels[: len(input_ids)]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        input_ids_list.append(input_ids)
        attention_mask_list.append(full_encoded["attention_mask"])
        labels_list.append(labels)

    return Dataset.from_dict(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }
    )


# =============================================================================
# 模型构建
# =============================================================================


def build_model_and_tokenizer(cfg: ProjectConfig):
    """
    加载基座模型和 tokenizer，并应用 LoRA 适配器

    对应 YAML 参数:
      model_name_or_path → cfg.model.model_name_or_path
      finetuning_type    → cfg.lora.finetuning_type
      lora_target        → cfg.lora.lora_target (→ peft target_modules)
      lora_rank          → cfg.lora.lora_rank (→ peft r)
      lora_alpha         → cfg.lora.lora_alpha (→ peft lora_alpha)
      lora_dropout       → cfg.lora.lora_dropout (→ peft lora_dropout)
    """
    logger.info(f"加载模型: {cfg.model.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型到 CPU（不指定 device_map）
    # DeepSpeed ZeRO-3 会自动将模型分片到多张 GPU，不能提前指定 device_map
    # 如果不使用 DeepSpeed（单卡/普通 DDP），模型会被 Trainer 自动移到 GPU
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=torch.float16 if cfg.training.fp16 else torch.float32,
        # 注意：不能加 device_map="auto"，否则与 DeepSpeed ZeRO-3 冲突
    )

    # 配置 LoRA
    # YAML: lora_target=all  →  peft: target_modules 根据模型自动检测
    if cfg.lora.lora_target == "all":
        target_modules = "all-linear"
    else:
        target_modules = cfg.lora.lora_target.split(",")

    if cfg.lora.target_modules:
        target_modules = cfg.lora.target_modules

    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.lora_rank,                # YAML: lora_rank
        lora_alpha=cfg.lora.lora_alpha,       # YAML: lora_alpha
        lora_dropout=cfg.lora.lora_dropout,   # YAML: lora_dropout
        target_modules=target_modules,        # YAML: lora_target
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# 训练主流程
# =============================================================================


def train(cfg: ProjectConfig) -> None:
    """
    执行 LoRA SFT 训练

    所有参数与 YAML 配置一一对应，见 config.py 中的对应关系表
    """
    model, tokenizer = build_model_and_tokenizer(cfg)

    # 加载数据
    train_data_raw = load_jsonl_data(cfg.data.train_file)
    val_data_raw = load_jsonl_data(cfg.data.val_file)

    train_dataset = preprocess_data(train_data_raw, tokenizer, cfg.data)
    val_dataset = preprocess_data(val_data_raw, tokenizer, cfg.data)

    logger.info(f"训练集: {len(train_dataset)} 条，验证集: {len(val_dataset)} 条")

    # 构建 TrainingArguments（与 YAML 参数一一对应）
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,                               # YAML: output_dir
        num_train_epochs=cfg.training.num_train_epochs,                   # YAML: num_train_epochs
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,  # YAML: per_device_train_batch_size
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,    # YAML: per_device_eval_batch_size
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,  # YAML: gradient_accumulation_steps
        learning_rate=cfg.training.learning_rate,                         # YAML: learning_rate
        lr_scheduler_type=cfg.training.lr_scheduler_type,                 # YAML: lr_scheduler_type
        warmup_ratio=cfg.training.warmup_ratio,                           # YAML: warmup_ratio
        weight_decay=cfg.training.weight_decay,                           # YAML: weight_decay
        max_grad_norm=cfg.training.max_grad_norm,                         # YAML: max_grad_norm
        fp16=cfg.training.fp16,                                           # YAML: fp16
        optim=cfg.training.optim,                                         # YAML: optim
        seed=cfg.training.seed,                                           # YAML: seed
        logging_steps=cfg.training.logging_steps,                         # YAML: logging_steps
        save_steps=cfg.training.save_steps,                               # YAML: save_steps
        save_total_limit=cfg.training.save_total_limit,                   # YAML: save_total_limit
        eval_strategy=cfg.training.eval_strategy,                         # YAML: eval_strategy
        eval_steps=cfg.training.eval_steps,                               # YAML: eval_steps
        gradient_checkpointing=cfg.training.gradient_checkpointing,       # YAML: gradient_checkpointing
        report_to=cfg.training.report_to,                                 # YAML: report_to
        logging_dir=os.path.join(cfg.training.output_dir, "logs"),
        overwrite_output_dir=True,
        remove_unused_columns=False,
        ddp_timeout=180000000,                                            # YAML: ddp_timeout
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # DeepSpeed ZeRO-3 配置（解决多卡 OOM）
        # 需配合 torchrun --nproc_per_node=N 启动，否则 DeepSpeed 不生效
        # YAML: deepspeed
        deepspeed=cfg.training.deepspeed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    logger.info("开始训练...")
    trainer.train()

    logger.info("保存最终模型...")
    trainer.save_model(cfg.training.output_dir)
    tokenizer.save_pretrained(cfg.training.output_dir)

    logger.info(f"训练完成！模型保存至: {cfg.training.output_dir}")


# =============================================================================
# 入口
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT 微调脚本（Python 版本）"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="项目配置 JSON 文件路径（可选，未指定则使用默认配置）",
    )
    # 允许命令行覆盖关键参数
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument(
        "--deepspeed", type=str, default=None,
        help="DeepSpeed 配置文件路径（如 configs/deepspeed/ds_z3_config.json）"
             "。设为 'none' 禁用 DeepSpeed。需配合 torchrun 启动。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    if args.config and Path(args.config).exists():
        logger.info(f"从配置文件加载: {args.config}")
        cfg = ProjectConfig.load(args.config)
    else:
        logger.info("使用默认配置")
        cfg = load_project_config()

    # --deepspeed none 表示显式禁用
    if args.deepspeed is not None and args.deepspeed.lower() == "none":
        args.deepspeed = None

    # 命令行参数覆盖
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    for key, value in overrides.items():
        if hasattr(cfg.model, key):
            setattr(cfg.model, key, value)
        elif hasattr(cfg.lora, key):
            setattr(cfg.lora, key, value)
        elif hasattr(cfg.data, key):
            setattr(cfg.data, key, value)
        elif hasattr(cfg.training, key):
            setattr(cfg.training, key, value)
        else:
            logger.warning(f"未知参数: {key}={value}")

    logger.info("最终配置:")
    logger.info(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    train(cfg)


if __name__ == "__main__":
    main()
