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

import compat  # numpy 兼容垫片，必须在 deepspeed/torch 之前

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
    EarlyStoppingCallback,
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
# DeepSpeed 工具
# =============================================================================


def _resolve_ds_auto_values(ds_dict: dict, cfg: "ProjectConfig") -> dict:
    """
    将 DeepSpeed 配置中的 'auto' 占位符替换为实际计算值。

    根本原因：
      ds_z3_offload_config.json 中使用 "auto" 作为占位符，设计上由
      HuggingFace Trainer 的 trainer_config_process() 在训练时替换。
      但在 from_pretrained 之前手动创建 HfDeepSpeedConfig(_ds_dict) 时，
      某些版本的 DeepSpeed (>=0.14) 会在 DeepSpeedConfig.__init__ 中调用
      _batch_assertion()，此时 "auto" 尚未被替换，导致：
        TypeError: '>' not supported between instances of 'str' and 'int'
        (assert train_batch_size > 0)

    修复：提前将 "auto" 替换为基于 machine.env 参数的实际值，再创建
    HfDeepSpeedConfig。Trainer 自己创建的那份 HfDeepSpeedConfig（从路径加载）
    走标准的 trainer_config_process() 路径，不受影响。
    """
    import copy

    ds = copy.deepcopy(ds_dict)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_bs = int(cfg.training.per_device_train_batch_size)
    grad_accum = int(cfg.training.gradient_accumulation_steps)
    global_bs = world_size * per_device_bs * grad_accum

    if ds.get("train_batch_size") == "auto":
        ds["train_batch_size"] = global_bs
    if ds.get("train_micro_batch_size_per_gpu") == "auto":
        ds["train_micro_batch_size_per_gpu"] = per_device_bs
    if ds.get("gradient_accumulation_steps") == "auto":
        ds["gradient_accumulation_steps"] = grad_accum
    if ds.get("gradient_clipping") == "auto":
        ds["gradient_clipping"] = float(cfg.training.max_grad_norm)

    fp16 = ds.get("fp16", {})
    if fp16.get("enabled") == "auto":
        fp16["enabled"] = bool(cfg.training.fp16)

    bf16 = ds.get("bf16", {})
    if bf16.get("enabled") == "auto":
        bf16["enabled"] = not bool(cfg.training.fp16)

    # zero_optimization 中的 "auto" — 用保守的大值作为上界默认值
    zero = ds.get("zero_optimization", {})
    if zero.get("reduce_bucket_size") == "auto":
        zero["reduce_bucket_size"] = int(5e8)
    if zero.get("stage3_prefetch_bucket_size") == "auto":
        zero["stage3_prefetch_bucket_size"] = int(4.5e8)
    if zero.get("stage3_param_persistence_threshold") == "auto":
        zero["stage3_param_persistence_threshold"] = int(1e6)

    logger.info(
        f"DeepSpeed 配置已解析（'auto' 已替换）："
        f" global_bs={global_bs}  per_device_bs={per_device_bs}"
        f"  grad_accum={grad_accum}  fp16={bool(cfg.training.fp16)}"
    )
    return ds


# =============================================================================
# GPU 诊断工具
# =============================================================================

def _gpu_memory_summary(tag: str = "") -> None:
    """
    打印当前所有 GPU 的显存使用情况，用于 OOM 问题排查。
    仅在 rank=0 进程打印（避免多进程重复输出）。
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return
    if not torch.cuda.is_available():
        logger.info(f"[GPU-MEM {tag}] CUDA 不可用")
        return

    sep = "─" * 62
    logger.info(sep)
    logger.info(f"[GPU-MEM] {tag}")
    for i in range(torch.cuda.device_count()):
        alloc  = torch.cuda.memory_allocated(i)  / 1024 ** 3
        reserv = torch.cuda.memory_reserved(i)   / 1024 ** 3
        total  = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
        logger.info(
            f"  GPU {i}: allocated={alloc:.2f}GB  reserved={reserv:.2f}GB  total={total:.1f}GB"
        )
    logger.info(sep)


def _print_config(cfg) -> None:
    """打印已解析的全量配置，rank=0 进程打印"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return
    sep = "═" * 62
    logger.info(sep)
    logger.info("[CONFIG] 当前训练配置（来自 machine.env + 代码默认值）")
    logger.info(sep)
    logger.info(f"  model_name_or_path  = {cfg.model.model_name_or_path}")
    logger.info(f"  output_dir          = {cfg.training.output_dir}")
    logger.info(f"  dataset_dir (train) = {cfg.data.train_file}")
    logger.info(f"  deepspeed           = {cfg.training.deepspeed}")
    logger.info(f"  deepspeed_exists    = {Path(cfg.training.deepspeed).exists() if cfg.training.deepspeed else False}")
    logger.info(f"  export_dir          = {cfg.export.export_dir}")
    logger.info(f"  lora_rank           = {cfg.lora.lora_rank}")
    logger.info(f"  lora_alpha          = {cfg.lora.lora_alpha}")
    logger.info(f"  lora_dropout        = {cfg.lora.lora_dropout}")
    logger.info(f"  lora_target         = {cfg.lora.lora_target}")
    logger.info(f"  fp16                = {cfg.training.fp16}")
    logger.info(f"  cutoff_len          = {cfg.data.max_seq_length}")
    logger.info(f"  per_device_bs       = {cfg.training.per_device_train_batch_size}")
    logger.info(f"  grad_accum_steps    = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  learning_rate       = {cfg.training.learning_rate}")
    logger.info(f"  num_train_epochs    = {cfg.training.num_train_epochs}")
    logger.info(f"  warmup_ratio        = {cfg.training.warmup_ratio}")
    logger.info(f"  eval_steps          = {cfg.training.eval_steps}")
    logger.info(f"  save_steps          = {cfg.training.save_steps}")
    logger.info(f"  early_stopping_patience   = {cfg.training.early_stopping_patience}")
    logger.info(f"  early_stopping_threshold  = {cfg.training.early_stopping_threshold}")
    logger.info(f"  GPU 数量 (LOCAL/ENV) = {torch.cuda.device_count()} / WORLD_SIZE={os.environ.get('WORLD_SIZE', '?')}")
    logger.info(sep)


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

    # gradient checkpointing 要求禁用 KV cache（否则报 Warning 且梯度错误）
    model.config.use_cache = False

    # PEFT + gradient checkpointing：确保 input embedding 产生梯度
    # 若不设置，部分模型（如 Qwen3）的 embedding 层梯度会被截断，
    # 导致靠近 embedding 的 LoRA adapter 实际未被训练
    model.enable_input_require_grads()

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
    # ── ZeRO-3 关键初始化（必须在 from_pretrained 之前）────────────────────────
    # 背景：
    #   若不在 from_pretrained 之前创建 HfDeepSpeedConfig，每个 torchrun 进程
    #   会各自将完整模型（Qwen3-8B FP16 ≈ 16GB）加载到 CPU RAM，
    #   N 卡训练需要 N×16GB CPU RAM，且 ZeRO-3 参数分片是训练开始后才执行。
    #
    # 修复方案：
    #   HfDeepSpeedConfig 会激活 deepspeed.zero.Init() 上下文，
    #   使 from_pretrained 在加载时即将参数分片，每个进程仅持有 1/N 的参数。
    #
    # 注意：_dschf 必须保持强引用直到 trainer.train() 结束，
    #       HuggingFace 内部用 weakref 持有该对象，GC 回收后配置失效。
    # 打印全量配置（rank=0），方便确认参数是否正确加载
    _print_config(cfg)
    _gpu_memory_summary("训练开始前")

    _dschf = None  # noqa: SIM assignment — must not be reassigned or deleted
    if cfg.training.deepspeed:
        _ds_path = Path(cfg.training.deepspeed)
        if _ds_path.exists():
            from transformers.integrations import HfDeepSpeedConfig
            with open(_ds_path, encoding="utf-8") as _f:
                _ds_dict = json.load(_f)
            if _ds_dict.get("zero_optimization", {}).get("stage") == 3:
                # 替换 "auto" 值，防止 DeepSpeed _batch_assertion() 在 __init__ 中
                # 因 "auto" > 0 比较触发 TypeError（见 _resolve_ds_auto_values 注释）
                _ds_resolved = _resolve_ds_auto_values(_ds_dict, cfg)
                _dschf = HfDeepSpeedConfig(_ds_resolved)
                logger.info("ZeRO-3: HfDeepSpeedConfig 已激活，from_pretrained 将自动分片参数")
        else:
            logger.warning(f"DeepSpeed 配置文件不存在: {_ds_path}，将不使用 DeepSpeed")

    model, tokenizer = build_model_and_tokenizer(cfg)
    _gpu_memory_summary("模型+PEFT加载后")

    # 加载数据
    train_data_raw = load_jsonl_data(cfg.data.train_file)
    val_data_raw = load_jsonl_data(cfg.data.val_file)

    train_dataset = preprocess_data(train_data_raw, tokenizer, cfg.data)
    val_dataset = preprocess_data(val_data_raw, tokenizer, cfg.data)

    logger.info(f"训练集: {len(train_dataset)} 条，验证集: {len(val_dataset)} 条")

    # 构建 TrainingArguments（与 YAML 参数一一对应，均可通过 machine.env 覆盖）
    # ⚠️ save_strategy 必须与 eval_strategy 一致，且 save_steps == eval_steps，
    #    否则 load_best_model_at_end=True 和 EarlyStoppingCallback 会报错。
    # 注：显式 int()/float()/bool() 转换为防御性编程，确保运行时类型正确
    #     （dataclass 字段类型注解在 Python 中仅作文档用途，不做强制校验）。
    training_args = TrainingArguments(
        output_dir=str(cfg.training.output_dir),
        num_train_epochs=float(cfg.training.num_train_epochs),
        per_device_train_batch_size=int(cfg.training.per_device_train_batch_size),
        per_device_eval_batch_size=int(cfg.training.per_device_eval_batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        learning_rate=float(cfg.training.learning_rate),
        lr_scheduler_type=str(cfg.training.lr_scheduler_type),
        warmup_ratio=float(cfg.training.warmup_ratio),
        weight_decay=float(cfg.training.weight_decay),
        max_grad_norm=float(cfg.training.max_grad_norm),
        fp16=bool(cfg.training.fp16),
        optim=str(cfg.training.optim),
        seed=int(cfg.training.seed),
        logging_steps=int(cfg.training.logging_steps),
        # save_strategy / eval_strategy 必须一致，save_steps 必须等于 eval_steps
        save_strategy=str(cfg.training.eval_strategy),
        save_steps=int(cfg.training.save_steps),
        save_total_limit=int(cfg.training.save_total_limit),
        eval_strategy=str(cfg.training.eval_strategy),
        eval_steps=int(cfg.training.eval_steps),
        # 早停必须：训练结束后自动加载最优 checkpoint（eval_loss 最低）
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=bool(cfg.training.gradient_checkpointing),
        report_to=str(cfg.training.report_to),
        logging_dir=os.path.join(str(cfg.training.output_dir), "logs"),
        overwrite_output_dir=True,
        remove_unused_columns=False,
        ddp_timeout=180000000,
        deepspeed=cfg.training.deepspeed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # 早停 callback：连续 patience 次评估 eval_loss 无改善则停止，并恢复最优 checkpoint
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=int(cfg.training.early_stopping_patience),
        early_stopping_threshold=float(cfg.training.early_stopping_threshold),
    )
    logger.info(
        f"早停已启用：patience={cfg.training.early_stopping_patience} 次评估 "
        f"({cfg.training.early_stopping_patience * cfg.training.eval_steps} 步)，"
        f"threshold={cfg.training.early_stopping_threshold}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_cb],
    )

    _gpu_memory_summary("Trainer 初始化后、训练前")
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
