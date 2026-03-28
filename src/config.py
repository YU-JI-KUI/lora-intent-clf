"""
统一配置模块 — 所有训练/推理/导出参数集中管理

说明：
  本模块使用 dataclass 定义所有配置项，与 YAML 配置文件中的参数一一对应。
  参数名可能不同（Python 使用下划线风格），但语义完全一致。

对应关系表（Python config ↔ YAML config）：
  ┌─────────────────────────────────┬─────────────────────────────────┐
  │ Python (TrainingConfig)         │ YAML (train_lora_sft.yaml)      │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │ model_name_or_path              │ model_name_or_path              │
  │ template                        │ template                        │
  │ finetuning_type                 │ finetuning_type                 │
  │ lora_target                     │ lora_target                     │
  │ lora_rank                       │ lora_rank                       │
  │ lora_alpha                      │ lora_alpha                      │
  │ lora_dropout                    │ lora_dropout                    │
  │ output_dir                      │ output_dir                      │
  │ num_train_epochs                │ num_train_epochs                │
  │ per_device_train_batch_size     │ per_device_train_batch_size     │
  │ per_device_eval_batch_size      │ per_device_eval_batch_size      │
  │ gradient_accumulation_steps     │ gradient_accumulation_steps     │
  │ learning_rate                   │ learning_rate                   │
  │ lr_scheduler_type               │ lr_scheduler_type               │
  │ warmup_ratio                    │ warmup_ratio                    │
  │ weight_decay                    │ weight_decay                    │
  │ max_grad_norm                   │ max_grad_norm                   │
  │ fp16                            │ fp16                            │
  │ optim                           │ optim                           │
  │ seed                            │ seed                            │
  │ logging_steps                   │ logging_steps                   │
  │ save_steps                      │ save_steps                      │
  │ save_total_limit                │ save_total_limit                │
  │ eval_strategy                   │ eval_strategy                   │
  │ eval_steps                      │ eval_steps                      │
  │ gradient_checkpointing          │ gradient_checkpointing          │
  │ report_to                       │ report_to                       │
  │ deepspeed                       │ deepspeed                       │
  │ cutoff_len (→ max_seq_length)   │ cutoff_len                      │
  │ labels                          │ (在 instruction prompt 中定义)   │
  └─────────────────────────────────┴─────────────────────────────────┘
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型相关配置"""

    # 模型名称或本地路径（HuggingFace Hub ID 或本地绝对路径）
    # 对应 YAML: model_name_or_path
    # 当前使用 ANS 网盘挂载路径（只读），无需拷贝到工作目录
    model_name_or_path: str = "/workspace/Qwen3-8B"

    # 对话模板，需要与模型匹配
    # 对应 YAML: template
    # 可选项: qwen, llama3, chatglm4, baichuan2, default
    template: str = "qwen"

    # 是否信任远程代码（部分模型需要）
    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    """LoRA 相关配置"""

    # 微调类型
    # 对应 YAML: finetuning_type
    # 可选项: lora, freeze, full
    finetuning_type: str = "lora"

    # LoRA 目标模块
    # 对应 YAML: lora_target
    # 可选项: all, q_proj,v_proj, q_proj,k_proj,v_proj,o_proj
    lora_target: str = "all"

    # LoRA 秩
    # 对应 YAML: lora_rank
    # 可选项: 8, 16, 32, 64, 128
    lora_rank: int = 16

    # LoRA 缩放因子（通常为 rank 的 1~2 倍）
    # 对应 YAML: lora_alpha
    lora_alpha: int = 32

    # LoRA Dropout 比率
    # 对应 YAML: lora_dropout
    # 可选项: 0.0, 0.05, 0.1, 0.2
    lora_dropout: float = 0.05

    # 目标模块列表（更精细的控制）
    # 如果设置了此项，将覆盖 lora_target
    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    """数据相关配置"""

    # 训练数据文件路径（使用绝对路径）
    train_file: str = "/workspace/lora-intent-clf/data/train.json"

    # 验证数据文件路径（使用绝对路径）
    val_file: str = "/workspace/lora-intent-clf/data/val.json"

    # 测试数据文件路径（使用绝对路径）
    test_file: str = "/workspace/lora-intent-clf/data/test.json"

    # 最大序列长度
    # 对应 YAML: cutoff_len
    # 可选项: 256, 512, 1024, 2048
    max_seq_length: int = 512

    # 数据预处理线程数
    # 对应 YAML: preprocessing_num_workers
    preprocessing_num_workers: int = 16

    # 系统提示词（instruction 部分）
    system_prompt: str = (
        "你是一个意图识别助手。请根据用户输入的文本，判断其意图类别。"
        "只能输出以下标签之一：{labels}。"
    )

    # 标签列表（可扩展为多分类）
    labels: List[str] = field(
        default_factory=lambda: ["寿险意图", "拒识"]
    )

    def get_instruction(self) -> str:
        """生成完整的 instruction 文本"""
        return self.system_prompt.format(
            labels="、".join(self.labels)
        )


@dataclass
class TrainingConfig:
    """训练超参数配置"""

    # 输出目录（使用绝对路径）
    # 对应 YAML: output_dir
    output_dir: str = "/workspace/lora-intent-clf/saves/qwen3-8b/lora/sft"

    # 训练轮数
    # 对应 YAML: num_train_epochs
    num_train_epochs: float = 5.0

    # 每设备训练批次大小
    # 对应 YAML: per_device_train_batch_size
    per_device_train_batch_size: int = 2

    # 每设备评估批次大小
    # 对应 YAML: per_device_eval_batch_size
    per_device_eval_batch_size: int = 4

    # 梯度累积步数
    # 对应 YAML: gradient_accumulation_steps
    gradient_accumulation_steps: int = 4

    # 学习率
    # 对应 YAML: learning_rate
    learning_rate: float = 1e-4

    # 学习率调度器类型
    # 对应 YAML: lr_scheduler_type
    # 可选项: linear, cosine, cosine_with_restarts, polynomial, constant
    lr_scheduler_type: str = "cosine"

    # 预热比例
    # 对应 YAML: warmup_ratio
    warmup_ratio: float = 0.1

    # 权重衰减
    # 对应 YAML: weight_decay
    weight_decay: float = 0.01

    # 最大梯度范数（梯度裁剪）
    # 对应 YAML: max_grad_norm
    max_grad_norm: float = 1.0

    # 使用 FP16 混合精度（V100 不支持 BF16）
    # 对应 YAML: fp16
    fp16: bool = True

    # 优化器
    # 对应 YAML: optim
    # 可选项: adamw_torch, adamw_hf, adafactor, sgd
    optim: str = "adamw_torch"

    # 随机种子
    # 对应 YAML: seed
    seed: int = 42

    # 日志记录间隔（步数）
    # 对应 YAML: logging_steps
    logging_steps: int = 10

    # 模型保存间隔（步数）
    # 对应 YAML: save_steps
    save_steps: int = 500

    # 最大 checkpoint 保留数量
    # 对应 YAML: save_total_limit
    save_total_limit: int = 3

    # 评估策略
    # 对应 YAML: eval_strategy
    # 可选项: steps, epoch, no
    eval_strategy: str = "steps"

    # 评估间隔（步数）
    # 对应 YAML: eval_steps
    eval_steps: int = 500

    # 梯度检查点（用计算换显存）
    # 对应 YAML: gradient_checkpointing
    gradient_checkpointing: bool = True

    # 日志输出目标
    # 对应 YAML: report_to
    # 可选项: tensorboard, wandb, mlflow, none
    report_to: str = "tensorboard"

    # DeepSpeed 配置文件路径（可选，用于多卡分布式训练）
    # 对应 YAML: deepspeed
    # 用法：需配合 torchrun 启动，不能用 python 直接运行
    #   torchrun --nproc_per_node=4 src/train.py --deepspeed configs/deepspeed/ds_z3_config.json
    # None 表示不使用 DeepSpeed（单卡或普通 DDP）
    deepspeed: Optional[str] = "/workspace/lora-intent-clf/configs/deepspeed/ds_z3_config.json"


@dataclass
class ExportConfig:
    """模型导出配置"""

    # 导出目录（使用绝对路径）
    # 对应 YAML: export_dir
    export_dir: str = "/workspace/lora-intent-clf/models/qwen3-8b-intent-clf"

    # 导出分片大小 (GB)
    # 对应 YAML: export_size
    export_size: int = 2

    # 导出设备
    # 对应 YAML: export_device
    export_device: str = "cpu"


@dataclass
class InferenceConfig:
    """推理配置"""

    # 温度
    temperature: float = 0.01

    # Top-p 采样
    top_p: float = 0.1

    # Top-k 采样
    top_k: int = 1

    # 最大生成 token 数
    max_new_tokens: int = 16

    # 重复惩罚
    repetition_penalty: float = 1.0


@dataclass
class ProjectConfig:
    """项目总配置 — 聚合所有子配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        """转换为嵌套字典"""
        return asdict(self)

    def save(self, path: str) -> None:
        """保存配置到 JSON 文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoraConfig(**data.get("lora", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            export=ExportConfig(**data.get("export", {})),
            inference=InferenceConfig(**data.get("inference", {})),
        )
