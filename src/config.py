"""
统一配置模块 — 所有训练/推理/导出参数集中管理

说明：
  本模块使用 dataclass 定义所有配置项，与 YAML 配置文件中的参数一一对应。
  机器特定参数（模型路径、输出目录等）从 machine.env 自动加载，
  无需手动修改代码或 YAML 文件。

使用方式：
  # 推荐：自动从 machine.env 加载机器特定参数
  from config import load_project_config
  cfg = load_project_config()

  # 备选：使用代码默认值（不读取 machine.env）
  from config import ProjectConfig
  cfg = ProjectConfig()

machine.env 中的参数对应关系：
  MODEL_PATH       → cfg.model.model_name_or_path
  OUTPUT_DIR       → cfg.training.output_dir  /  cfg.export.adapter_name_or_path
  DATASET_DIR      → cfg.data.train_file / val_file / test_file 的目录部分
  DEEPSPEED_CONFIG → cfg.training.deepspeed
  EXPORT_DIR       → cfg.export.export_dir
  NPROC_PER_NODE   → 仅 shell 脚本使用（torchrun），Python 层不使用

YAML ↔ Python 对应关系表：
  ┌─────────────────────────────────┬─────────────────────────────────┐
  │ Python (TrainingConfig)         │ YAML (train_lora_sft.yaml)      │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │ model_name_or_path              │ model_name_or_path              │
  │ template                        │ template                        │
  │ output_dir                      │ output_dir                      │
  │ deepspeed                       │ deepspeed                       │
  │ ... (其余超参见 TrainingConfig) │ ...                             │
  └─────────────────────────────────┴─────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# machine.env 加载器
# =============================================================================

def _load_machine_env() -> Dict[str, str]:
    """
    读取项目根目录的 machine.env 文件。
    若 machine.env 不存在，回退到 machine.env.example。
    返回 key=value 字典（仅纯字符串，不展开 shell 变量）。
    """
    # src/config.py → PROJECT_ROOT = src/ 的上级
    project_root = Path(__file__).parent.parent

    env_file = project_root / "machine.env"
    if not env_file.exists():
        env_file = project_root / "machine.env.example"

    result: Dict[str, str] = {}
    if not env_file.exists():
        return result

    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith("#"):
                continue
            # 跳过注释掉的可选参数（如 # OUTPUT_DIR=...）
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 跳过值为空的行
            if key and value:
                result[key] = value

    return result


def load_project_config() -> "ProjectConfig":
    """
    从 machine.env 加载机器特定参数，构建 ProjectConfig。

    优先级：machine.env > machine.env.example > 代码默认值

    推荐在所有 Python 入口脚本（train.py / export_model.py / evaluate.py）
    中使用此函数，而非直接实例化 ProjectConfig()。
    """
    env = _load_machine_env()
    project_root = Path(__file__).parent.parent

    def _e(key: str, default: str) -> str:
        """读取 env，未设置时返回 default"""
        return env.get(key, default)

    # ── [路径] 从 machine.env 读取，未设置时用 PROJECT_ROOT 派生默认值 ────────
    model_path  = _e("MODEL_PATH",  "/workspace/Qwen3-8B")
    output_dir  = _e("OUTPUT_DIR",  str(project_root / "saves" / "qwen3-8b" / "lora" / "sft"))
    dataset_dir = _e("DATASET_DIR", str(project_root / "data"))
    deepspeed   = _e(
        "DEEPSPEED_CONFIG",
        # 默认 CPU offload 版本，兼容 V100 16GB；A100 40GB+ 可改为 ds_z3_config.json
        str(project_root / "configs" / "deepspeed" / "ds_z3_offload_config.json"),
    )
    export_dir  = _e("EXPORT_DIR",  str(project_root / "models" / "qwen3-8b-intent-clf"))

    # ── [LoRA] 超参 ────────────────────────────────────────────────────────────
    lora_rank    = int(_e("LORA_RANK",    "8"))
    lora_alpha   = int(_e("LORA_ALPHA",   "16"))
    lora_dropout = float(_e("LORA_DROPOUT", "0.1"))
    lora_target  = _e("LORA_TARGET", "all")

    # ── [训练] 超参 ────────────────────────────────────────────────────────────
    cutoff_len              = int(_e("CUTOFF_LEN",               "256"))
    per_device_train_bs     = int(_e("PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
    gradient_accum_steps    = int(_e("GRADIENT_ACCUMULATION_STEPS", "4"))
    learning_rate           = float(_e("LEARNING_RATE",          "2e-4"))
    num_train_epochs        = float(_e("NUM_TRAIN_EPOCHS",        "10"))
    warmup_ratio            = float(_e("WARMUP_RATIO",            "0.05"))
    weight_decay            = float(_e("WEIGHT_DECAY",            "0.01"))
    max_grad_norm           = float(_e("MAX_GRAD_NORM",           "1.0"))

    # ── [监控 & 早停] ──────────────────────────────────────────────────────────
    logging_steps               = int(_e("LOGGING_STEPS",           "20"))
    eval_steps                  = int(_e("EVAL_STEPS",              "200"))
    save_steps                  = int(_e("SAVE_STEPS",              str(eval_steps)))
    early_stopping_patience     = int(_e("EARLY_STOPPING_PATIENCE", "5"))
    early_stopping_threshold    = float(_e("EARLY_STOPPING_THRESHOLD", "1e-4"))

    return ProjectConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
        ),
        lora=LoraConfig(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target=lora_target,
        ),
        data=DataConfig(
            train_file=str(Path(dataset_dir) / "train.json"),
            val_file=str(Path(dataset_dir) / "val.json"),
            test_file=str(Path(dataset_dir) / "test.json"),
            max_seq_length=cutoff_len,
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            deepspeed=deepspeed,
            per_device_train_batch_size=per_device_train_bs,
            gradient_accumulation_steps=gradient_accum_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ),
        export=ExportConfig(
            export_dir=export_dir,
        ),
    )


# =============================================================================
# 配置 dataclass 定义
# =============================================================================

@dataclass
class ModelConfig:
    """模型相关配置"""

    # 对应 YAML: model_name_or_path
    # 由 machine.env 的 MODEL_PATH 覆盖（通过 load_project_config()）
    model_name_or_path: str = "/workspace/Qwen3-8B"

    # 对应 YAML: template
    template: str = "qwen"

    trust_remote_code: bool = True


@dataclass
class LoraConfig:
    """LoRA 相关配置（所有字段均可通过 machine.env 覆盖）"""

    # 对应 YAML: finetuning_type
    finetuning_type: str = "lora"

    # 对应 YAML: lora_target / machine.env: LORA_TARGET
    lora_target: str = "all"

    # 对应 YAML: lora_rank / machine.env: LORA_RANK
    # 二分类简单任务推荐 8；复杂任务可用 16/32
    lora_rank: int = 8

    # 对应 YAML: lora_alpha / machine.env: LORA_ALPHA
    # 通常为 lora_rank 的 2 倍（缩放比 = alpha/rank）
    lora_alpha: int = 16

    # 对应 YAML: lora_dropout / machine.env: LORA_DROPOUT
    lora_dropout: float = 0.1

    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    """数据相关配置"""

    # 以下三个路径由 load_project_config() 基于 machine.env 的 DATASET_DIR 派生
    # 对应 YAML: dataset_dir（间接）
    train_file: str = "/workspace/lora-intent-clf/data/train.json"
    val_file: str = "/workspace/lora-intent-clf/data/val.json"
    test_file: str = "/workspace/lora-intent-clf/data/test.json"

    # 对应 YAML: cutoff_len / machine.env: CUTOFF_LEN
    # 意图识别文本通常 < 100 tokens，256 足够且节省显存
    max_seq_length: int = 256

    # 对应 YAML: preprocessing_num_workers
    preprocessing_num_workers: int = 16

    # 系统提示词模板（{labels} 占位符由 get_instruction() 替换）
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
        return self.system_prompt.format(labels="、".join(self.labels))


@dataclass
class TrainingConfig:
    """训练超参数配置（所有字段均可通过 machine.env 覆盖）"""

    # 对应 YAML: output_dir / machine.env: OUTPUT_DIR
    output_dir: str = "/workspace/lora-intent-clf/saves/qwen3-8b/lora/sft"

    # machine.env: NUM_TRAIN_EPOCHS — 配合早停可以设大一些，由模型表现决定何时停止
    num_train_epochs: float = 10.0
    # machine.env: PER_DEVICE_TRAIN_BATCH_SIZE
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    # machine.env: GRADIENT_ACCUMULATION_STEPS
    gradient_accumulation_steps: int = 4
    # machine.env: LEARNING_RATE
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    # machine.env: WARMUP_RATIO
    warmup_ratio: float = 0.05
    # machine.env: WEIGHT_DECAY
    weight_decay: float = 0.01
    # machine.env: MAX_GRAD_NORM
    max_grad_norm: float = 1.0
    fp16: bool = True
    optim: str = "adamw_torch"
    seed: int = 42
    # machine.env: LOGGING_STEPS
    logging_steps: int = 20
    # machine.env: SAVE_STEPS — 必须等于 eval_steps，早停依赖每次评估后的 checkpoint
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    # machine.env: EVAL_STEPS
    eval_steps: int = 200
    gradient_checkpointing: bool = True
    report_to: str = "tensorboard"

    # 对应 YAML: deepspeed / machine.env: DEEPSPEED_CONFIG
    # 默认 offload 版本：兼容 V100 16GB；A100 40GB+ 可改为 ds_z3_config.json
    deepspeed: Optional[str] = "/workspace/lora-intent-clf/configs/deepspeed/ds_z3_offload_config.json"

    # ── 早停配置 / machine.env: EARLY_STOPPING_PATIENCE / EARLY_STOPPING_THRESHOLD
    # 监控 eval_loss，连续 patience 次评估无改善则停止训练，并自动恢复最优 checkpoint
    # patience=5 × eval_steps=200 ≈ 1000步 ≈ 1.3轮（以50K数据/8卡为基准）无改善时停止
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4


@dataclass
class ExportConfig:
    """模型导出配置"""

    # 对应 YAML: export_dir
    # 由 machine.env 的 EXPORT_DIR 覆盖（通过 load_project_config()）
    export_dir: str = "/workspace/lora-intent-clf/models/qwen3-8b-intent-clf"

    export_size: int = 2
    export_device: str = "cpu"


@dataclass
class InferenceConfig:
    """推理配置"""

    temperature: float = 0.01
    top_p: float = 0.1
    top_k: int = 1
    max_new_tokens: int = 16
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
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProjectConfig":
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
