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

    # ── 从 machine.env 读取，未设置时用 PROJECT_ROOT 派生默认值 ──────────────
    model_path = env.get(
        "MODEL_PATH",
        "/workspace/Qwen3-8B",
    )
    output_dir = env.get(
        "OUTPUT_DIR",
        str(project_root / "saves" / "qwen3-8b" / "lora" / "sft"),
    )
    dataset_dir = env.get(
        "DATASET_DIR",
        str(project_root / "data"),
    )
    deepspeed = env.get(
        "DEEPSPEED_CONFIG",
        # 默认使用 CPU offload 版本，确保在显存紧张的 V100 16GB 环境下可用。
        # 若目标机器显存充足（A100 40GB+），可在 machine.env 中改为
        # ds_z3_config.json（无 offload，速度更快）。
        str(project_root / "configs" / "deepspeed" / "ds_z3_offload_config.json"),
    )
    export_dir = env.get(
        "EXPORT_DIR",
        str(project_root / "models" / "qwen3-8b-intent-clf"),
    )

    return ProjectConfig(
        model=ModelConfig(
            model_name_or_path=model_path,
        ),
        data=DataConfig(
            train_file=str(Path(dataset_dir) / "train.json"),
            val_file=str(Path(dataset_dir) / "val.json"),
            test_file=str(Path(dataset_dir) / "test.json"),
        ),
        training=TrainingConfig(
            output_dir=output_dir,
            deepspeed=deepspeed,
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
    """LoRA 相关配置"""

    # 对应 YAML: finetuning_type
    finetuning_type: str = "lora"

    # 对应 YAML: lora_target
    lora_target: str = "all"

    # 对应 YAML: lora_rank
    lora_rank: int = 16

    # 对应 YAML: lora_alpha（通常为 rank 的 1~2 倍）
    lora_alpha: int = 32

    # 对应 YAML: lora_dropout
    lora_dropout: float = 0.05

    target_modules: Optional[List[str]] = None


@dataclass
class DataConfig:
    """数据相关配置"""

    # 以下三个路径由 load_project_config() 基于 machine.env 的 DATASET_DIR 派生
    # 对应 YAML: dataset_dir（间接）
    train_file: str = "/workspace/lora-intent-clf/data/train.json"
    val_file: str = "/workspace/lora-intent-clf/data/val.json"
    test_file: str = "/workspace/lora-intent-clf/data/test.json"

    # 对应 YAML: cutoff_len
    max_seq_length: int = 512

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
    """训练超参数配置"""

    # 对应 YAML: output_dir
    # 由 machine.env 的 OUTPUT_DIR 覆盖（通过 load_project_config()）
    output_dir: str = "/workspace/lora-intent-clf/saves/qwen3-8b/lora/sft"

    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    optim: str = "adamw_torch"
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 500
    gradient_checkpointing: bool = True
    report_to: str = "tensorboard"

    # 对应 YAML: deepspeed
    # 由 machine.env 的 DEEPSPEED_CONFIG 覆盖（通过 load_project_config()）
    # 默认 offload 版本：兼容 V100 16GB；A100 40GB+ 可改为 ds_z3_config.json
    deepspeed: Optional[str] = "/workspace/lora-intent-clf/configs/deepspeed/ds_z3_offload_config.json"


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
