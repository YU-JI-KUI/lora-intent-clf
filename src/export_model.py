"""
模型导出脚本 — 将 LoRA 适配器合并到基座模型

说明：
  将训练产出的 LoRA 适配器合并进基座模型，生成可独立部署的完整模型。

用法：
  uv run python src/export_model.py                           # 使用默认配置
  uv run python src/export_model.py --config config.json      # 使用自定义配置
  uv run python src/export_model.py --adapter_path saves/...  # 指定适配器路径

对应 YAML: configs/export_lora.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ProjectConfig, load_project_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def export_merged_model(cfg: ProjectConfig, adapter_path: str | None = None) -> None:
    """
    合并 LoRA 适配器到基座模型并保存

    对应 YAML 参数:
      model_name_or_path     → cfg.model.model_name_or_path
      adapter_name_or_path   → adapter_path 或 cfg.training.output_dir
      export_dir             → cfg.export.export_dir
      export_size            → cfg.export.export_size
      export_device          → cfg.export.export_device
    """
    adapter_dir = adapter_path or cfg.training.output_dir

    logger.info(f"基座模型: {cfg.model.model_name_or_path}")
    logger.info(f"适配器路径: {adapter_dir}")
    logger.info(f"导出目录: {cfg.export.export_dir}")
    logger.info(f"导出设备: {cfg.export.export_device}")

    # 加载基座模型（在 CPU 上，避免 GPU 显存不足）
    device = cfg.export.export_device
    logger.info(f"在 {device} 上加载基座模型...")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=torch.float16,
        device_map=device if device != "cpu" else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
    )

    # 加载 LoRA 适配器
    logger.info("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(model, adapter_dir)

    # 合并适配器
    logger.info("合并 LoRA 适配器到基座模型...")
    model = model.merge_and_unload()

    # 保存合并后的模型
    export_dir = Path(cfg.export.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存合并模型到: {export_dir}")
    # 分片保存（对应 YAML: export_size）
    max_shard_size = f"{cfg.export.export_size}GB"
    model.save_pretrained(
        export_dir,
        max_shard_size=max_shard_size,
        safe_serialization=True,  # 对应 YAML: export_legacy_format=false → safetensors
    )
    tokenizer.save_pretrained(export_dir)

    logger.info(f"模型导出完成！保存在: {export_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA 模型导出脚本")
    parser.add_argument(
        "--config", type=str, default=None,
        help="项目配置 JSON 文件路径",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="LoRA 适配器路径（覆盖配置中的 output_dir）",
    )
    parser.add_argument(
        "--export_dir", type=str, default=None,
        help="合并模型输出路径",
    )
    parser.add_argument(
        "--export_device", type=str, default=None,
        choices=["cpu", "auto"],
        help="导出使用的设备",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=None,
        help="基座模型路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        cfg = ProjectConfig.load(args.config)
    else:
        cfg = load_project_config()

    # 命令行覆盖
    if args.model_name_or_path:
        cfg.model.model_name_or_path = args.model_name_or_path
    if args.export_dir:
        cfg.export.export_dir = args.export_dir
    if args.export_device:
        cfg.export.export_device = args.export_device

    export_merged_model(cfg, adapter_path=args.adapter_path)


if __name__ == "__main__":
    main()
