"""
推理脚本 — 使用微调后的模型进行意图分类推理

说明：
  支持两种模式:
    1. 使用导出后的合并模型（推荐，对应 YAML: configs/inference.yaml）
    2. 使用基座模型 + LoRA 适配器（对应 YAML: configs/inference_lora.yaml）

用法：
  uv run python src/inference.py                                     # 交互式推理
  uv run python src/inference.py --input "我想买寿险"                  # 单条推理
  uv run python src/inference.py --input_file data/test.json          # 批量推理
  uv run python src/inference.py --adapter_path saves/qwen3-8b/...   # 使用 LoRA 适配器
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ProjectConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class IntentClassifier:
    """意图分类推理器"""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        trust_remote_code: bool = True,
        device: str = "auto",
    ):
        """
        初始化推理器

        Args:
            model_path: 模型路径（合并模型 或 基座模型）
            adapter_path: LoRA 适配器路径（仅在使用基座模型时需要）
            trust_remote_code: 是否信任远程代码
            device: 推理设备 (auto, cpu, cuda:0)
        """
        logger.info(f"加载模型: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            device_map=device if device != "cpu" else None,
        )

        # 如果提供了 adapter_path，加载 LoRA 适配器
        if adapter_path:
            logger.info(f"加载 LoRA 适配器: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def predict(
        self,
        text: str,
        instruction: str,
        temperature: float = 0.01,
        top_p: float = 0.1,
        top_k: int = 1,
        max_new_tokens: int = 16,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        对单条文本进行意图分类

        Args:
            text: 用户输入文本
            instruction: 系统指令
            temperature: 生成温度
            top_p: Top-p 采样
            top_k: Top-k 采样
            max_new_tokens: 最大生成 token 数
            repetition_penalty: 重复惩罚

        Returns:
            预测的意图标签
        """
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": text},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # 只取新生成的 token
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return result

    def predict_batch(
        self,
        texts: list[str],
        instruction: str,
        **kwargs,
    ) -> list[str]:
        """批量推理"""
        results = []
        for text in texts:
            result = self.predict(text, instruction, **kwargs)
            results.append(result)
        return results


def interactive_mode(classifier: IntentClassifier, cfg: ProjectConfig):
    """交互式推理模式"""
    instruction = cfg.data.get_instruction()
    print("\n" + "=" * 60)
    print("意图识别交互式推理")
    print(f"标签列表: {', '.join(cfg.data.labels)}")
    print("输入文本后按 Enter 进行推理，输入 'exit' 或 'quit' 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input("用户输入 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if text.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        if not text:
            continue

        result = classifier.predict(
            text,
            instruction,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
            top_k=cfg.inference.top_k,
            max_new_tokens=cfg.inference.max_new_tokens,
            repetition_penalty=cfg.inference.repetition_penalty,
        )
        print(f"预测结果 > {result}\n")


def batch_mode(
    classifier: IntentClassifier,
    cfg: ProjectConfig,
    input_file: str,
    output_file: Optional[str] = None,
):
    """批量推理模式"""
    instruction = cfg.data.get_instruction()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    correct = 0
    total = 0

    for item in data:
        text = item["input"]
        expected = item.get("output", None)

        predicted = classifier.predict(
            text,
            instruction,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
            top_k=cfg.inference.top_k,
            max_new_tokens=cfg.inference.max_new_tokens,
        )

        result_item = {
            "input": text,
            "predicted": predicted,
            "expected": expected,
        }

        if expected:
            is_correct = predicted.strip() == expected.strip()
            result_item["correct"] = is_correct
            if is_correct:
                correct += 1
            total += 1

        results.append(result_item)
        logger.info(f"输入: {text} → 预测: {predicted}" + (
            f" (期望: {expected}, {'✓' if result_item.get('correct') else '✗'})"
            if expected else ""
        ))

    if total > 0:
        accuracy = correct / total * 100
        logger.info(f"准确率: {correct}/{total} = {accuracy:.2f}%")

    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"推理结果已保存到: {output_file}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="意图分类推理脚本")
    parser.add_argument(
        "--config", type=str, default=None,
        help="项目配置 JSON 文件路径",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="模型路径（合并模型目录 或 基座模型路径）",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="LoRA 适配器路径（使用基座模型 + 适配器时需要）",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="单条推理文本",
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="批量推理输入文件（JSON 格式）",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="批量推理结果输出文件",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="推理设备 (auto, cpu, cuda:0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        cfg = ProjectConfig.load(args.config)
    else:
        cfg = ProjectConfig()

    # 确定模型路径
    model_path = args.model_path or cfg.export.export_dir

    # 如果提供了 adapter_path，使用基座模型
    if args.adapter_path:
        model_path = args.model_path or cfg.model.model_name_or_path

    classifier = IntentClassifier(
        model_path=model_path,
        adapter_path=args.adapter_path,
        trust_remote_code=cfg.model.trust_remote_code,
        device=args.device,
    )

    if args.input:
        # 单条推理
        instruction = cfg.data.get_instruction()
        result = classifier.predict(
            args.input,
            instruction,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
            top_k=cfg.inference.top_k,
            max_new_tokens=cfg.inference.max_new_tokens,
        )
        print(f"预测结果: {result}")
    elif args.input_file:
        # 批量推理
        batch_mode(classifier, cfg, args.input_file, args.output_file)
    else:
        # 交互式推理
        interactive_mode(classifier, cfg)


if __name__ == "__main__":
    main()
