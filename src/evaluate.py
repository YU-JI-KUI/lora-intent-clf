"""
评估脚本 — 在测试集上评估微调后模型的分类性能

说明：
  计算准确率、精确率、召回率、F1-score，并生成分类报告。

用法：
  uv run python src/evaluate.py                                      # 使用默认配置
  uv run python src/evaluate.py --test_file data/test.json           # 指定测试集
  uv run python src/evaluate.py --adapter_path saves/qwen3-8b/...   # 使用 LoRA 适配器
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from config import ProjectConfig
from inference import IntentClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """
    计算分类指标（不依赖 sklearn，纯 Python 实现）

    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 所有可能的标签

    Returns:
        包含各项指标的字典
    """
    assert len(y_true) == len(y_pred), "真实标签和预测标签数量不一致"

    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    # 每个标签的 TP, FP, FN
    metrics_per_label = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        support = sum(1 for t in y_true if t == label)
        metrics_per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # 加权平均
    total_support = sum(m["support"] for m in metrics_per_label.values())
    macro_precision = sum(m["precision"] for m in metrics_per_label.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in metrics_per_label.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in metrics_per_label.values()) / len(labels)

    weighted_precision = (
        sum(m["precision"] * m["support"] for m in metrics_per_label.values()) / total_support
        if total_support > 0 else 0.0
    )
    weighted_recall = (
        sum(m["recall"] * m["support"] for m in metrics_per_label.values()) / total_support
        if total_support > 0 else 0.0
    )
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in metrics_per_label.values()) / total_support
        if total_support > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_label": metrics_per_label,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
    }


def print_classification_report(metrics: dict, labels: list[str]) -> None:
    """打印分类报告（类似 sklearn 的格式）"""
    print("\n" + "=" * 70)
    print("分类评估报告")
    print("=" * 70)
    print(f"{'标签':<15} {'精确率':>10} {'召回率':>10} {'F1':>10} {'支持数':>10}")
    print("-" * 70)

    for label in labels:
        m = metrics["per_label"][label]
        print(
            f"{label:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10d}"
        )

    print("-" * 70)
    print(
        f"{'macro avg':<15} {metrics['macro']['precision']:>10.4f} "
        f"{metrics['macro']['recall']:>10.4f} {metrics['macro']['f1']:>10.4f} "
        f"{metrics['total']:>10d}"
    )
    print(
        f"{'weighted avg':<15} {metrics['weighted']['precision']:>10.4f} "
        f"{metrics['weighted']['recall']:>10.4f} {metrics['weighted']['f1']:>10.4f} "
        f"{metrics['total']:>10d}"
    )
    print("-" * 70)
    print(f"准确率: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print("=" * 70)


def evaluate(
    cfg: ProjectConfig,
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    test_file: Optional[str] = None,
    output_file: Optional[str] = None,
    device: str = "auto",
) -> dict:
    """
    执行模型评估

    Args:
        cfg: 项目配置
        model_path: 模型路径
        adapter_path: LoRA 适配器路径
        test_file: 测试文件路径
        output_file: 输出文件路径
        device: 推理设备

    Returns:
        评估指标字典
    """
    _model_path = model_path or cfg.export.export_dir
    _test_file = test_file or cfg.data.test_file

    if adapter_path:
        _model_path = model_path or cfg.model.model_name_or_path

    # 加载模型
    classifier = IntentClassifier(
        model_path=_model_path,
        adapter_path=adapter_path,
        trust_remote_code=cfg.model.trust_remote_code,
        device=device,
    )

    # 加载测试数据
    with open(_test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    instruction = cfg.data.get_instruction()

    # 执行推理
    y_true = []
    y_pred = []
    details = []

    for item in test_data:
        text = item["input"]
        expected = item["output"]

        predicted = classifier.predict(
            text,
            instruction,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
            top_k=cfg.inference.top_k,
            max_new_tokens=cfg.inference.max_new_tokens,
        )

        y_true.append(expected)
        y_pred.append(predicted.strip())
        details.append({
            "input": text,
            "expected": expected,
            "predicted": predicted.strip(),
            "correct": predicted.strip() == expected.strip(),
        })

        status = "✓" if predicted.strip() == expected.strip() else "✗"
        logger.info(f"[{status}] {text} → 预测: {predicted}, 期望: {expected}")

    # 计算指标
    metrics = compute_metrics(y_true, y_pred, cfg.data.labels)
    print_classification_report(metrics, cfg.data.labels)

    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {"metrics": metrics, "details": details}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {output_file}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="意图分类模型评估脚本")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA 适配器路径")
    parser.add_argument("--test_file", type=str, default=None, help="测试集路径")
    parser.add_argument("--output_file", type=str, default=None, help="输出结果路径")
    parser.add_argument("--device", type=str, default="auto", help="推理设备")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        cfg = ProjectConfig.load(args.config)
    else:
        cfg = ProjectConfig()

    evaluate(
        cfg=cfg,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        test_file=args.test_file,
        output_file=args.output_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
