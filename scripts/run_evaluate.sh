#!/usr/bin/env bash
# =============================================================================
# 批量评估脚本 — 在测试集上评估微调效果，输出分类报告
# 兼容 LlamaFactory v0.9.4.dev0
# 工作目录：/workspace/lora-intent-clf
#
# 用法:
#   bash scripts/run_evaluate.sh              # 模式A：LlamaFactory 批量预测 + 解析
#   bash scripts/run_evaluate.sh --python     # 模式B：纯 Python 推理评估
#
# 两种模式说明：
#   模式A（默认，推荐）：
#     1. llamafactory-cli 在测试集上批量推理，生成 generated_predictions.jsonl
#     2. python src/evaluate.py 解析预测文件，计算 accuracy/F1/分类报告
#     优点：速度快（8条/batch），LlamaFactory 负责推理，evaluate.py 只做统计
#
#   模式B（--python）：
#     python src/evaluate.py 直接加载模型推理评估（单条，速度慢）
#     适用于：LlamaFactory 未安装，或需要查看逐条预测详情
# =============================================================================

set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ------------------------------- 参数解析 ------------------------------------
USE_PYTHON=false

for arg in "$@"; do
    case $arg in
        --python) USE_PYTHON=true ;;
        --help|-h)
            echo "用法: bash scripts/run_evaluate.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --python    使用 Python 脚本直接推理评估（慢，单条）"
            echo "  默认        使用 LlamaFactory 批量预测 + Python 解析（推荐）"
            exit 0
            ;;
        *) error "未知参数: $arg，使用 --help 查看可用选项" ;;
    esac
done

# ------------------------------- 配置路径（绝对路径）--------------------------
PROJECT_ROOT="/workspace/lora-intent-clf"
PREDICT_CONFIG="${PROJECT_ROOT}/configs/predict_lora.yaml"
PREDICT_OUTPUT_DIR="${PROJECT_ROOT}/saves/qwen3-8b/lora/predict"
PRED_FILE="${PREDICT_OUTPUT_DIR}/generated_predictions.jsonl"
EVAL_OUTPUT="${PROJECT_ROOT}/saves/qwen3-8b/lora/predict/eval_report.json"

[[ -d "${PROJECT_ROOT}" ]] || error "项目目录不存在: ${PROJECT_ROOT}"

# ------------------------------- 模式A：LlamaFactory 批量预测 ----------------
if [[ "${USE_PYTHON}" == "false" ]]; then
    info "=========================================="
    info "模式A：LlamaFactory 批量预测 + Python 评估"
    info "=========================================="

    if ! command -v llamafactory-cli &> /dev/null; then
        error "未找到 llamafactory-cli，请使用 --python 模式或先安装 LlamaFactory"
    fi

    [[ -f "${PREDICT_CONFIG}" ]] || error "预测配置文件不存在: ${PREDICT_CONFIG}"

    info "Step 1/2: 运行 LlamaFactory 批量预测..."
    info "配置文件: ${PREDICT_CONFIG}"
    info "输出目录: ${PREDICT_OUTPUT_DIR}"

    # 使用 train 子命令运行 predict（stage: sft, do_predict: true）
    llamafactory-cli train "${PREDICT_CONFIG}"

    [[ -f "${PRED_FILE}" ]] || error "预测文件未生成: ${PRED_FILE}"

    info "Step 2/2: 解析预测结果，生成分类报告..."
    cd "${PROJECT_ROOT}"
    python src/evaluate.py \
        --pred_file "${PRED_FILE}" \
        --output_file "${EVAL_OUTPUT}"

    info "=========================================="
    info "评估完成！"
    info "预测文件: ${PRED_FILE}"
    info "评估报告: ${EVAL_OUTPUT}"
    info "=========================================="

# ------------------------------- 模式B：纯 Python 推理评估 -------------------
else
    info "=========================================="
    info "模式B：Python 脚本直接推理评估"
    info "=========================================="

    ADAPTER_PATH="${PROJECT_ROOT}/saves/qwen3-8b/lora/sft"
    TEST_FILE="${PROJECT_ROOT}/data/test.json"
    EVAL_OUTPUT_PY="${PROJECT_ROOT}/saves/qwen3-8b/lora/sft/eval_report.json"

    [[ -d "${ADAPTER_PATH}" ]] || error "LoRA 适配器不存在: ${ADAPTER_PATH}"
    [[ -f "${TEST_FILE}" ]] || error "测试文件不存在: ${TEST_FILE}"

    info "适配器路径: ${ADAPTER_PATH}"
    info "测试文件: ${TEST_FILE}"

    cd "${PROJECT_ROOT}"
    python src/evaluate.py \
        --adapter_path "${ADAPTER_PATH}" \
        --test_file "${TEST_FILE}" \
        --output_file "${EVAL_OUTPUT_PY}"

    info "=========================================="
    info "评估完成！报告保存在: ${EVAL_OUTPUT_PY}"
    info "=========================================="
fi
