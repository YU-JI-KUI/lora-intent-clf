#!/usr/bin/env bash
# =============================================================================
# 批量评估脚本 — 在测试集上评估微调效果，输出分类报告
# 兼容 LlamaFactory v0.9.4.dev0
#
# 用法:
#   bash scripts/run_evaluate.sh              # 模式A：LlamaFactory 批量预测 + 解析
#   bash scripts/run_evaluate.sh --python     # 模式B：纯 Python 推理评估
#
# 机器特定参数从 machine.env 自动加载，无需手动修改本脚本。
# =============================================================================

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

USE_PYTHON=false

for arg in "$@"; do
    case $arg in
        --python) USE_PYTHON=true ;;
        --help|-h)
            echo "用法: bash scripts/run_evaluate.sh [选项]"
            echo "选项:"
            echo "  --python    使用 Python 脚本直接推理评估（慢，单条）"
            echo "  默认        使用 LlamaFactory 批量预测 + Python 解析（推荐）"
            exit 0 ;;
        *) error "未知参数: $arg，使用 --help 查看可用选项" ;;
    esac
done

PREDICT_CONFIG="${PROJECT_ROOT}/configs/predict_lora.yaml"
PRED_FILE="${PREDICT_OUTPUT_DIR}/generated_predictions.jsonl"
EVAL_OUTPUT="${PREDICT_OUTPUT_DIR}/eval_report.json"

# ─── 模式A：LlamaFactory 批量预测 ────────────────────────────────────────────
if [[ "${USE_PYTHON}" == "false" ]]; then
    info "模式A：LlamaFactory 批量预测 + Python 评估"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli，请使用 --python 模式"
    [[ -f "${PREDICT_CONFIG}" ]] || error "预测配置不存在: ${PREDICT_CONFIG}"

    info "Step 1/2: LlamaFactory 批量预测"
    info "  基座模型: ${MODEL_PATH}"
    info "  Adapter:  ${OUTPUT_DIR}"
    info "  数据集:   ${DATASET_DIR}"
    info "  输出目录: ${PREDICT_OUTPUT_DIR}"

    llamafactory-cli train "${PREDICT_CONFIG}" \
        --model_name_or_path   "${MODEL_PATH}" \
        --adapter_name_or_path "${OUTPUT_DIR}" \
        --dataset_dir          "${DATASET_DIR}" \
        --output_dir           "${PREDICT_OUTPUT_DIR}"

    [[ -f "${PRED_FILE}" ]] || error "预测文件未生成: ${PRED_FILE}"

    info "Step 2/2: 解析预测结果"
    cd "${PROJECT_ROOT}"
    "${PYTHON}" src/evaluate.py \
        --pred_file   "${PRED_FILE}" \
        --output_file "${EVAL_OUTPUT}"

    info "评估完成！报告: ${EVAL_OUTPUT}"

# ─── 模式B：纯 Python 推理评估 ───────────────────────────────────────────────
else
    info "模式B：Python 脚本直接推理评估"
    TEST_FILE="${DATASET_DIR}/test.json"
    EVAL_OUTPUT_PY="${OUTPUT_DIR}/eval_report.json"

    [[ -d "${OUTPUT_DIR}" ]] || error "LoRA 适配器不存在: ${OUTPUT_DIR}"
    [[ -f "${TEST_FILE}" ]]  || error "测试文件不存在: ${TEST_FILE}"

    info "  基座模型: ${MODEL_PATH}"
    info "  Adapter:  ${OUTPUT_DIR}"
    info "  测试文件: ${TEST_FILE}"

    cd "${PROJECT_ROOT}"
    "${PYTHON}" src/evaluate.py \
        --adapter_path "${OUTPUT_DIR}" \
        --test_file    "${TEST_FILE}" \
        --output_file  "${EVAL_OUTPUT_PY}"

    info "评估完成！报告: ${EVAL_OUTPUT_PY}"
fi
