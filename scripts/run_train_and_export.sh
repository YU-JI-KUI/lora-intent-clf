#!/usr/bin/env bash
# =============================================================================
# 一键执行 LoRA SFT 微调 + Adapter 导出
# 兼容 LlamaFactory v0.9.4.dev0
#
# 用法: bash scripts/run_train_and_export.sh [--skip-train] [--skip-export]
#
# 机器特定参数（MODEL_PATH / OUTPUT_DIR / DATASET_DIR / DEEPSPEED_CONFIG /
# EXPORT_DIR / NPROC_PER_NODE）从 machine.env 自动加载，无需手动修改本脚本。
# =============================================================================

set -euo pipefail

# 加载公共配置（machine.env + PROJECT_ROOT 推导 + venv 路径）
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
SKIP_TRAIN=false
SKIP_EXPORT=false

for arg in "$@"; do
    case $arg in
        --skip-train)  SKIP_TRAIN=true ;;
        --skip-export) SKIP_EXPORT=true ;;
        --help|-h)
            echo "用法: bash scripts/run_train_and_export.sh [选项]"
            echo "选项:"
            echo "  --skip-train   跳过训练步骤（仅执行导出）"
            echo "  --skip-export  跳过导出步骤（仅执行训练）"
            echo ""
            echo "当前配置（来自 machine.env）："
            echo "  MODEL_PATH       = ${MODEL_PATH}"
            echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
            echo "  DATASET_DIR      = ${DATASET_DIR}"
            echo "  DEEPSPEED_CONFIG = ${DEEPSPEED_CONFIG}"
            echo "  EXPORT_DIR       = ${EXPORT_DIR}"
            echo "  NPROC_PER_NODE   = ${NPROC_PER_NODE}"
            exit 0 ;;
        *) error "未知参数: $arg，使用 --help 查看可用选项" ;;
    esac
done

# protobuf >= 3.20 与 TensorBoard 2.9.0 不兼容，强制使用纯 Python 实现
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

TRAIN_CONFIG="${PROJECT_ROOT}/configs/train_lora_sft.yaml"
EXPORT_CONFIG="${PROJECT_ROOT}/configs/export_lora.yaml"

# ─── 环境检查 ─────────────────────────────────────────────────────────────────
info "=========================================="
info "机器配置（来自 machine.env）"
info "  MODEL_PATH       = ${MODEL_PATH}"
info "  OUTPUT_DIR       = ${OUTPUT_DIR}"
info "  DATASET_DIR      = ${DATASET_DIR}"
info "  DEEPSPEED_CONFIG = ${DEEPSPEED_CONFIG}"
info "  EXPORT_DIR       = ${EXPORT_DIR}"
info "  NPROC_PER_NODE   = ${NPROC_PER_NODE}"
info "=========================================="

[[ -f "${TRAIN_CONFIG}" ]]                  || error "训练配置不存在: ${TRAIN_CONFIG}"
[[ -f "${EXPORT_CONFIG}" ]]                 || error "导出配置不存在: ${EXPORT_CONFIG}"
[[ -f "${DATASET_DIR}/dataset_info.json" ]] || error "数据集配置不存在: ${DATASET_DIR}/dataset_info.json"
[[ -f "${DEEPSPEED_CONFIG}" ]]              || error "DeepSpeed 配置不存在: ${DEEPSPEED_CONFIG}"
command -v llamafactory-cli &>/dev/null     || error "未找到 llamafactory-cli"

if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    info "检测到 ${GPU_COUNT} 块 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# ─── Step 1: 训练 ─────────────────────────────────────────────────────────────
if [[ "${SKIP_TRAIN}" == "false" ]]; then
    info "Step 1/2: 开始 LoRA SFT 微调 | 配置: ${TRAIN_CONFIG}"

    # 机器特定参数通过 CLI 覆盖 YAML 中的静态值（CLI 优先级 > YAML）
    FORCE_TORCHRUN=1 llamafactory-cli train "${TRAIN_CONFIG}" \
        --model_name_or_path  "${MODEL_PATH}" \
        --output_dir          "${OUTPUT_DIR}" \
        --dataset_dir         "${DATASET_DIR}" \
        --deepspeed           "${DEEPSPEED_CONFIG}"

    info "训练完成！"
else
    warn "跳过训练步骤（--skip-train）"
fi

# ─── Step 2: 导出 ─────────────────────────────────────────────────────────────
if [[ "${SKIP_EXPORT}" == "false" ]]; then
    info "Step 2/2: 导出合并模型 | 配置: ${EXPORT_CONFIG}"

    # adapter_name_or_path = OUTPUT_DIR（训练产物就是 adapter）
    llamafactory-cli export "${EXPORT_CONFIG}" \
        --model_name_or_path   "${MODEL_PATH}" \
        --adapter_name_or_path "${OUTPUT_DIR}" \
        --export_dir           "${EXPORT_DIR}"

    info "模型导出完成！"
else
    warn "跳过导出步骤（--skip-export）"
fi

info "=========================================="
info "全部流程完成！"
info "  训练输出(adapter): ${OUTPUT_DIR}"
info "  合并模型:          ${EXPORT_DIR}"
info "  TensorBoard:       bash scripts/run_tensorboard.sh"
info "  推理测试:          bash scripts/run_inference.sh --lora"
info "=========================================="
