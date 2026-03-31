#!/usr/bin/env bash
# =============================================================================
# common.sh — 公共配置加载器（所有脚本 source 此文件）
#
# 功能：
#   1. 动态推导 PROJECT_ROOT（无需硬编码路径）
#   2. 加载 machine.env（机器特定配置，gitignored）
#   3. 为未设置的可选参数提供基于 PROJECT_ROOT 的默认值
#   4. 导出 PYTHON / TORCHRUN 指向 .venv（保证使用正确的虚拟环境）
#
# 使用方式（在其他脚本中的第一行）：
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# 加载后可用变量：
#   $PROJECT_ROOT       项目根目录（自动推导）
#   $MODEL_PATH         基座模型路径
#   $OUTPUT_DIR         训练输出目录（LoRA adapter）
#   $DATASET_DIR        数据集目录（含 dataset_info.json）
#   $DEEPSPEED_CONFIG   DeepSpeed 配置文件路径
#   $EXPORT_DIR         模型导出目录
#   $PREDICT_OUTPUT_DIR 批量预测输出目录（= OUTPUT_DIR/../predict）
#   $NPROC_PER_NODE     GPU 数量（用于 torchrun）
#   $PYTHON             .venv/bin/python（或系统 python）
#   $TORCHRUN           .venv/bin/torchrun（或系统 torchrun）
# =============================================================================

# ─── 1. 动态推导 PROJECT_ROOT ─────────────────────────────────────────────────
# BASH_SOURCE[0] 是 common.sh 本身（scripts/ 目录内），
# PROJECT_ROOT 是其上级目录。
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_COMMON_DIR}/.." && pwd)"

# ─── 2. 加载 machine.env ──────────────────────────────────────────────────────
_ENV_FILE="${PROJECT_ROOT}/machine.env"
_ENV_EXAMPLE="${PROJECT_ROOT}/machine.env.example"

if [[ -f "${_ENV_FILE}" ]]; then
    # shellcheck source=/dev/null
    set -a  # 自动 export 所有变量
    source "${_ENV_FILE}"
    set +a
elif [[ -f "${_ENV_EXAMPLE}" ]]; then
    echo -e "\033[1;33m[WARN]\033[0m machine.env 不存在，使用 machine.env.example 中的默认值"
    echo -e "\033[1;33m[WARN]\033[0m 建议: cp ${PROJECT_ROOT}/machine.env.example ${PROJECT_ROOT}/machine.env && vim ${PROJECT_ROOT}/machine.env"
    set -a
    source "${_ENV_EXAMPLE}"
    set +a
else
    echo -e "\033[0;31m[ERROR]\033[0m 找不到 machine.env 或 machine.env.example" >&2
    exit 1
fi

# ─── 3. 必需参数校验 ──────────────────────────────────────────────────────────
_missing=()
[[ -z "${MODEL_PATH:-}" ]]      && _missing+=("MODEL_PATH")
[[ -z "${NPROC_PER_NODE:-}" ]]  && _missing+=("NPROC_PER_NODE")

if [[ ${#_missing[@]} -gt 0 ]]; then
    echo -e "\033[0;31m[ERROR]\033[0m machine.env 中未设置以下必需参数: ${_missing[*]}" >&2
    echo -e "\033[0;31m[ERROR]\033[0m 请编辑 ${PROJECT_ROOT}/machine.env" >&2
    exit 1
fi

# ─── 4. 路径参数默认值（从 PROJECT_ROOT 派生）────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saves/qwen3-8b/lora/sft}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/ds_z3_offload_config.json}"
EXPORT_DIR="${EXPORT_DIR:-${PROJECT_ROOT}/models/qwen3-8b-intent-clf}"

# 预测输出目录：与 OUTPUT_DIR 同级的 predict/ 子目录
PREDICT_OUTPUT_DIR="$(dirname "${OUTPUT_DIR}")/predict"

# ─── 4b. LoRA 超参数默认值 ────────────────────────────────────────────────────
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET="${LORA_TARGET:-all}"

# ─── 4c. 训练超参数默认值 ─────────────────────────────────────────────────────
CUTOFF_LEN="${CUTOFF_LEN:-256}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# ─── 4d. 监控 & 早停默认值 ───────────────────────────────────────────────────
# ⚠️ SAVE_STEPS 必须等于 EVAL_STEPS，早停依赖每次评估后的 checkpoint
LOGGING_STEPS="${LOGGING_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-200}"
SAVE_STEPS="${SAVE_STEPS:-${EVAL_STEPS}}"  # 默认与 EVAL_STEPS 保持一致
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-1e-4}"

# ─── 5. 虚拟环境中的 Python / torchrun ───────────────────────────────────────
_VENV="${PROJECT_ROOT}/.venv"
if [[ -f "${_VENV}/bin/python" ]]; then
    PYTHON="${_VENV}/bin/python"
    TORCHRUN="${_VENV}/bin/torchrun"
else
    # 回退到系统环境（uv venv 未初始化时）
    PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo 'python')"
    TORCHRUN="$(command -v torchrun 2>/dev/null || echo 'torchrun')"
fi

# ─── 6. 统一导出 ──────────────────────────────────────────────────────────────
export PROJECT_ROOT MODEL_PATH OUTPUT_DIR DATASET_DIR DEEPSPEED_CONFIG \
       EXPORT_DIR PREDICT_OUTPUT_DIR NPROC_PER_NODE PYTHON TORCHRUN \
       LORA_RANK LORA_ALPHA LORA_DROPOUT LORA_TARGET \
       CUTOFF_LEN PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS \
       LEARNING_RATE NUM_TRAIN_EPOCHS WARMUP_RATIO WEIGHT_DECAY MAX_GRAD_NORM \
       LOGGING_STEPS EVAL_STEPS SAVE_STEPS \
       EARLY_STOPPING_PATIENCE EARLY_STOPPING_THRESHOLD

# ─── 7. 启动时打印所有已解析参数（方便排查配置是否生效）────────────────────
_CYAN='\033[0;36m'; _GREEN='\033[0;32m'; _YELLOW='\033[1;33m'; _NC='\033[0m'
echo -e "${_CYAN}══════════════════════════════════════════════════════════════${_NC}"
echo -e "${_CYAN}  [CONFIG] 已加载的配置（来源: ${_ENV_FILE:-${_ENV_EXAMPLE}}）${_NC}"
echo -e "${_CYAN}══════════════════════════════════════════════════════════════${_NC}"
echo -e "  ${_GREEN}--- 机器 ---${_NC}"
echo -e "  ${_GREEN}PROJECT_ROOT               ${_NC}= ${PROJECT_ROOT}"
echo -e "  ${_GREEN}MODEL_PATH                 ${_NC}= ${MODEL_PATH}"
echo -e "  ${_GREEN}NPROC_PER_NODE             ${_NC}= ${NPROC_PER_NODE}"
echo -e "  ${_GREEN}DEEPSPEED_CONFIG           ${_NC}= ${DEEPSPEED_CONFIG}"
echo -e "  ${_GREEN}OUTPUT_DIR                 ${_NC}= ${OUTPUT_DIR}"
echo -e "  ${_GREEN}DATASET_DIR                ${_NC}= ${DATASET_DIR}"
echo -e "  ${_GREEN}EXPORT_DIR                 ${_NC}= ${EXPORT_DIR}"
echo -e "  ${_GREEN}PYTHON                     ${_NC}= ${PYTHON}"
echo -e "  ${_GREEN}TORCHRUN                   ${_NC}= ${TORCHRUN}"
echo -e "  ${_GREEN}--- LoRA 超参 ---${_NC}"
echo -e "  ${_GREEN}LORA_RANK                  ${_NC}= ${LORA_RANK}"
echo -e "  ${_GREEN}LORA_ALPHA                 ${_NC}= ${LORA_ALPHA}"
echo -e "  ${_GREEN}LORA_DROPOUT               ${_NC}= ${LORA_DROPOUT}"
echo -e "  ${_GREEN}LORA_TARGET                ${_NC}= ${LORA_TARGET}"
echo -e "  ${_GREEN}--- 训练超参 ---${_NC}"
echo -e "  ${_GREEN}CUTOFF_LEN                 ${_NC}= ${CUTOFF_LEN}"
echo -e "  ${_GREEN}PER_DEVICE_TRAIN_BATCH_SIZE${_NC}= ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo -e "  ${_GREEN}GRADIENT_ACCUMULATION_STEPS${_NC}= ${GRADIENT_ACCUMULATION_STEPS}"
echo -e "  ${_GREEN}LEARNING_RATE              ${_NC}= ${LEARNING_RATE}"
echo -e "  ${_GREEN}NUM_TRAIN_EPOCHS           ${_NC}= ${NUM_TRAIN_EPOCHS}"
echo -e "  ${_GREEN}WARMUP_RATIO               ${_NC}= ${WARMUP_RATIO}"
echo -e "  ${_GREEN}WEIGHT_DECAY               ${_NC}= ${WEIGHT_DECAY}"
echo -e "  ${_GREEN}MAX_GRAD_NORM              ${_NC}= ${MAX_GRAD_NORM}"
echo -e "  ${_GREEN}--- 监控 & 早停 ---${_NC}"
echo -e "  ${_GREEN}LOGGING_STEPS              ${_NC}= ${LOGGING_STEPS}"
echo -e "  ${_GREEN}EVAL_STEPS                 ${_NC}= ${EVAL_STEPS}"
echo -e "  ${_GREEN}SAVE_STEPS                 ${_NC}= ${SAVE_STEPS}"
echo -e "  ${_GREEN}EARLY_STOPPING_PATIENCE    ${_NC}= ${EARLY_STOPPING_PATIENCE}"
echo -e "  ${_GREEN}EARLY_STOPPING_THRESHOLD   ${_NC}= ${EARLY_STOPPING_THRESHOLD}"

# 校验关键文件/路径是否存在
_warn_missing() {
    [[ -e "$2" ]] || echo -e "  ${_YELLOW}[WARN] ${1} 路径不存在: $2${_NC}"
}
_warn_missing "MODEL_PATH"       "${MODEL_PATH}"
_warn_missing "DEEPSPEED_CONFIG" "${DEEPSPEED_CONFIG}"
_warn_missing "DATASET_DIR"      "${DATASET_DIR}"

echo -e "${_CYAN}══════════════════════════════════════════════════════════════${_NC}"
