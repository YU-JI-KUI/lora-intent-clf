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

# ─── 4. 可选参数默认值（从 PROJECT_ROOT 派生）────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/saves/qwen3-8b/lora/sft}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/data}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed/ds_z3_config.json}"
EXPORT_DIR="${EXPORT_DIR:-${PROJECT_ROOT}/models/qwen3-8b-intent-clf}"

# 预测输出目录：与 OUTPUT_DIR 同级的 predict/ 子目录
PREDICT_OUTPUT_DIR="$(dirname "${OUTPUT_DIR}")/predict"

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
       EXPORT_DIR PREDICT_OUTPUT_DIR NPROC_PER_NODE PYTHON TORCHRUN
