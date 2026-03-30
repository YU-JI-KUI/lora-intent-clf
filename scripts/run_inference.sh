#!/usr/bin/env bash
# =============================================================================
# 推理测试脚本 — 启动交互式聊天测试微调效果
# 兼容 LlamaFactory v0.9.4.dev0
#
# 用法: bash scripts/run_inference.sh [--lora]
#
# 机器特定参数从 machine.env 自动加载，无需手动修改本脚本。
# =============================================================================

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

USE_LORA=false

for arg in "$@"; do
    case $arg in
        --lora) USE_LORA=true ;;
        --help|-h)
            echo "用法: bash scripts/run_inference.sh [选项]"
            echo "选项:"
            echo "  --lora  使用基座模型 + LoRA 适配器（不需要先 export）"
            echo "  默认    使用导出后的合并模型"
            exit 0 ;;
        *) error "未知参数: $arg" ;;
    esac
done

INFERENCE_CONFIG="${PROJECT_ROOT}/configs/inference.yaml"
INFERENCE_LORA_CONFIG="${PROJECT_ROOT}/configs/inference_lora.yaml"

if [[ "${USE_LORA}" == "true" ]]; then
    CONFIG="${INFERENCE_LORA_CONFIG}"
    info "使用模式: 基座模型 + LoRA 适配器"
    info "  基座模型: ${MODEL_PATH}"
    info "  Adapter:  ${OUTPUT_DIR}"

    [[ -f "${CONFIG}" ]] || error "配置文件不存在: ${CONFIG}"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli"

    # 机器特定参数通过 CLI 覆盖 YAML
    llamafactory-cli chat "${CONFIG}" \
        --model_name_or_path   "${MODEL_PATH}" \
        --adapter_name_or_path "${OUTPUT_DIR}"
else
    CONFIG="${INFERENCE_CONFIG}"
    info "使用模式: 导出后的合并模型"
    info "  模型路径: ${EXPORT_DIR}"

    [[ -f "${CONFIG}" ]] || error "配置文件不存在: ${CONFIG}"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli"

    llamafactory-cli chat "${CONFIG}" \
        --model_name_or_path "${EXPORT_DIR}"
fi
