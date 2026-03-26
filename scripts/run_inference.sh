#!/usr/bin/env bash
# =============================================================================
# 推理测试脚本 — 启动交互式聊天测试微调效果
# 兼容 LlamaFactory v0.9.0
# 用法: bash scripts/run_inference.sh [--lora]
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ------------------------------- 参数解析 ------------------------------------
USE_LORA=false

for arg in "$@"; do
    case $arg in
        --lora) USE_LORA=true ;;
        --help|-h)
            echo "用法: bash scripts/run_inference.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --lora      使用基座模型 + LoRA 适配器推理（不需要先 export）"
            echo "  默认        使用导出后的合并模型推理"
            exit 0
            ;;
        *) error "未知参数: $arg" ;;
    esac
done

# ------------------------------- 配置路径 ------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${USE_LORA}" == "true" ]]; then
    CONFIG="${PROJECT_ROOT}/configs/inference_lora.yaml"
    info "使用模式: 基座模型 + LoRA 适配器"
else
    CONFIG="${PROJECT_ROOT}/configs/inference.yaml"
    info "使用模式: 导出后的合并模型"
fi

[[ -f "${CONFIG}" ]] || error "配置文件不存在: ${CONFIG}"

# ------------------------------- 启动推理 ------------------------------------
info "启动交互式推理..."
info "配置文件: ${CONFIG}"
info "输入文本后按 Enter 进行推理，输入 exit 退出"
info "=========================================="

cd "${PROJECT_ROOT}"
llamafactory-cli chat "${CONFIG}"
