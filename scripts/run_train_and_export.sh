#!/usr/bin/env bash
# =============================================================================
# 一键执行 LoRA SFT 微调 + Adapter 导出
# 兼容 LlamaFactory v0.9.4.dev0
# 工作目录：/workspace/lora-intent-clf
# 用法: bash scripts/run_train_and_export.sh [--skip-train] [--skip-export]
# 注意: 所有配置使用绝对路径，可以在任意目录下执行本脚本
# =============================================================================

set -euo pipefail

# ------------------------------- 颜色输出 ------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ------------------------------- 参数解析 ------------------------------------
SKIP_TRAIN=false
SKIP_EXPORT=false

for arg in "$@"; do
    case $arg in
        --skip-train)  SKIP_TRAIN=true ;;
        --skip-export) SKIP_EXPORT=true ;;
        --help|-h)
            echo "用法: bash scripts/run_train_and_export.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-train   跳过训练步骤（仅执行导出）"
            echo "  --skip-export  跳过导出步骤（仅执行训练）"
            echo "  --help, -h     显示帮助信息"
            exit 0
            ;;
        *) error "未知参数: $arg，使用 --help 查看可用选项" ;;
    esac
done

# ------------------------------- 环境变量 ------------------------------------
# protobuf >= 3.20 引入破坏性 API 变更，与 TensorBoard 2.9.0 不兼容，会报：
#   TypeError: Descriptors cannot be created directly.
# 强制使用纯 Python 实现作为双保险（根本解法是 pyproject.toml 锁定 protobuf<3.20）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ------------------------------- 配置路径（绝对路径）--------------------------
PROJECT_ROOT="/workspace/lora-intent-clf"
TRAIN_CONFIG="${PROJECT_ROOT}/configs/train_lora_sft.yaml"
EXPORT_CONFIG="${PROJECT_ROOT}/configs/export_lora.yaml"

# ------------------------------- 环境检查 ------------------------------------
info "项目根目录: ${PROJECT_ROOT}"

# 检查项目目录是否存在
[[ -d "${PROJECT_ROOT}" ]] || error "项目目录不存在: ${PROJECT_ROOT}"

# 检查 llamafactory-cli 是否可用
if ! command -v llamafactory-cli &> /dev/null; then
    error "未找到 llamafactory-cli，请先安装 LlamaFactory v0.9.4.dev0"
fi

info "LlamaFactory CLI 已就绪"

# 检查配置文件
[[ -f "${TRAIN_CONFIG}" ]] || error "训练配置文件不存在: ${TRAIN_CONFIG}"
[[ -f "${EXPORT_CONFIG}" ]] || error "导出配置文件不存在: ${EXPORT_CONFIG}"

# 检查 dataset_info.json
[[ -f "${PROJECT_ROOT}/data/dataset_info.json" ]] || error "数据集配置不存在: ${PROJECT_ROOT}/data/dataset_info.json"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    info "检测到 ${GPU_COUNT} 块 GPU"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    warn "未检测到 nvidia-smi，将使用 CPU 训练（非常慢）"
fi

# ------------------------------- Step 1: 训练 --------------------------------
if [[ "${SKIP_TRAIN}" == "false" ]]; then
    info "=========================================="
    info "Step 1/2: 开始 LoRA SFT 微调"
    info "配置文件: ${TRAIN_CONFIG}"
    info "=========================================="

    # 使用 llamafactory-cli train 命令（兼容 v0.9.4.dev0）
    # 注意：YAML 中所有路径已使用绝对路径，无需 cd 到项目目录
    # FORCE_TORCHRUN=1 确保使用 torchrun 启动多 GPU 训练（配合 DeepSpeed ZeRO-3）
    FORCE_TORCHRUN=1 llamafactory-cli train "${TRAIN_CONFIG}"

    if [[ $? -eq 0 ]]; then
        info "训练完成！"
    else
        error "训练失败，请检查日志"
    fi
else
    warn "跳过训练步骤（--skip-train）"
fi

# ------------------------------- Step 2: 导出 --------------------------------
if [[ "${SKIP_EXPORT}" == "false" ]]; then
    info "=========================================="
    info "Step 2/2: 导出合并模型"
    info "配置文件: ${EXPORT_CONFIG}"
    info "=========================================="

    # 使用 llamafactory-cli export 命令（兼容 v0.9.4.dev0）
    llamafactory-cli export "${EXPORT_CONFIG}"

    if [[ $? -eq 0 ]]; then
        info "模型导出完成！"
        info "合并模型保存在: ${PROJECT_ROOT}/models/qwen3-8b-intent-clf"
    else
        error "模型导出失败，请检查日志"
    fi
else
    warn "跳过导出步骤（--skip-export）"
fi

# ------------------------------- 完成提示 ------------------------------------
info "=========================================="
info "全部流程完成！"
info ""
info "训练输出:   ${PROJECT_ROOT}/saves/qwen3-8b/lora/sft"
info "合并模型:   ${PROJECT_ROOT}/models/qwen3-8b-intent-clf"
info ""
info "后续操作:"
info "  1. 查看 TensorBoard:  tensorboard --logdir ${PROJECT_ROOT}/saves/qwen3-8b/lora/sft"
info "  2. 使用合并模型推理:  llamafactory-cli chat ${PROJECT_ROOT}/configs/inference.yaml"
info "  3. 使用 LoRA 推理:    llamafactory-cli chat ${PROJECT_ROOT}/configs/inference_lora.yaml"
info "=========================================="
