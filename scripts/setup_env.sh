#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 跨机器统一环境初始化脚本
#
# 功能：
#   1. 检测/安装 uv
#   2. 创建虚拟环境（.venv）
#   3. 从 Nexus 内部仓库安装依赖
#   4. 安装 LlamaFactory（可选）
#
# 用法：
#   bash scripts/setup_env.sh              # 仅安装生产依赖
#   bash scripts/setup_env.sh --dev        # 同时安装开发依赖
#   bash scripts/setup_env.sh --llamafactory  # 同时安装 LlamaFactory
#   bash scripts/setup_env.sh --dev --llamafactory
#
# 注意：
#   - 内网仓库地址：http://maven.paic.com.cn/repository/pypi/simple
#   - 脚本幂等，重复执行安全
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
NEXUS_URL="http://maven.paic.com.cn/repository/pypi/simple"

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
INSTALL_DEV=false
INSTALL_LLAMAFACTORY=false
for arg in "$@"; do
    case "$arg" in
        --dev)             INSTALL_DEV=true ;;
        --llamafactory)    INSTALL_LLAMAFACTORY=true ;;
        --help|-h)
            grep '^#' "$0" | head -20 | sed 's/^# \{0,1\}//'
            exit 0 ;;
    esac
done

# ─── 颜色输出 ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

cd "${PROJECT_DIR}"

# ─── 1. 安装 uv ───────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    warn "uv 未安装，尝试自动安装..."

    # 方式 A：官方安装脚本（需公网，内网环境可能不可用）
    if curl -fsSL https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        export PATH="${HOME}/.local/bin:${PATH}"
        info "uv 安装成功（官方脚本）"
    # 方式 B：通过内部 pip 安装
    elif pip install uv --index-url "${NEXUS_URL}" --trusted-host maven.paic.com.cn 2>/dev/null; then
        info "uv 安装成功（内部 Nexus）"
    else
        error "uv 安装失败，请手动安装后重试"
        error "  curl -fsSL https://astral.sh/uv/install.sh | sh"
        error "  或: pip install uv --index-url ${NEXUS_URL}"
        exit 1
    fi
else
    info "uv 已安装：$(uv --version)"
fi

# 确保 uv 在 PATH 中
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"

# ─── 2. 创建虚拟环境 ──────────────────────────────────────────────────────────
if [[ ! -d "${VENV_DIR}" ]]; then
    info "创建虚拟环境：${VENV_DIR}"
    uv venv "${VENV_DIR}" --python python3
else
    info "虚拟环境已存在：${VENV_DIR}"
fi

# ─── 3. 安装项目依赖 ──────────────────────────────────────────────────────────
info "安装生产依赖（来源：Nexus 内部仓库）..."
uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${NEXUS_URL}" \
    --trusted-host maven.paic.com.cn \
    -e ".[dev]" 2>/dev/null || \
uv pip install \
    --python "${VENV_DIR}/bin/python" \
    --index-url "${NEXUS_URL}" \
    --trusted-host maven.paic.com.cn \
    -r requirements.txt

if [[ "${INSTALL_DEV}" == "true" ]]; then
    info "安装开发依赖..."
    uv pip install \
        --python "${VENV_DIR}/bin/python" \
        --index-url "${NEXUS_URL}" \
        --trusted-host maven.paic.com.cn \
        ruff==0.13.2 pytest==7.4.1
fi

# ─── 4. 安装 LlamaFactory ─────────────────────────────────────────────────────
if [[ "${INSTALL_LLAMAFACTORY}" == "true" ]]; then
    info "安装 LlamaFactory..."
    # LlamaFactory 通常需要从 GitHub 克隆安装，如果内部有镜像则替换 URL
    if [[ -d "/workspace/LLaMA-Factory" ]]; then
        uv pip install \
            --python "${VENV_DIR}/bin/python" \
            --index-url "${NEXUS_URL}" \
            --trusted-host maven.paic.com.cn \
            -e /workspace/LLaMA-Factory
    else
        warn "未找到 /workspace/LLaMA-Factory，跳过 LlamaFactory 安装"
        warn "请确保 LlamaFactory 已在机器上安装并在 PATH 中可用"
    fi
fi

# ─── 5. 完成 ──────────────────────────────────────────────────────────────────
echo ""
info "✅ 环境初始化完成"
info "激活虚拟环境："
echo "    source ${VENV_DIR}/bin/activate"
echo ""

# GPU 信息
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    info "检测到 ${GPU_COUNT}x ${GPU_NAME}"
    if [[ "${GPU_COUNT}" -ge 8 ]]; then
        info "V100×8 配置 → 建议使用 --nproc_per_node=8"
    elif [[ "${GPU_COUNT}" -ge 4 ]]; then
        info "V100×4 配置 → 建议使用 --nproc_per_node=4"
    fi
fi
