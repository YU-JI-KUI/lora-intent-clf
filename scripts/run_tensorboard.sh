#!/usr/bin/env bash
# =============================================================================
# TensorBoard 启动脚本
#
# 用法：
#   bash scripts/run_tensorboard.sh                 # 前台启动（端口 6006）
#   bash scripts/run_tensorboard.sh --port 6007     # 指定端口
#   bash scripts/run_tensorboard.sh --background    # 后台启动
#   bash scripts/run_tensorboard.sh --kill          # 终止后台 TensorBoard
#   bash scripts/run_tensorboard.sh --logdir /path  # 指定日志目录
#
# 访问地址：http://<服务器IP>:6006
# 注意：服务器防火墙需放行对应端口，或通过 SSH 隧道访问：
#   本地执行：ssh -L 6006:localhost:6006 user@server
# =============================================================================

set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ------------------------------- 默认配置 ------------------------------------
PROJECT_ROOT="/workspace/lora-intent-clf"
DEFAULT_LOGDIR="${PROJECT_ROOT}/saves/qwen3-8b/lora/sft"
DEFAULT_PORT=6006
PID_FILE="${PROJECT_ROOT}/logs/tensorboard.pid"

LOGDIR="${DEFAULT_LOGDIR}"
PORT="${DEFAULT_PORT}"
BACKGROUND=false

# ------------------------------- 参数解析 ------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)
            LOGDIR="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --background|-b)
            BACKGROUND=true; shift ;;
        --kill)
            if [[ -f "${PID_FILE}" ]]; then
                PID=$(cat "${PID_FILE}")
                if kill -0 "${PID}" 2>/dev/null; then
                    kill "${PID}" && rm -f "${PID_FILE}"
                    info "TensorBoard (PID=${PID}) 已终止"
                else
                    warn "进程 PID=${PID} 已结束"
                    rm -f "${PID_FILE}"
                fi
            else
                warn "未找到后台 TensorBoard 进程"
            fi
            exit 0 ;;
        --help|-h)
            echo "用法: bash scripts/run_tensorboard.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --logdir  DIR    日志目录（默认: ${DEFAULT_LOGDIR}）"
            echo "  --port    PORT   监听端口（默认: ${DEFAULT_PORT}）"
            echo "  --background,-b  后台启动"
            echo "  --kill           终止后台 TensorBoard"
            echo "  --help           显示帮助"
            echo ""
            echo "SSH 隧道访问（在本地机器执行）:"
            echo "  ssh -L ${DEFAULT_PORT}:localhost:${DEFAULT_PORT} user@<服务器IP>"
            exit 0 ;;
        *)
            error "未知参数: $1，使用 --help 查看帮助" ;;
    esac
done

# ------------------------------- 检查依赖 ------------------------------------
command -v tensorboard &>/dev/null || error "未找到 tensorboard，请先安装: pip install tensorboard==2.9.0"

# ------------------------------- 检查日志目录 --------------------------------
if [[ ! -d "${LOGDIR}" ]]; then
    warn "日志目录不存在: ${LOGDIR}"
    warn "训练完成后日志会自动创建，TensorBoard 可以提前启动等待"
fi

# 检查端口是否已被占用
if lsof -Pi ":${PORT}" -sTCP:LISTEN -t &>/dev/null 2>&1; then
    warn "端口 ${PORT} 已被占用，可能已有 TensorBoard 在运行"
    warn "可用 --port 指定其他端口，或先 --kill 再重启"
fi

mkdir -p "${PROJECT_ROOT}/logs"

# ------------------------------- 启动 TensorBoard ----------------------------
info "=========================================="
info "TensorBoard 启动"
info "日志目录: ${LOGDIR}"
info "访问地址: http://localhost:${PORT}"
info "=========================================="
info "SSH 隧道（本地执行）: ssh -L ${PORT}:localhost:${PORT} user@<服务器IP>"
info "=========================================="

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup tensorboard \
        --logdir "${LOGDIR}" \
        --port "${PORT}" \
        --bind_all \
        >> "${PROJECT_ROOT}/logs/tensorboard.log" 2>&1 &
    TB_PID=$!
    echo "${TB_PID}" > "${PID_FILE}"

    sleep 2
    if kill -0 "${TB_PID}" 2>/dev/null; then
        info "TensorBoard 已在后台启动，PID=${TB_PID}"
        info "终止命令: bash scripts/run_tensorboard.sh --kill"
        info "日志: ${PROJECT_ROOT}/logs/tensorboard.log"
    else
        error "TensorBoard 启动失败，查看日志: cat ${PROJECT_ROOT}/logs/tensorboard.log"
    fi
else
    # 前台运行，Ctrl+C 退出
    info "前台运行中，Ctrl+C 退出（训练不受影响）"
    tensorboard \
        --logdir "${LOGDIR}" \
        --port "${PORT}" \
        --bind_all
fi
