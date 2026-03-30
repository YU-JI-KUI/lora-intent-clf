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
# 默认日志目录 = OUTPUT_DIR（来自 machine.env），无需手动修改本脚本。
# =============================================================================

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# 默认日志目录 = 训练输出目录（含 TensorBoard events）
LOGDIR="${OUTPUT_DIR}"
PORT=6006
BACKGROUND=false
PID_FILE="${PROJECT_ROOT}/logs/tensorboard.pid"

while [[ $# -gt 0 ]]; do
    case $1 in
        --logdir)     LOGDIR="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --background|-b) BACKGROUND=true; shift ;;
        --kill)
            if [[ -f "${PID_FILE}" ]]; then
                PID=$(cat "${PID_FILE}")
                if kill -0 "${PID}" 2>/dev/null; then
                    kill "${PID}" && rm -f "${PID_FILE}"
                    info "TensorBoard (PID=${PID}) 已终止"
                else
                    warn "进程 PID=${PID} 已结束"; rm -f "${PID_FILE}"
                fi
            else
                warn "未找到后台 TensorBoard 进程"
            fi
            exit 0 ;;
        --help|-h)
            echo "用法: bash scripts/run_tensorboard.sh [选项]"
            echo "选项:"
            echo "  --logdir  DIR    日志目录（默认: OUTPUT_DIR = ${OUTPUT_DIR}）"
            echo "  --port    PORT   监听端口（默认: 6006）"
            echo "  --background,-b  后台启动"
            echo "  --kill           终止后台 TensorBoard"
            exit 0 ;;
        *) error "未知参数: $1，使用 --help 查看帮助" ;;
    esac
done

command -v tensorboard &>/dev/null || error "未找到 tensorboard，请先安装: pip install tensorboard==2.9.0"

[[ -d "${LOGDIR}" ]] || warn "日志目录不存在: ${LOGDIR}（训练完成后会自动创建）"
mkdir -p "${PROJECT_ROOT}/logs"

info "TensorBoard 启动 | 日志: ${LOGDIR} | 端口: ${PORT}"
info "SSH 隧道（本地执行）: ssh -L ${PORT}:localhost:${PORT} user@<服务器IP>"

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all \
        >> "${PROJECT_ROOT}/logs/tensorboard.log" 2>&1 &
    TB_PID=$!
    echo "${TB_PID}" > "${PID_FILE}"
    sleep 2
    kill -0 "${TB_PID}" 2>/dev/null && info "TensorBoard 已后台启动 PID=${TB_PID}" \
        || error "TensorBoard 启动失败: cat ${PROJECT_ROOT}/logs/tensorboard.log"
else
    info "前台运行中，Ctrl+C 退出（训练不受影响）"
    tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
fi
