#!/usr/bin/env bash
# =============================================================================
# 后台训练 + 导出启动脚本 — SSH 断开/终端关闭不影响训练
#
# 【SIGHUP 防护】
#   nohup + setsid 双重保护：setsid 创建新 session，所有子/孙进程
#   （torchrun worker 等）自动继承，SIGHUP 彻底无法传入。
#
# 用法：
#   bash scripts/train_background.sh              # LlamaFactory 方案（推荐）
#   bash scripts/train_background.sh --python     # Python torchrun 方案
#   bash scripts/train_background.sh --status     # 查看运行状态
#   bash scripts/train_background.sh --kill       # 终止后台任务
#   bash scripts/train_background.sh --log        # 实时查看日志
#
# 机器特定参数从 machine.env 自动加载，无需手动修改本脚本。
# =============================================================================

set -euo pipefail

# 加载公共配置（machine.env + PROJECT_ROOT 推导 + PYTHON/TORCHRUN 路径）
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train.pid"
LOG_LINK="${LOG_DIR}/train_latest.log"

mkdir -p "${LOG_DIR}"

# ─── 辅助命令 ─────────────────────────────────────────────────────────────────
check_status() {
    if [[ ! -f "${PID_FILE}" ]]; then
        warn "未找到 PID 文件，任务可能未启动或已完成"
        return
    fi
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        info "任务正在运行中，PID=${PID}"
        info "日志文件: $(readlink -f "${LOG_LINK}" 2>/dev/null || echo '未知')"
        [[ -f "${LOG_LINK}" ]] && echo -e "${CYAN}最近 5 条日志：${NC}" && tail -n 5 "${LOG_LINK}" 2>/dev/null || true
    else
        warn "PID=${PID} 的进程已结束（完成或异常退出）"
        rm -f "${PID_FILE}"
    fi
}

kill_training() {
    if [[ ! -f "${PID_FILE}" ]]; then
        warn "未找到 PID 文件，无法终止"
        return
    fi
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        warn "正在终止任务进程 PID=${PID} 及其所有子进程..."
        kill -TERM "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
        sleep 5
        kill -0 "${PID}" 2>/dev/null && { kill -KILL "-${PID}" 2>/dev/null || true; }
        rm -f "${PID_FILE}"
        info "任务进程已终止"
    else
        warn "PID=${PID} 的进程不存在（已结束）"
        rm -f "${PID_FILE}"
    fi
}

show_log() {
    if [[ -f "${LOG_LINK}" ]]; then
        info "实时监控日志（Ctrl+C 退出监控，不影响后台任务）"
        tail -f "${LOG_LINK}"
    else
        warn "日志文件不存在，任务可能未启动"
    fi
}

# ─── 参数解析 ─────────────────────────────────────────────────────────────────
USE_PYTHON=false

for arg in "$@"; do
    case $arg in
        --python)  USE_PYTHON=true ;;
        --status)  check_status; exit 0 ;;
        --kill)    kill_training; exit 0 ;;
        --log)     show_log; exit 0 ;;
        --help|-h)
            echo "用法: bash scripts/train_background.sh [选项]"
            echo ""
            echo "选项:"
            echo "  (无参数)   后台运行 LlamaFactory CLI（推荐）"
            echo "  --python   后台运行 Python torchrun 方案"
            echo "  --status   查看任务是否仍在运行"
            echo "  --kill     终止后台任务（包括所有子进程）"
            echo "  --log      实时查看最新日志（Ctrl+C 不影响后台）"
            echo ""
            echo "当前配置（来自 machine.env）："
            echo "  MODEL_PATH       = ${MODEL_PATH}"
            echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
            echo "  NPROC_PER_NODE   = ${NPROC_PER_NODE}"
            echo "  DEEPSPEED_CONFIG = ${DEEPSPEED_CONFIG}"
            echo "  TORCHRUN         = ${TORCHRUN}"
            echo "  PYTHON           = ${PYTHON}"
            exit 0 ;;
        *) error "未知参数: $arg，使用 --help 查看帮助" ;;
    esac
done

# ─── 防止重复启动 ─────────────────────────────────────────────────────────────
if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        error "任务已在后台运行（PID=${PID}）。先用 --kill 终止它，再重新启动。"
    else
        warn "旧的 PID 文件已失效，清理中..."
        rm -f "${PID_FILE}"
    fi
fi

# ─── 准备日志文件 ─────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
ln -sf "${LOG_FILE}" "${LOG_LINK}"

# ─── 构建命令 ─────────────────────────────────────────────────────────────────
if [[ "${USE_PYTHON}" == "false" ]]; then
    # LlamaFactory 方案：复用 run_train_and_export.sh
    RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_train_and_export.sh"
    [[ -f "${RUN_SCRIPT}" ]] || error "启动脚本不存在: ${RUN_SCRIPT}"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli"

    CMD="bash ${RUN_SCRIPT}"
    MODE="LlamaFactory CLI（训练 + 导出）+ DeepSpeed ZeRO-3"
else
    # Python 方案：直接用 .venv/bin/torchrun，避免 PATH 问题
    [[ -f "${PROJECT_ROOT}/src/train.py" ]]        || error "训练脚本不存在: src/train.py"
    [[ -f "${PROJECT_ROOT}/src/export_model.py" ]] || error "导出脚本不存在: src/export_model.py"

    # 【torchrun bug 修复】
    # 问题：直接用 `torchrun` 会找到系统 Python 的 torchrun，但项目依赖在 .venv 中，
    #       导致 "the following arguments are required: training_script" 等错误。
    # 修复：使用 common.sh 导出的 $TORCHRUN（指向 .venv/bin/torchrun），
    #       确保 torchrun 和 train.py 使用同一个 Python 环境。
    [[ -x "${TORCHRUN}" ]] || error "未找到 torchrun: ${TORCHRUN}（请先运行 bash scripts/setup_env.sh）"

    CMD="cd ${PROJECT_ROOT} && \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        ${TORCHRUN} --nproc_per_node=${NPROC_PER_NODE} src/train.py && \
        echo '>>> 训练完成，开始导出...' && \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        ${PYTHON} src/export_model.py && \
        echo '>>> 导出完成！'"
    MODE="Python torchrun（${NPROC_PER_NODE} GPU）+ DeepSpeed ZeRO-3 + 导出"
fi

# ─── 后台启动 ─────────────────────────────────────────────────────────────────
info "=========================================="
info "后台任务启动"
info "模式: ${MODE}"
info "日志: ${LOG_FILE}"
info "SIGHUP 防护: setsid（新 session，所有子进程免疫终端关闭信号）"
info "=========================================="

nohup setsid bash -c "${CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${PID_FILE}"

sleep 3
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    error "任务启动后立即退出，请检查日志: cat ${LOG_FILE}"
fi

info "任务已在后台启动，PID=${TRAIN_PID}"
echo ""
echo -e "${CYAN}常用命令：${NC}"
echo "  实时查看日志:  bash scripts/train_background.sh --log"
echo "  查看运行状态:  bash scripts/train_background.sh --status"
echo "  终止任务:      bash scripts/train_background.sh --kill"
echo ""
echo -e "${CYAN}日志文件：${NC} ${LOG_FILE}"
