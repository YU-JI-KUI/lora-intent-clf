#!/usr/bin/env bash
# =============================================================================
# 后台训练 + 导出启动脚本 — SSH 断开/终端关闭/机器重启登录后训练不丢失
#
# 【SIGHUP 问题说明】
#   直接用 nohup ... & 仍会被 SIGHUP 中断的原因：
#   nohup 只保护它直接启动的进程，但 llamafactory-cli 内部通过 subprocess
#   启动 torchrun，torchrun 再 fork 多个 worker 进程，这些"孙进程"全在
#   同一个进程组里，终端关闭时 SIGHUP 还是会传递给整个进程组。
#
#   解决方案：setsid（本脚本使用）
#   setsid 创建全新的 session，进程组从根源上与控制终端脱钩，
#   所有子/孙进程（包括 torchrun worker）自动继承新 session，
#   SIGHUP 彻底传不进来。
#
# 执行顺序：训练完成 → 自动执行导出（train + export 一体）
#
# 用法：
#   bash scripts/train_background.sh              # LlamaFactory 方案（推荐）
#   bash scripts/train_background.sh --python     # Python torchrun 方案
#   bash scripts/train_background.sh --status     # 查看是否仍在运行
#   bash scripts/train_background.sh --kill       # 终止后台任务
#   bash scripts/train_background.sh --log        # 实时查看最新日志
#
# 日志文件：/workspace/lora-intent-clf/logs/train_YYYYMMDD_HHMMSS.log
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ------------------------------- 路径配置 ------------------------------------
PROJECT_ROOT="/workspace/lora-intent-clf"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train.pid"
LOG_LINK="${LOG_DIR}/train_latest.log"   # 软链接，始终指向最新日志

mkdir -p "${LOG_DIR}"

# ------------------------------- 辅助功能 ------------------------------------

check_status() {
    if [[ ! -f "${PID_FILE}" ]]; then
        warn "未找到 PID 文件，任务可能未启动或已完成"
        return
    fi
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        info "任务正在运行中，PID=${PID}"
        info "日志文件: $(readlink -f "${LOG_LINK}" 2>/dev/null || echo '未知')"
        info "实时查看: tail -f ${LOG_LINK}"
        if [[ -f "${LOG_LINK}" ]]; then
            echo ""
            echo -e "${CYAN}最近 5 条日志：${NC}"
            tail -n 5 "${LOG_LINK}" 2>/dev/null || true
        fi
    else
        warn "PID=${PID} 的进程已结束（完成或异常退出）"
        warn "查看完整日志: cat ${LOG_LINK}"
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
        # 用负号终止整个进程组（包括 torchrun worker）
        # kill -TERM -PID 终止以 PID 为 leader 的进程组
        kill -TERM "-${PID}" 2>/dev/null || kill -TERM "${PID}" 2>/dev/null || true
        sleep 5
        if kill -0 "${PID}" 2>/dev/null; then
            kill -KILL "-${PID}" 2>/dev/null || kill -KILL "${PID}" 2>/dev/null || true
        fi
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
        info "日志文件: $(readlink -f "${LOG_LINK}" 2>/dev/null)"
        echo "==========================================="
        tail -f "${LOG_LINK}"
    else
        warn "日志文件不存在，任务可能未启动"
    fi
}

# ------------------------------- 参数解析 ------------------------------------
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
            echo "  (无参数)   后台运行 LlamaFactory CLI（训练 + 导出，推荐）"
            echo "  --python   后台运行 Python 脚本（训练 + 导出，torchrun + DeepSpeed）"
            echo "  --status   查看任务是否仍在运行"
            echo "  --kill     终止后台任务（包括所有子进程）"
            echo "  --log      实时查看最新日志（Ctrl+C 不影响后台任务）"
            echo "  --help     显示此帮助"
            echo ""
            echo "日志目录: ${LOG_DIR}"
            exit 0
            ;;
        *) error "未知参数: $arg，使用 --help 查看帮助" ;;
    esac
done

# ------------------------------- 防止重复启动 ---------------------------------
if [[ -f "${PID_FILE}" ]]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        error "任务已在后台运行（PID=${PID}）。先用 --kill 终止它，再重新启动。"
    else
        warn "旧的 PID 文件已失效，清理中..."
        rm -f "${PID_FILE}"
    fi
fi

# ------------------------------- 准备日志文件 ---------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# 软链接指向最新日志
ln -sf "${LOG_FILE}" "${LOG_LINK}"

# ------------------------------- 构建完整命令（训练 + 导出）------------------
if [[ "${USE_PYTHON}" == "false" ]]; then
    # -------------------------------------------------------------------------
    # LlamaFactory 方案：复用 run_train_and_export.sh（已包含训练 + 导出）
    # -------------------------------------------------------------------------
    RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_train_and_export.sh"
    [[ -f "${RUN_SCRIPT}" ]] || error "启动脚本不存在: ${RUN_SCRIPT}"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli"

    # 将整个 train+export 流程包在一个 bash 里，setsid 对这个 bash 生效，
    # 所有子孙进程（torchrun worker 等）都继承新 session，彻底免疫 SIGHUP
    CMD="bash ${RUN_SCRIPT}"
    MODE="LlamaFactory CLI（训练 + 导出）+ DeepSpeed ZeRO-3"

else
    # -------------------------------------------------------------------------
    # Python 方案：torchrun 训练 → python 导出，两步顺序执行
    # -------------------------------------------------------------------------
    [[ -f "${PROJECT_ROOT}/src/train.py" ]]        || error "训练脚本不存在: src/train.py"
    [[ -f "${PROJECT_ROOT}/src/export_model.py" ]] || error "导出脚本不存在: src/export_model.py"
    command -v torchrun &>/dev/null || error "未找到 torchrun（PyTorch >= 1.9 内置）"

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")
    DEEPSPEED_CONFIG="${PROJECT_ROOT}/configs/deepspeed/ds_z3_config.json"

    # && 保证训练成功才执行导出，训练失败时停止并记录错误
    CMD="cd ${PROJECT_ROOT} && \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        torchrun --nproc_per_node=${GPU_COUNT} src/train.py --deepspeed ${DEEPSPEED_CONFIG} && \
        echo '>>> 训练完成，开始导出...' && \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        python src/export_model.py && \
        echo '>>> 导出完成！'"
    MODE="Python torchrun + DeepSpeed ZeRO-3（${GPU_COUNT} GPU）训练 + 导出"
fi

# ------------------------------- 后台启动 ------------------------------------
info "=========================================="
info "后台任务启动"
info "模式: ${MODE}"
info "日志: ${LOG_FILE}"
info "=========================================="
info "SIGHUP 防护: setsid（新 session，所有子进程免疫终端关闭信号）"
info "=========================================="

# 关键：setsid 创建新 session → nohup 忽略 SIGHUP → bash -c 执行命令
# 三层防护：setsid（新session） + nohup（忽略信号） + 重定向（脱离终端IO）
nohup setsid bash -c "${CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# 保存 PID（这是 setsid bash 的 PID，也是新进程组的 leader）
echo "${TRAIN_PID}" > "${PID_FILE}"

# 等待 3 秒确认进程没有立即退出
sleep 3
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    error "任务启动后立即退出，请检查日志: cat ${LOG_FILE}"
fi

info "任务已在后台启动，PID=${TRAIN_PID}"
echo ""
echo -e "${CYAN}常用命令：${NC}"
echo "  实时查看日志:  bash scripts/train_background.sh --log"
echo "             或: tail -f ${LOG_LINK}"
echo "  查看运行状态:  bash scripts/train_background.sh --status"
echo "  终止任务:      bash scripts/train_background.sh --kill"
echo ""
echo -e "${CYAN}日志文件：${NC}"
echo "  ${LOG_FILE}"
