#!/usr/bin/env bash
# =============================================================================
# 后台训练启动脚本 — 训练过程在后台运行，断开 SSH/关闭终端也不会中断
#
# 原理：使用 nohup 将进程与终端解绑，stdout/stderr 重定向到日志文件
#       PID 保存到 logs/train.pid，方便后续监控和终止
#
# 用法：
#   bash scripts/train_background.sh              # 后台运行 LlamaFactory 方案
#   bash scripts/train_background.sh --python     # 后台运行 Python 脚本方案
#   bash scripts/train_background.sh --status     # 查看训练是否仍在运行
#   bash scripts/train_background.sh --kill       # 终止训练
#   bash scripts/train_background.sh --log        # 实时查看最新日志
#
# 日志文件：/workspace/lora-intent-clf/logs/train_YYYYMMDD_HHMMSS.log
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

# ------------------------------- 路径配置 ------------------------------------
PROJECT_ROOT="/workspace/lora-intent-clf"
LOG_DIR="${PROJECT_ROOT}/logs"
PID_FILE="${LOG_DIR}/train.pid"
LOG_LINK="${LOG_DIR}/train_latest.log"   # 软链接，始终指向最新日志

mkdir -p "${LOG_DIR}"

# ------------------------------- 辅助功能 ------------------------------------

check_status() {
    if [[ ! -f "${PID_FILE}" ]]; then
        warn "未找到 PID 文件，训练可能未启动或已完成"
        return
    fi
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        info "训练正在运行中，PID=${PID}"
        info "日志文件: $(readlink -f ${LOG_LINK} 2>/dev/null || echo '未知')"
        info "实时查看: tail -f ${LOG_LINK}"
        # 显示当前进度（从日志中提取最后的进度行）
        if [[ -f "${LOG_LINK}" ]]; then
            echo ""
            echo -e "${CYAN}最近 5 条日志：${NC}"
            tail -n 5 "${LOG_LINK}" 2>/dev/null || true
        fi
    else
        warn "PID=${PID} 的进程已结束（训练完成或异常退出）"
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
        warn "正在终止训练进程 PID=${PID}..."
        # 先发 SIGTERM，给进程保存 checkpoint 的机会
        kill -TERM "${PID}" 2>/dev/null || true
        sleep 5
        # 如果还在，强制 kill
        if kill -0 "${PID}" 2>/dev/null; then
            kill -KILL "${PID}" 2>/dev/null || true
        fi
        rm -f "${PID_FILE}"
        info "训练进程已终止"
    else
        warn "PID=${PID} 的进程不存在（已结束）"
        rm -f "${PID_FILE}"
    fi
}

show_log() {
    if [[ -f "${LOG_LINK}" ]]; then
        info "实时监控日志（Ctrl+C 退出监控，不影响训练）"
        info "日志文件: $(readlink -f ${LOG_LINK} 2>/dev/null)"
        echo "==========================================="
        tail -f "${LOG_LINK}"
    else
        warn "日志文件不存在，训练可能未启动"
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
            echo "  (无参数)   后台运行 LlamaFactory CLI 训练（推荐）"
            echo "  --python   后台运行 Python 脚本训练（需配合 torchrun）"
            echo "  --status   查看训练是否仍在运行"
            echo "  --kill     终止后台训练进程"
            echo "  --log      实时查看最新训练日志"
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
        error "训练已在后台运行（PID=${PID}）。先用 --kill 终止它，再重新启动。"
    else
        warn "旧的 PID 文件已失效，清理中..."
        rm -f "${PID_FILE}"
    fi
fi

# ------------------------------- 准备日志文件 ---------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# 软链接指向最新日志，方便 --log / --status 查看
ln -sf "${LOG_FILE}" "${LOG_LINK}"

# ------------------------------- 构建启动命令 ---------------------------------
if [[ "${USE_PYTHON}" == "false" ]]; then
    # LlamaFactory 方案
    TRAIN_CONFIG="${PROJECT_ROOT}/configs/train_lora_sft.yaml"
    EXPORT_CONFIG="${PROJECT_ROOT}/configs/export_lora.yaml"

    [[ -f "${TRAIN_CONFIG}" ]] || error "训练配置不存在: ${TRAIN_CONFIG}"
    command -v llamafactory-cli &>/dev/null || error "未找到 llamafactory-cli"

    CMD="FORCE_TORCHRUN=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python llamafactory-cli train ${TRAIN_CONFIG}"
    MODE="LlamaFactory CLI + DeepSpeed ZeRO-3"
else
    # Python 脚本方案（需要 torchrun 启动，才能使用 DeepSpeed 多卡）
    [[ -f "${PROJECT_ROOT}/src/train.py" ]] || error "训练脚本不存在: ${PROJECT_ROOT}/src/train.py"
    command -v torchrun &>/dev/null || error "未找到 torchrun，请确认 PyTorch 版本 >= 1.9"

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 || echo "1")
    DEEPSPEED_CONFIG="${PROJECT_ROOT}/configs/deepspeed/ds_z3_config.json"

    CMD="PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python torchrun \
        --nproc_per_node=${GPU_COUNT} \
        ${PROJECT_ROOT}/src/train.py \
        --deepspeed ${DEEPSPEED_CONFIG}"
    MODE="Python torchrun + DeepSpeed ZeRO-3 (${GPU_COUNT} GPU)"
fi

# ------------------------------- 后台启动 ------------------------------------
info "=========================================="
info "后台训练启动"
info "模式: ${MODE}"
info "日志: ${LOG_FILE}"
info "=========================================="

# nohup：忽略 SIGHUP（终端关闭信号），进程继续运行
# 2>&1：将 stderr 也重定向到同一日志文件
# &：放入后台
nohup bash -c "${CMD}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

# 保存 PID
echo "${TRAIN_PID}" > "${PID_FILE}"

# 等待 2 秒确认进程没有立即退出
sleep 2
if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    error "训练进程启动后立即退出，请检查日志: cat ${LOG_FILE}"
fi

info "训练已在后台启动，PID=${TRAIN_PID}"
echo ""
echo -e "${CYAN}常用命令：${NC}"
echo "  实时查看日志:  bash scripts/train_background.sh --log"
echo "             或: tail -f ${LOG_LINK}"
echo "  查看运行状态:  bash scripts/train_background.sh --status"
echo "  终止训练:      bash scripts/train_background.sh --kill"
echo ""
echo -e "${CYAN}日志文件位置：${NC}"
echo "  ${LOG_FILE}"
