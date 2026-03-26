#!/usr/bin/env bash
# =============================================================================
# 使用 GitHub CLI 创建公开仓库并推送代码
# 前提：已安装 gh CLI 并完成 gh auth login
# 用法: bash scripts/setup_github.sh [repo-name]
# =============================================================================

set -euo pipefail

REPO_NAME="${1:-lora-intent-clf}"

GREEN='\033[0;32m'
NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC} $*"; }

# 检查 gh CLI
if ! command -v gh &> /dev/null; then
    echo "错误: 未找到 gh CLI，请先安装: https://cli.github.com/"
    exit 1
fi

# 检查登录状态
if ! gh auth status &> /dev/null; then
    echo "错误: 未登录 GitHub，请先执行: gh auth login"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

info "创建公开仓库: ${REPO_NAME}"
gh repo create "${REPO_NAME}" --public --source=. --push

info "完成！仓库地址:"
gh repo view --web 2>/dev/null || gh repo view --json url -q .url
