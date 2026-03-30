#!/usr/bin/env python3
"""
环境诊断脚本 — 找出 numpy.bool8 / numpy.bool / numpy.int / numpy.float 等
已删除 API 的来源包，并输出修复建议。

用法:
    python scripts/diagnose_env.py
    python scripts/diagnose_env.py --full   # 扫描所有 site-packages（较慢）
"""

import argparse
import importlib
import subprocess
import sys
import traceback
from pathlib import Path


# ─── 1. 扫描已安装包中是否使用了废弃 numpy API ────────────────────────────────

DEPRECATED_APIS = [
    r"np\.bool8",
    r"numpy\.bool8",
    r"np\.bool[^_\[]",
    r"numpy\.bool[^_\[]",
    r"np\.int[^0-9_\[\(]",
    r"numpy\.int[^0-9_\[\(]",
    r"np\.float[^0-9_\[\(]",
    r"numpy\.float[^0-9_\[\(]",
    r"np\.complex[^0-9_\[\(]",
    r"np\.object[^_\[]",
    r"np\.str[^_\[]",
]

# 重点检查的包（DeepSpeed 的常见依赖，历史上使用过废弃 API）
SUSPECT_PACKAGES = [
    "deepspeed",
    "triton",
    "pynvml",
    "hjson",
    "ninja",
    "bitsandbytes",
    "transformer_engine",
    "apex",
    "torch",
    "accelerate",
]


def get_site_packages() -> list[Path]:
    import site

    dirs = []
    try:
        dirs += [Path(p) for p in site.getsitepackages()]
    except AttributeError:
        pass
    try:
        dirs.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    # 也检查当前 venv
    venv_site = Path(sys.executable).parent.parent / "lib"
    for p in venv_site.glob("python*/site-packages"):
        dirs.append(p)
    return [d for d in dict.fromkeys(dirs) if d.exists()]


def scan_package(pkg_name: str, site_dirs: list[Path]) -> list[str]:
    """扫描指定包目录，返回含废弃 API 的行（最多 20 条）"""
    results = []
    for site_dir in site_dirs:
        pkg_dir = site_dir / pkg_name
        if not pkg_dir.exists():
            # 也尝试 pkg_name.replace("-", "_")
            pkg_dir = site_dir / pkg_name.replace("-", "_")
        if not pkg_dir.exists():
            continue

        patterns = "|".join(DEPRECATED_APIS)
        cmd = ["grep", "-rEn", patterns, str(pkg_dir), "--include=*.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.stdout:
                lines = r.stdout.strip().split("\n")
                results += lines[:20]
                if len(lines) > 20:
                    results.append(f"  ... (共 {len(lines)} 处，只显示前 20 条)")
        except subprocess.TimeoutExpired:
            results.append(f"  [超时]")
        break

    return results


def scan_all(site_dirs: list[Path]) -> list[str]:
    """扫描全部 site-packages（较慢）"""
    results = []
    for site_dir in site_dirs:
        patterns = "|".join(DEPRECATED_APIS)
        cmd = ["grep", "-rEn", patterns, str(site_dir), "--include=*.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.stdout:
                lines = r.stdout.strip().split("\n")
                results += lines[:50]
        except subprocess.TimeoutExpired:
            pass
    return results


# ─── 2. 动态导入捕获完整 traceback ────────────────────────────────────────────

def try_import_and_capture(module_name: str) -> tuple[bool, str]:
    """尝试导入模块，返回 (成功, traceback或'') """
    try:
        # 强制重新导入
        if module_name in sys.modules:
            del sys.modules[module_name]
        importlib.import_module(module_name)
        return True, ""
    except AttributeError as e:
        if "bool8" in str(e) or "bool" in str(e).lower():
            return False, traceback.format_exc()
        return False, f"其他 AttributeError: {e}"
    except Exception as e:
        return False, f"导入失败（非 bool8 原因）: {type(e).__name__}: {e}"


# ─── 3. 检查已安装版本 ────────────────────────────────────────────────────────

def get_installed_version(pkg: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg)
    except Exception:
        return "未安装"


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="扫描全部 site-packages")
    args = parser.parse_args()

    print("=" * 60)
    print("numpy 兼容性诊断")
    print("=" * 60)

    # numpy 版本
    try:
        import numpy as np
        print(f"numpy 版本: {np.__version__}")
        has_bool8 = hasattr(np, "bool8")
        print(f"np.bool8 存在: {has_bool8}  ({'OK' if has_bool8 else '❌ 已删除，numpy >= 1.24'})")
    except ImportError:
        print("❌ numpy 未安装")
        return

    print()

    # 重点包版本
    print("─── 关键包版本 ───")
    for pkg in ["deepspeed", "torch", "numpy", "triton", "accelerate", "bitsandbytes"]:
        ver = get_installed_version(pkg)
        print(f"  {pkg:<20} {ver}")

    print()

    # site-packages 路径
    site_dirs = get_site_packages()
    print(f"─── site-packages 目录 ───")
    for d in site_dirs:
        print(f"  {d}")
    print()

    if args.full:
        # 全量扫描
        print("─── 全量扫描（可能较慢）───")
        results = scan_all(site_dirs)
        if results:
            print("发现废弃 API 用法:")
            for line in results:
                print(f"  {line}")
        else:
            print("✅ 未发现废弃 numpy API 用法")
    else:
        # 只扫描嫌疑包
        print("─── 扫描嫌疑包 ───")
        found_any = False
        for pkg in SUSPECT_PACKAGES:
            results = scan_package(pkg, site_dirs)
            if results:
                found_any = True
                print(f"\n❌ [{pkg}] 发现废弃 API:")
                for line in results:
                    print(f"   {line}")
        if not found_any:
            print("✅ 嫌疑包中未发现废弃 API（可用 --full 扫描全部）")

    print()

    # 动态导入测试
    print("─── 导入测试 ───")
    for mod in ["deepspeed", "deepspeed.ops.adam", "triton"]:
        ok, err = try_import_and_capture(mod)
        status = "✅ OK" if ok else "❌ 失败"
        print(f"  import {mod:<30} {status}")
        if err and "bool8" in err.lower():
            print(f"\n  完整 traceback（含 bool8 错误）:\n")
            for line in err.strip().split("\n"):
                print(f"    {line}")
            print()

    print()
    print("─── 修复建议 ───")
    ds_ver = get_installed_version("deepspeed")
    triton_ver = get_installed_version("triton")
    bb_ver = get_installed_version("bitsandbytes")

    print("如果错误来自 triton:")
    print(f"  当前版本: {triton_ver}")
    print("  pip install 'triton>=2.3.0'")
    print()
    print("如果错误来自 bitsandbytes:")
    print(f"  当前版本: {bb_ver}")
    print("  pip install 'bitsandbytes>=0.42.0'")
    print()
    print("如果错误来自 deepspeed 内部（某些 op 脚本）:")
    print(f"  当前版本: {ds_ver}")
    print("  pip install 'deepspeed>=0.16.0' --upgrade")
    print()
    print("通用临时绕过（在报错包修复前）:")
    print("  在代码最前面加: import src.compat  # numpy 兼容性补丁")
    print()
    print("若要完整扫描所有已安装包，运行:")
    print("  python scripts/diagnose_env.py --full 2>&1 | head -100")


if __name__ == "__main__":
    main()
