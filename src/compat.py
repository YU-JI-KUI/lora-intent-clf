"""
numpy 兼容性垫片（compat shim）

背景：
  numpy 1.20 将以下类型别名标记为 deprecated，numpy 1.24 正式删除：
    np.bool8, np.bool, np.int, np.float, np.complex, np.object, np.str

  torch 2.6.0 要求 numpy >= 1.26，因此上述别名已不存在。
  但 DeepSpeed 的某些依赖包（triton / bitsandbytes 旧版等）仍在使用。

作用：
  在 import deepspeed / torch 之前，将已删除的别名重新注册为对应的新 API，
  消除 AttributeError: module numpy has no attribute 'bool8'。

使用方式（本项目所有入口脚本已自动在首行 import）：
  import compat  # 必须在 import deepspeed / torch 之前执行

注意：
  这是临时绕过方案。根本修复是升级使用废弃 API 的那个包。
"""

import numpy as np

# numpy 1.24 删除的别名 → 对应的正确 API
_REMOVED_ALIASES: dict[str, str] = {
    "bool8":   "bool_",
    "bool":    "bool_",
    "int":     "int_",
    "float":   "float64",
    "complex": "complex128",
    "object":  "object_",
    "str":     "str_",
}

_patched: list[str] = []

for _alias, _target in _REMOVED_ALIASES.items():
    if not hasattr(np, _alias) and hasattr(np, _target):
        setattr(np, _alias, getattr(np, _target))
        _patched.append(f"np.{_alias} → np.{_target}")

if _patched:
    import logging as _logging
    _log = _logging.getLogger(__name__)
    _log.debug(
        "numpy compat shim 已应用（numpy %s），修复了 %d 个已删除别名: %s",
        np.__version__,
        len(_patched),
        ", ".join(_patched),
    )
