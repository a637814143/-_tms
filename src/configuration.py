"""轻量级配置加载模块，统一管理路径等参数。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _expand_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.expanduser().resolve()


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, object]:
    """读取 YAML 配置，默认使用 ``config/default.yaml``。"""

    explicit = config_path or os.getenv("MALDET_CONFIG")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(DEFAULT_CONFIG_PATH)

    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return {}


def get_path(key: str, default: Optional[str] = None, *, ensure_exists: bool = True) -> Path:
    """根据配置获取路径，默认相对项目根目录。"""

    config = load_config()
    paths = config.get("paths") if isinstance(config, dict) else {}
    if isinstance(paths, dict) and key in paths:
        raw_value = paths[key]
    else:
        raw_value = default
    if raw_value is None:
        raise KeyError(f"配置中缺少路径键: {key}")
    path = _expand_path(str(raw_value))
    if ensure_exists:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_paths(keys: Optional[Dict[str, str]] = None, *, ensure_exists: bool = True) -> Dict[str, Path]:
    """批量获取路径。``keys`` 允许提供别名到配置键的映射。"""

    result: Dict[str, Path] = {}
    if keys:
        for alias, config_key in keys.items():
            result[alias] = get_path(config_key, ensure_exists=ensure_exists)
        return result

    config = load_config()
    paths = config.get("paths") if isinstance(config, dict) else {}
    if isinstance(paths, dict):
        for key, value in paths.items():
            try:
                result[key] = get_path(key, str(value), ensure_exists=ensure_exists)
            except KeyError:
                continue
    return result


def project_root() -> Path:
    return PROJECT_ROOT


__all__ = ["load_config", "get_path", "get_paths", "project_root"]