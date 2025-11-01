# -*- coding: utf-8 -*-
"""应用设置持久化工具类，从主界面文件中拆分出来。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AppSettings:
    """简单的 JSON 设置存储，封装读写逻辑。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                self.data = payload
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass


__all__ = ["AppSettings"]
