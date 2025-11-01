# -*- coding: utf-8 -*-
"""后台监控线程相关实现。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set

from PyQt5 import QtCore


class OnlineDetectionWorker(QtCore.QThread):
    new_file = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()

    def __init__(self, watch_dir: str, poll_seconds: int = 5, parent=None):
        super().__init__(parent)
        self.watch_dir = Path(watch_dir)
        self.poll_seconds = max(1, int(poll_seconds))
        self._stop_flag = False
        self._seen: Set[str] = set()

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:  # pragma: no cover - 线程逻辑难以单元测试
        if not self.watch_dir.exists():
            try:
                self.watch_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.error.emit(f"无法创建监控目录：{exc}")
                return
        self.status.emit(f"监控目录：{self.watch_dir}")
        patterns = ("*.pcap", "*.pcapng")
        while not self._stop_flag:
            try:
                files: List[Path] = []
                for pattern in patterns:
                    files.extend(sorted(self.watch_dir.glob(pattern)))
                for path in files:
                    norm = str(path.resolve())
                    if norm in self._seen:
                        continue
                    self._seen.add(norm)
                    self.new_file.emit(norm)
            except Exception as exc:
                self.error.emit(str(exc))
            for _ in range(self.poll_seconds * 10):
                if self._stop_flag:
                    break
                self.msleep(100)
        self.stopped.emit()


__all__ = ["OnlineDetectionWorker"]
