# -*- coding: utf-8 -*-
"""封装所有后台任务相关的线程/任务，减少界面文件的代码量。"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5 import QtCore


class FunctionThread(QtCore.QThread):
    """通用 QThread 封装，按需注入 progress/cancel 参数。"""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        progress_arg: Optional[str] = "progress_cb",
        cancel_arg: Optional[str] = None,
        result_adapter: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._progress_arg = progress_arg
        self._cancel_arg = cancel_arg
        self._result_adapter = result_adapter

    def _emit_progress(self, value: Any) -> None:
        try:
            pct = int(float(value))
        except Exception:
            pct = 0
        self.progress.emit(max(0, min(100, pct)))

    def run(self) -> None:  # pragma: no cover - PyQt thread lifecycle
        kwargs = dict(self._kwargs)
        if self._progress_arg:
            kwargs[self._progress_arg] = self._emit_progress
        if self._cancel_arg:
            kwargs[self._cancel_arg] = self.isInterruptionRequested
        try:
            result = self._fn(*self._args, **kwargs)
            if self._result_adapter:
                result = self._result_adapter(result)
        except Exception as exc:  # pragma: no cover - surface to UI
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)


class BackgroundTask(QtCore.QRunnable):
    """轻量级 QRunnable 包装器，用于线程池任务。"""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = WorkerSignals()

    @property
    def finished(self):
        return self.signals.finished

    @property
    def error(self):
        return self.signals.error

    @property
    def progress(self):
        return self.signals.progress

    def _emit_progress(self, value: Any) -> None:
        try:
            pct = int(float(value))
        except Exception:
            pct = 0
        self.signals.progress.emit(max(0, min(100, pct)))

    def run(self) -> None:  # pragma: no cover - QRunnable lifecycle
        kwargs = dict(self._kwargs)
        kwargs.setdefault("progress_cb", self._emit_progress)
        try:
            result = self._fn(*self._args, **kwargs)
        except Exception as exc:  # pragma: no cover - surface to UI
            self.signals.error.emit(str(exc))
            return
        self.signals.finished.emit(result)
