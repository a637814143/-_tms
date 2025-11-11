# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os, platform, subprocess, math, shutil, json, io, time, textwrap
from pathlib import Path
import numpy as np
from typing import Callable, Collection, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

import yaml
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "PingFang SC",
    "SimHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from joblib import load as joblib_load

# ---- 业务函数（保持导入路径）----
from src.configuration import get_path, get_paths, load_config, project_root
from src.functions.info import get_pcap_features as info
from src.functions.feature_extractor import (
    extract_features as fe_single,
    extract_features_dir as fe_dir,
    get_loaded_plugin_info,
)
from src.functions.csv_utils import read_csv_flexible
try:
    from src.functions.modeling import (
        META_COLUMNS as TRAIN_META_COLUMNS,
        train_unsupervised_on_split as run_train,
    )
except Exception:  # pragma: no cover - fallback for minimal environments
    from src.functions.simple_unsupervised import train_unsupervised_on_split as run_train  # type: ignore

    TRAIN_META_COLUMNS = {
        "pcap_file",
        "flow_id",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "__source_file__",
        "__source_path__",
    }

try:
    from src.functions import summarize_prediction_labels
except Exception:  # pragma: no cover - helper unavailable in minimal builds
    summarize_prediction_labels = None  # type: ignore
from src.functions.analyze_results import analyze_results as run_analysis
from src.functions.vectorizer import (
    preprocess_feature_dir as preprocess_dir,
    FeatureSource,
)
from src.functions.annotations import (
    upsert_annotation,
    annotation_summary,
    apply_annotations_to_frame,
)

try:
    import pandas as pd
except Exception:
    pd = None


# ---- 简化的通用工具（保持在单文件中）----


APP_STYLE = """
QWidget {
  background-color: #F5F6FA;
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI";
  font-size: 14px;
  color: #1F1F1F;
}

#TitleBar {
  background: #F5F6FA;
  border-bottom: 1px solid #E5E7EB;
}

QGroupBox {
  border: 1px solid #E5E7EB;
  border-radius: 10px;
  margin-top: 12px;
  background: #FFFFFF;
}

QGroupBox::title {
  subcontrol-origin: margin;
  left: 12px;
  padding: 0 6px;
  font-weight: 600;
  color: #111827;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: 8px 10px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
  border: 1px solid #60A5FA;
}

QPushButton {
  background-color: #0EA5E9;
  color: #FFFFFF;
  border-radius: 10px;
  padding: 8px 16px;
  min-height: 38px;
  border: 0;
}

QPushButton:hover { background-color: #38BDF8; }
QPushButton:pressed { background-color: #0284C7; }
QPushButton:disabled { background-color: #C7CDD4; color: #F9FAFB; }

QPushButton#secondary {
  background: #EEF1F6;
  color: #111827;
  border-radius: 10px;
  padding: 8px 16px;
  min-height: 38px;
}

QPushButton#secondary:hover { background: #E5EAF1; }
QPushButton#secondary:pressed { background: #D9E0EA; }

QHeaderView::section {
  background: #F3F4F6;
  border: 1px solid #E5E7EB;
  padding: 8px;
  font-weight: 600;
}

QTableView {
  background: #FFFFFF;
  gridline-color: #E5E7EB;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

QSplitter::handle {
  background-color: #E5E7EB;
  width: 2px;
}

QStatusBar {
  background: #F3F4F6;
  border-top: 1px solid #E5E7EB;
  font-size: 12px;
  color: #6B7280;
  padding: 4px 8px;
}

QTextEdit, QPlainTextEdit, QListWidget {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

QToolButton {
  background: transparent;
  border: none;
  padding: 4px;
}

QToolButton:hover {
  background: rgba(14, 165, 233, 0.1);
  border-radius: 6px;
}

QScrollArea { border: none; }

#OnlineStatusLabel {
  font-size: 12px;
  color: #5F6368;
  padding: 6px 8px;
  border-radius: 6px;
  background-color: #F3F4F6;
}
"""


class AppSettings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: Dict[str, object] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.data = payload if isinstance(payload, dict) else {}
        except Exception:
            self.data = {}

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass


class PandasFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, frame: "pd.DataFrame", parent=None) -> None:
        super().__init__(parent)
        if pd is None or frame is None:
            self._df = pd.DataFrame() if pd is not None else frame
        else:
            self._df = frame
        # 兼容旧逻辑：外部会访问 _df 属性获取原始 DataFrame
        self._frame = self._df
        self._columns = list(self._df.columns) if pd is not None else []

    def rowCount(self, parent=None):  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        if pd is None or self._df is None:
            return 0
        return len(self._df)

    def columnCount(self, parent=None):  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        if pd is None or self._df is None:
            return 0
        return len(self._columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid() or role not in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole, QtCore.Qt.ToolTipRole):
            return None
        if pd is None or self._df is None:
            return ""
        try:
            value = self._df.iloc[index.row(), index.column()]
        except Exception:
            return ""
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):  # type: ignore[override]
        if role != QtCore.Qt.DisplayRole:
            return None
        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._columns[section]
            except Exception:
                return None
        return section + 1

    def flags(self, index):  # type: ignore[override]
        if not index.isValid():
            return QtCore.Qt.NoItemFlags
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class FunctionThread(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(
        self,
        fn,
        *args,
        progress_arg: Optional[str] = "progress_cb",
        cancel_arg: Optional[str] = None,
        result_adapter=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._progress_arg = progress_arg
        self._cancel_arg = cancel_arg
        self._result_adapter = result_adapter

    def _emit_progress(self, value) -> None:
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


class _WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)


class BackgroundTask(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.signals = _WorkerSignals()

    @property
    def finished(self):
        return self.signals.finished

    @property
    def error(self):
        return self.signals.error

    @property
    def progress(self):
        return self.signals.progress

    def _emit_progress(self, value) -> None:
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


class FeaturePickDialog(QtWidgets.QDialog):
    def __init__(self, features: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择特征列")
        self.resize(420, 480)
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.addItems(features)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        layout.addWidget(self.list_widget)
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def selected(self) -> List[str]:
        return [item.text() for item in self.list_widget.selectedItems()]


class ResultsDashboard(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("结果仪表盘", parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.summary_label = QtWidgets.QLabel("暂无数据")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        grid = QtWidgets.QGridLayout()
        layout.addLayout(grid)
        self.metrics: Dict[str, QtWidgets.QLabel] = {}
        names = [
            ("total", "总流量"),
            ("anomalies", "异常流"),
            ("alerts", "高风险"),
            ("last_update", "更新时间"),
        ]
        for idx, (key, label) in enumerate(names):
            grid.addWidget(QtWidgets.QLabel(label + ":"), idx, 0)
            value_label = QtWidgets.QLabel("-")
            value_label.setObjectName(f"metric_{key}")
            grid.addWidget(value_label, idx, 1)
            self.metrics[key] = value_label

    def update_summary(self, summary: Dict[str, object]) -> None:
        if not summary:
            self.summary_label.setText("暂无数据")
            for widget in self.metrics.values():
                widget.setText("-")
            return
        total = summary.get("total_flows", 0)
        anomalies = summary.get("anomalies", 0)
        self.summary_label.setText(f"总计 {total} 条流，其中 {anomalies} 条疑似异常")
        self.metrics["total"].setText(str(total))
        self.metrics["anomalies"].setText(str(anomalies))
        self.metrics["alerts"].setText(str(summary.get("alerts", 0)))
        ts = summary.get("last_update")
        if ts:
            self.metrics["last_update"].setText(str(ts))
        else:
            self.metrics["last_update"].setText("-")

    def update_metrics(self, analysis: Optional[dict], metadata: Optional[dict]) -> None:
        metrics: Dict[str, object] = {}
        if isinstance(analysis, dict):
            metrics_candidate = analysis.get("metrics")
            if isinstance(metrics_candidate, dict):
                metrics = metrics_candidate

        def _as_float(value) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        def _as_int(value) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        ratio_value = None
        for key in ("malicious_ratio", "anomaly_ratio", "ratio"):
            ratio_value = _as_float(metrics.get(key))
            if ratio_value is not None:
                break

        total_value = None
        for key in ("total_count", "total_rows", "total", "total_flows"):
            total_value = _as_int(metrics.get(key))
            if total_value is not None:
                break

        anomaly_value = None
        for key in ("anomaly_count", "malicious_total", "malicious", "anomalies"):
            anomaly_value = _as_int(metrics.get(key))
            if anomaly_value is not None:
                break

        alerts_value = None
        if isinstance(metrics.get("drift_alerts"), dict):
            alerts_value = len(metrics.get("drift_alerts"))
        elif isinstance(metrics.get("alerts"), dict):
            alerts_value = len(metrics.get("alerts"))
        elif isinstance(metrics.get("alerts"), (list, tuple, set)):
            alerts_value = len(metrics.get("alerts"))
        else:
            alerts_value = _as_int(metrics.get("alerts"))

        timestamp_value: Optional[str] = None
        for key in ("last_update", "generated_at", "timestamp"):
            raw = metrics.get(key) if metrics else None
            if raw:
                timestamp_value = str(raw)
                break
        if timestamp_value is None and isinstance(analysis, dict):
            for key in ("last_update", "generated_at", "timestamp"):
                raw = analysis.get(key)
                if raw:
                    timestamp_value = str(raw)
                    break
        if timestamp_value is None and isinstance(metadata, dict):
            raw = metadata.get("timestamp")
            if raw:
                timestamp_value = str(raw)
        if timestamp_value is None and (total_value is not None or anomaly_value is not None):
            timestamp_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        summary_payload: Dict[str, object] = {}
        if total_value is not None:
            summary_payload["total_flows"] = total_value
        if anomaly_value is not None:
            summary_payload["anomalies"] = anomaly_value
        if alerts_value is not None and alerts_value >= 0:
            summary_payload["alerts"] = alerts_value
        if timestamp_value is not None:
            summary_payload["last_update"] = timestamp_value

        self.update_summary(summary_payload)

        training_ratio = None
        if isinstance(metadata, dict):
            for key in ("training_anomaly_ratio", "contamination"):
                training_ratio = _as_float(metadata.get(key))
                if training_ratio is not None:
                    break

        info_lines: List[str] = []
        if total_value is not None:
            info_lines.append(f"总计 {total_value} 条流")
        if anomaly_value is not None:
            info_lines.append(f"其中 {anomaly_value} 条疑似异常")
        if ratio_value is not None:
            info_lines.append(f"当前异常占比 {ratio_value:.2%}")
        if training_ratio is not None:
            info_lines.append(f"训练异常占比 {training_ratio:.2%}")
        if timestamp_value and (total_value is not None or anomaly_value is not None):
            info_lines.append(f"更新时间 {timestamp_value}")

        if info_lines:
            self.summary_label.setText("，".join(info_lines))



class AnomalyDetailDialog(QtWidgets.QDialog):
    annotation_saved = QtCore.pyqtSignal(float)

    def __init__(self, record: Dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("异常样本详情")
        self.resize(720, 520)
        self._record = record

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("双击列可复制，支持筛选"))
        self.notes_edit = QtWidgets.QLineEdit()
        self.notes_edit.setPlaceholderText("标注备注（可选）")
        header.addWidget(self.notes_edit)
        layout.addLayout(header)

        self.table = QtWidgets.QTableWidget(len(record), 2)
        self.table.setHorizontalHeaderLabels(["字段", "值"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, (key, value) in enumerate(sorted(record.items())):
            key_item = QtWidgets.QTableWidgetItem(str(key))
            val_item = QtWidgets.QTableWidgetItem(str(value))
            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)
        layout.addWidget(self.table)

        btns = QtWidgets.QDialogButtonBox()
        self.btn_mark_normal = btns.addButton("标注为正常", QtWidgets.QDialogButtonBox.ActionRole)
        self.btn_mark_anomaly = btns.addButton("标注为异常", QtWidgets.QDialogButtonBox.ActionRole)
        btns.addButton(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(btns)

        btns.rejected.connect(self.reject)
        self.btn_mark_normal.clicked.connect(lambda: self._store_label(0.0))
        self.btn_mark_anomaly.clicked.connect(lambda: self._store_label(1.0))

    def _store_label(self, value: float) -> None:
        try:
            upsert_annotation(self._record, label=value, notes=self.notes_edit.text().strip() or None)
            QtWidgets.QMessageBox.information(self, "标注成功", "已保存人工标注。")
            self.annotation_saved.emit(value)
            self.accept()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "保存失败", f"无法写入标注：{exc}")


class ConfigEditorDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        title: str = "编辑全局配置",
        text: str = "",
        path: Optional[Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 560)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        if path is not None:
            path_label = QtWidgets.QLabel(f"配置文件：{path}")
            path_label.setWordWrap(True)
            path_label.setStyleSheet("color: #4B5563;")
            layout.addWidget(path_label)

        self.editor = QtWidgets.QPlainTextEdit(self)
        try:
            fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            self.editor.setFont(fixed_font)
        except Exception:
            pass
        if hasattr(self.editor, "setTabStopDistance"):
            metrics = self.editor.fontMetrics()
            self.editor.setTabStopDistance(metrics.horizontalAdvance(" ") * 4)
        self.editor.setPlainText(text or "")
        layout.addWidget(self.editor)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def text(self) -> str:
        return self.editor.toPlainText()


class LogViewerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        log_dir: Path,
        *,
        reveal_callback: Optional[Callable[[str], None]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._log_dir = Path(log_dir)
        self._reveal_callback = reveal_callback

        self.setWindowTitle("查看日志")
        self.resize(900, 620)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("日志文件列表")
        title.setStyleSheet("font-weight: 600; color: #0F172A;")
        header.addWidget(title)
        header.addStretch(1)
        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.setObjectName("secondary")
        header.addWidget(self.refresh_btn)
        layout.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.file_list.setMinimumWidth(260)
        splitter.addWidget(self.file_list)

        self.viewer = QtWidgets.QPlainTextEdit()
        self.viewer.setReadOnly(True)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.open_btn = QtWidgets.QPushButton("在资源管理器中打开")
        footer.addWidget(self.open_btn)
        layout.addLayout(footer)

        self.refresh_btn.clicked.connect(self._reload)
        self.file_list.currentItemChanged.connect(self._display_file)
        self.open_btn.clicked.connect(self._open_in_explorer)

        self._reload()

    def _reload(self) -> None:
        self.file_list.clear()
        self.viewer.clear()

        if not self._log_dir.exists():
            self.viewer.setPlainText(f"日志目录不存在：{self._log_dir}")
            return

        candidates: Set[Path] = set()
        for pattern in ("*.log", "*.txt", "*.json", "*.yaml", "*.yml"):
            candidates.update(self._log_dir.rglob(pattern))
        if not candidates:
            candidates.update(p for p in self._log_dir.rglob("*") if p.is_file())

        entries: List[Tuple[float, Path]] = []
        for path in sorted(candidates):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, path))

        if not entries:
            self.viewer.setPlainText("日志目录中暂无文件。")
            return

        entries.sort(key=lambda item: item[0], reverse=True)

        for mtime, path in entries[:200]:
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            item = QtWidgets.QListWidgetItem(f"{path.name}  ({timestamp})")
            item.setToolTip(str(path))
            item.setData(QtCore.Qt.UserRole, str(path))
            self.file_list.addItem(item)

        if self.file_list.count():
            self.file_list.setCurrentRow(0)

    def _display_file(self, current: Optional[QtWidgets.QListWidgetItem]) -> None:
        if current is None:
            self.viewer.clear()
            return

        path = current.data(QtCore.Qt.UserRole)
        if not path:
            self.viewer.clear()
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="gb18030", errors="replace") as fh:
                    text = fh.read()
            except Exception as exc:
                self.viewer.setPlainText(f"无法读取日志文件：{exc}")
                return
        except Exception as exc:
            self.viewer.setPlainText(f"无法读取日志文件：{exc}")
            return

        max_len = 200_000
        if len(text) > max_len:
            text = text[-max_len:]
            text = "...\n" + text
        self.viewer.setPlainText(text)
        self.viewer.moveCursor(QtGui.QTextCursor.End)

    def _open_in_explorer(self) -> None:
        current = self.file_list.currentItem()
        if current is None:
            return
        path = current.data(QtCore.Qt.UserRole)
        if not path:
            return
        if self._reveal_callback:
            self._reveal_callback(path)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(path)))

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

def _resolve_data_base() -> Path:
    env = os.getenv("MALDET_DATA_DIR")
    if env and env.strip():
        base = Path(env).expanduser()
        base.mkdir(parents=True, exist_ok=True)
        return base.resolve()
    try:
        return get_path("data_dir")
    except Exception:
        fallback = Path.home() / "maldet_data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


# 仅表格预览上限（全部数据都会落盘）
PREVIEW_LIMIT_FOR_TABLE = 50

DATA_BASE = _resolve_data_base()
PATHS = get_paths(
    {
        "split": "split_dir",
        "csv_info": "csv_info_dir",
        "csv_feature": "csv_feature_dir",
        "csv_preprocess": "csv_preprocess_dir",
        "models": "models_dir",
        "results_analysis": "results_analysis_dir",
        "results_pred": "results_prediction_dir",
        "results_abnormal": "results_abnormal_dir",
        "results": "results_dir",
        "logs": "logs_dir",
        "settings": "settings_dir",
    }
)
if "results" in PATHS and "results_analysis" not in PATHS:
    PATHS["results_analysis"] = PATHS["results"] / "analysis"
if "results" in PATHS and "results_pred" not in PATHS:
    PATHS["results_pred"] = PATHS["results"] / "modelprediction"
if "results" in PATHS and "results_abnormal" not in PATHS:
    PATHS["results_abnormal"] = PATHS["results"] / "abnormal"
for key in (
    "split",
    "csv_info",
    "csv_feature",
    "csv_preprocess",
    "models",
    "results_analysis",
    "results_pred",
    "results_abnormal",
    "settings",
):
    if key not in PATHS:
        PATHS[key] = DATA_BASE / {
            "split": "split",
            "csv_info": "CSV/info",
            "csv_feature": "CSV/feature",
            "csv_preprocess": "CSV/DP",
            "models": "models",
            "results_analysis": "results/analysis",
            "results_pred": "results/modelprediction",
            "results_abnormal": "results/abnormal",
            "settings": "settings",
        }[key]
    PATHS[key].mkdir(parents=True, exist_ok=True)

default_logs = PATHS.get("logs", DATA_BASE / "logs")
logs_env = os.getenv("MALDET_LOG_DIR")
LOGS_DIR = Path(logs_env).expanduser().resolve() if logs_env else Path(default_logs).expanduser().resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_DIR = PATHS.get("settings", DATA_BASE / "settings")
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = SETTINGS_DIR / "settings.json"

MODEL_SCHEMA_VERSION = "2025.10"


def _feature_order_from_metadata(metadata: dict) -> List[str]:
    if not isinstance(metadata, dict):
        return []

    candidate = (
        metadata.get("feature_names_in")
        or metadata.get("feature_order")
        or (metadata.get("preprocessor") or {}).get("feature_order")
        or (metadata.get("preprocessor") or {}).get("input_columns")
        or metadata.get("feature_columns")
    )

    if candidate is None:
        return []

    if isinstance(candidate, (list, tuple, set)):
        values = list(candidate)
    else:
        try:
            values = list(candidate)
        except TypeError:
            values = [candidate]

    return [str(col) for col in values if str(col)]


def _align_input_features(
    df: "pd.DataFrame",
    metadata: dict,
    *,
    strict: bool = False,
    allow_extra: Optional[Collection[str]] = None,
) -> tuple["pd.DataFrame", dict]:
    info: dict[str, object] = {}
    if not isinstance(metadata, dict):
        raise ValueError("模型缺少有效的元数据。")

    schema_version = metadata.get("schema_version")
    if schema_version is None:
        raise ValueError("模型元数据缺少 schema_version 字段，请重新训练模型。")
    info["schema_version"] = schema_version

    feature_order = _feature_order_from_metadata(metadata)
    if not feature_order:
        raise ValueError("模型元数据缺少 feature_order 描述，无法校验列。")

    default_fill = 0.0
    allow_set = {str(col) for col in allow_extra or []}
    # 始终允许内置的元信息列及自动生成的临时列
    allow_set.update(TRAIN_META_COLUMNS)
    allow_set.update({
        "label",
        "labels",
        "ground_truth",
        "attack",
        "attacks",
        "is_attack",
        "is_malicious",
        "malicious",
        "score",
        "is_anomaly",
        "prediction",
        "anomaly_score",
        "risk_score",
        "vote_ratio",
    })
    fill_values = {}
    preprocessor_meta = metadata.get("preprocessor") if isinstance(metadata.get("preprocessor"), dict) else None
    if isinstance(metadata.get("fill_values"), dict):
        fill_values.update({str(k): v for k, v in metadata["fill_values"].items()})
    if isinstance(preprocessor_meta, dict):
        default_fill = float(preprocessor_meta.get("fill_value", default_fill))
        fill_values.update({str(k): v for k, v in (preprocessor_meta.get("fill_values") or {}).items()})
    if "fill_value" in metadata:
        try:
            default_fill = float(metadata.get("fill_value", default_fill))
        except (TypeError, ValueError):
            default_fill = float(default_fill)

    working = df.copy()
    missing_columns: List[str] = []
    for column in feature_order:
        if column in working.columns:
            continue
        fill_value = fill_values.get(column, default_fill)
        working[column] = fill_value
        missing_columns.append(column)

    dropped_columns = [col for col in working.columns if col not in feature_order]
    extra_columns: List[str] = []
    ignored_columns: List[str] = []
    for column in dropped_columns:
        ignored_columns.append(column)
        column_norm = str(column)
        if column_norm in allow_set:
            continue
        if column_norm.startswith("__") or column_norm.lower().startswith("unnamed:"):
            continue
        extra_columns.append(column_norm)
    info["missing_filled"] = missing_columns
    info["extra_columns"] = extra_columns
    info["feature_order"] = feature_order
    info["ignored_columns"] = ignored_columns

    if strict and (missing_columns or extra_columns):
        parts: List[str] = []
        if missing_columns:
            sample = ", ".join(missing_columns[:10])
            more = " ..." if len(missing_columns) > 10 else ""
            parts.append(f"缺少列: {sample}{more}")
        if extra_columns:
            sample = ", ".join(extra_columns[:10])
            more = " ..." if len(extra_columns) > 10 else ""
            parts.append(f"多余列: {sample}{more}")
        raise ValueError("特征列与训练时不一致，请选择正确的特征CSV。\n" + "\n".join(parts))

    aligned = working.loc[:, feature_order].copy()
    numeric_cols = aligned.select_dtypes(include=["number"]).columns
    if len(numeric_cols):
        aligned.loc[:, numeric_cols] = aligned.loc[:, numeric_cols].astype("float32", copy=False)

    return aligned, info


# =============== 主 UI ===============
class Ui_MainWindow(object):
    # --------- 基本结构 ----------
    def setupUi(self, MainWindow):
        self._main_window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 840)
        MainWindow.setStyleSheet(APP_STYLE)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.title_bar = QtWidgets.QFrame(self.centralwidget)
        self.title_bar.setObjectName("TitleBar")
        self.title_bar.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.title_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        self.title_bar.setFixedHeight(56)
        title_layout = QtWidgets.QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 10, 0, 10)
        title_layout.setSpacing(0)
        self.page_title = QtWidgets.QLabel("恶意流量检测系统 — 主功能页面", self.title_bar)
        self.page_title.setObjectName("pageTitle")
        self.page_title.setAlignment(QtCore.Qt.AlignCenter)
        self.page_title.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        title_layout.addWidget(self.page_title)
        self.main_layout.addWidget(self.title_bar, 0)

        self.content_area = QtWidgets.QWidget(self.centralwidget)
        self.content_layout = QtWidgets.QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(12, 10, 12, 10)
        self.content_layout.setSpacing(8)
        self.main_layout.addWidget(self.content_area, 1)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.content_area)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(2)
        self.content_layout.addWidget(self.splitter, 1)
        self.main_layout.setStretch(0, 0)
        self.main_layout.setStretch(1, 1)

        # 左侧滚动区
        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.left_container = QtWidgets.QWidget()
        self.left_scroll.setWidget(self.left_container)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(8, 8, 8, 8)
        self.left_layout.setSpacing(8)

        self._build_path_bar()
        self._build_param_panel()
        self._build_center_tabs()
        self._build_paging_toolbar()
        self._build_output_list()

        self.splitter.addWidget(self.left_scroll)

        # 右侧按钮列
        self._build_right_panel()
        screen = QtWidgets.QApplication.primaryScreen()
        screen_width = 0
        if screen is not None:
            geometry = screen.availableGeometry()
            screen_width = geometry.width()
        if screen_width <= 0:
            screen_width = max(MainWindow.width(), 1400)
        left_width = int(screen_width * 0.72)
        right_width = int(screen_width * 0.28)
        if left_width <= 0 or right_width <= 0:
            left_width, right_width = 720, 280
        self.splitter.setSizes([left_width, right_width])
        self.splitter.setStretchFactor(0, 8)
        self.splitter.setStretchFactor(1, 3)

        MainWindow.setCentralWidget(self.centralwidget)
        self._build_status_bar(MainWindow)
        self._update_status_message("@2025 恶意流量检测系统")
        self._bind_signals()

        # 状态缓存
        self._last_preview_df: Optional["pd.DataFrame"] = None
        self._last_out_csv: Optional[str] = None
        self._analysis_summary: Optional[dict] = None
        self._latest_prediction_summary: Optional[dict] = None
        self._auto_analyze_tip_shown: bool = False

        # 分页状态
        self._csv_paged_path: Optional[str] = None
        self._csv_total_rows: Optional[int] = None
        self._csv_current_page: int = 1

        # worker
        self.worker: Optional[FunctionThread] = None
        self.fe_worker: Optional[FunctionThread] = None
        self.dir_fe_worker: Optional[FunctionThread] = None
        self.preprocess_worker: Optional[FunctionThread] = None
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        self._running_tasks: Set[BackgroundTask] = set()

        # 用户偏好
        self._settings = AppSettings(SETTINGS_PATH)
        self._loading_settings = False
        self._settings_ready = False
        self._apply_saved_preferences()

        self._model_registry: Dict[str, dict] = {}
        self._selected_model_key: Optional[str] = None
        self._selected_metadata: Optional[dict] = None
        self._selected_metadata_path: Optional[str] = None
        self._selected_pipeline_path: Optional[str] = None
        self._online_worker: Optional[OnlineDetectionWorker] = None
        self._online_output_dir: Optional[str] = None

        self._refresh_model_versions()
        self._update_plugin_summary()
        self.dashboard.update_metrics(None, None)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def _parent_widget(self):
        parent = getattr(self, "_main_window", None)
        if isinstance(parent, QtWidgets.QWidget):
            return parent
        central = getattr(self, "centralwidget", None)
        if isinstance(central, QtWidgets.QWidget):
            return central
        return None

    def _build_path_bar(self):
        self.file_group = QtWidgets.QGroupBox("数据源选择")
        fg = QtWidgets.QVBoxLayout(self.file_group)
        fg.setContentsMargins(12, 10, 12, 10)
        fg.setSpacing(10)

        hint = QtWidgets.QLabel("选择流量文件或目录 (.pcap/.pcapng 或 split 目录)")
        hint.setWordWrap(True)
        fg.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.path_tool_button = QtWidgets.QToolButton(self.file_group)
        self.path_tool_button.setIcon(self.file_group.style().standardIcon(QtWidgets.QStyle.SP_DirIcon))
        self.path_tool_button.setIconSize(QtCore.QSize(20, 20))
        self.path_tool_button.setAutoRaise(True)
        self.path_tool_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.path_tool_button.setToolTip("浏览文件或目录")
        row.addWidget(self.path_tool_button)

        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("请选择文件或目录路径")
        self.file_edit.setClearButtonEnabled(True)
        row.addWidget(self.file_edit, 1)
        row.addStretch(1)

        self.btn_pick_file = QtWidgets.QPushButton("选文件")
        self.btn_pick_dir = QtWidgets.QPushButton("选目录")
        self.btn_browse = QtWidgets.QPushButton("浏览")
        for btn in (self.btn_pick_file, self.btn_pick_dir, self.btn_browse):
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        row.addWidget(self.btn_pick_file)
        row.addWidget(self.btn_pick_dir)
        row.addWidget(self.btn_browse)

        fg.addLayout(row)
        self.left_layout.addWidget(self.file_group)

    def _build_param_panel(self):
        self.param_group = QtWidgets.QGroupBox("查看流量信息参数")
        pg = QtWidgets.QFormLayout()
        pg.setContentsMargins(12, 10, 12, 10)
        pg.setHorizontalSpacing(24)
        pg.setVerticalSpacing(12)
        pg.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        pg.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.mode_combo = QtWidgets.QComboBox()
        self._mode_map = {
            "自动(文件=单文件/目录=全部)": "auto",
            "单文件": "file",
            "整个目录(从上到下)": "all",
            "分批(按文件名排序)": "batch"
        }
        self.mode_combo.addItems(list(self._mode_map.keys()))
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 99999)
        self.batch_spin.setValue(10)
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, 10**9)
        self.start_spin.setSingleStep(10)
        self.start_spin.setValue(0)
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(8)
        self.proto_combo = QtWidgets.QComboBox()
        self.proto_combo.addItems(["TCP+UDP", "仅TCP", "仅UDP"])
        self.whitelist_edit = QtWidgets.QLineEdit()
        self.whitelist_edit.setPlaceholderText("例: 80,443,53")
        self.blacklist_edit = QtWidgets.QLineEdit()
        self.blacklist_edit.setPlaceholderText("例: 135,137,138,139")

        self.fast_check = QtWidgets.QCheckBox("极速模式（UI开关保留）")
        self.fast_check.setChecked(True)
        self.fast_check.setMinimumHeight(38)

        for widget in (
            self.mode_combo,
            self.batch_spin,
            self.start_spin,
            self.workers_spin,
            self.proto_combo,
            self.whitelist_edit,
            self.blacklist_edit,
        ):
            widget.setMinimumHeight(38)

        self.btn_prev = QtWidgets.QPushButton("上一批")
        self.btn_next = QtWidgets.QPushButton("下一批")
        for btn in (self.btn_prev, self.btn_next):
            btn.setMinimumWidth(120)
        nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(12)
        nav_layout.addWidget(self.btn_prev, 1)
        nav_layout.addWidget(self.btn_next, 1)

        for btn in (self.btn_prev, self.btn_next):
            btn.setObjectName("secondary")

        pg.addRow("处理模式：", self.mode_combo)
        pg.addRow("批处理数量", self.batch_spin)
        pg.addRow("起始索引", self.start_spin)
        pg.addRow("并发数", self.workers_spin)
        pg.addRow("协议", self.proto_combo)
        pg.addRow("端口白名单", self.whitelist_edit)
        pg.addRow("端口黑名单", self.blacklist_edit)
        pg.addRow("极速模式", self.fast_check)
        pg.addRow(nav_widget)

        for field in (
            self.mode_combo,
            self.batch_spin,
            self.start_spin,
            self.workers_spin,
            self.proto_combo,
            self.whitelist_edit,
            self.blacklist_edit,
            self.fast_check,
        ):
            label = pg.labelForField(field)
            if isinstance(label, QtWidgets.QLabel):
                label.setMinimumWidth(140)
                label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.param_group.setLayout(pg)
        self.left_layout.addWidget(self.param_group)
        self.mode_combo.currentIndexChanged.connect(self._update_batch_controls)
        self._update_batch_controls()

    def _build_center_tabs(self):
        self.display_tabs = QtWidgets.QTabWidget(self.left_container)

        self.results_widget = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(self.results_widget)
        rl.setContentsMargins(12, 10, 12, 10)
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        rl.addWidget(self.results_text)
        self.display_tabs.addTab(self.results_widget, "结果（文本）")

        self.table_widget = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(self.table_widget)
        tl.setContentsMargins(12, 10, 12, 10)
        self.table_view = QtWidgets.QTableView()
        self.table_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.verticalHeader().setDefaultSectionSize(28)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table_model = None  # type: Optional[PandasFrameModel]
        self._table_proxy = None  # type: Optional[QtCore.QSortFilterProxyModel]
        self.table_widget.setMinimumHeight(22 * 16 + 40)
        tl.addWidget(self.table_view)
        self.display_tabs.addTab(self.table_widget, "流量表格")

        self.left_layout.addWidget(self.display_tabs, stretch=3)

    def _build_paging_toolbar(self):
        bar = QtWidgets.QWidget(self.left_container)
        hb = QtWidgets.QHBoxLayout(bar)
        hb.setContentsMargins(12, 0, 12, 0)
        hb.setSpacing(10)

        self.btn_page_prev = QtWidgets.QPushButton("上一页")
        self.btn_page_next = QtWidgets.QPushButton("下一页")
        self.page_info = QtWidgets.QLabel("第 0/0 页")
        self.page_size_label = QtWidgets.QLabel("每页行数：")
        self.page_size_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        self.page_size_spin = QtWidgets.QSpinBox()
        self.page_size_spin.setRange(20, 200000)
        self.page_size_spin.setSingleStep(10)
        self.page_size_spin.setValue(50)
        self.btn_show_all = QtWidgets.QPushButton("显示全部（可能较慢）")

        for btn in (self.btn_page_prev, self.btn_page_next, self.btn_show_all):
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            btn.setObjectName("secondary")
            btn.setMinimumWidth(120)

        self.page_size_spin.setMinimumHeight(38)
        self.page_info.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        hb.addWidget(self.btn_page_prev); hb.addWidget(self.btn_page_next)
        hb.addSpacing(8); hb.addWidget(self.page_info); hb.addStretch(1)
        hb.addWidget(self.page_size_label); hb.addWidget(self.page_size_spin)
        hb.addSpacing(8); hb.addWidget(self.btn_show_all)

        self.left_layout.addWidget(bar)

    def _build_output_list(self):
        self.out_group = QtWidgets.QGroupBox("输出文件（双击打开所在目录，右键复制路径）")
        og = QtWidgets.QVBoxLayout(self.out_group)
        og.setContentsMargins(12, 10, 12, 10)
        self.output_list = QtWidgets.QListWidget()
        self.output_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        og.addWidget(self.output_list)
        self.left_layout.addWidget(self.out_group, stretch=1)

    def _build_status_bar(self, MainWindow: QtWidgets.QMainWindow) -> None:
        self.status_bar = QtWidgets.QStatusBar(MainWindow)
        self.status_bar.setObjectName("MainStatusBar")
        self.status_bar.setSizeGripEnabled(False)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.status_bar.addPermanentWidget(self.status_label, 1)
        MainWindow.setStatusBar(self.status_bar)

    def _action_buttons(self) -> List[QtWidgets.QPushButton]:
        return [
            btn
            for btn in (
                getattr(self, "btn_view", None),
                getattr(self, "btn_fe", None),
                getattr(self, "btn_vector", None),
                getattr(self, "btn_train", None),
                getattr(self, "btn_analysis", None),
                getattr(self, "btn_predict", None),
                getattr(self, "btn_export", None),
            )
            if btn is not None
        ]

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._action_buttons():
            btn.setEnabled(enabled)

    def _start_background_task(
        self,
        task: BackgroundTask,
        finished_cb,
        error_cb,
        progress_cb=None,
    ) -> None:
        self._running_tasks.add(task)
        if finished_cb:
            task.finished.connect(finished_cb)
        if error_cb:
            task.error.connect(error_cb)
        if progress_cb:
            task.progress.connect(progress_cb)

        def _cleanup(*_args):
            self._running_tasks.discard(task)

        task.finished.connect(_cleanup)
        task.error.connect(_cleanup)
        self.thread_pool.start(task)

    def _collect_pipeline_config(self) -> Dict[str, bool]:
        return {}

    def _update_status_message(self, message: Optional[str] = None) -> None:
        label = getattr(self, "status_label", None)
        if label is None:
            return
        base = f"数据目录: {DATA_BASE}"
        if message:
            label.setText(f"{message} | {base}")
        else:
            label.setText(base)

    def _create_collapsible_group(
        self,
        title: str,
        layout_cls=QtWidgets.QVBoxLayout,
        *,
        spacing: int = 8,
    ):
        group = QtWidgets.QGroupBox(title)
        group.setCheckable(True)
        group.setChecked(True)

        content_widget = QtWidgets.QWidget(group)
        inner_layout = layout_cls()
        if isinstance(inner_layout, QtWidgets.QFormLayout):
            inner_layout.setContentsMargins(0, 0, 0, 0)
            inner_layout.setHorizontalSpacing(24)
            inner_layout.setVerticalSpacing(spacing)
            inner_layout.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            inner_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        else:
            inner_layout.setContentsMargins(0, 0, 0, 0)
            inner_layout.setSpacing(spacing)
        content_widget.setLayout(inner_layout)

        wrapper_layout = QtWidgets.QVBoxLayout()
        wrapper_layout.setContentsMargins(12, 10, 12, 10)
        wrapper_layout.setSpacing(spacing)
        wrapper_layout.addWidget(content_widget)
        group.setLayout(wrapper_layout)

        def _toggle(checked: bool) -> None:
            content_widget.setVisible(checked)
            content_widget.setEnabled(checked)

        group.toggled.connect(_toggle)
        _toggle(True)
        return group, inner_layout

    def _build_right_panel(self):
        self.right_scroll = QtWidgets.QScrollArea(self.centralwidget)
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.right_frame = QtWidgets.QFrame()
        self.right_frame.setObjectName("RightPanelFrame")
        self.right_scroll.setWidget(self.right_frame)

        self.right_layout = QtWidgets.QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(8, 8, 8, 8)
        self.right_layout.setSpacing(10)

        self.right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.btn_view = QtWidgets.QPushButton("查看流量信息")
        self.btn_fe = QtWidgets.QPushButton("提取特征")
        self.btn_vector = QtWidgets.QPushButton("数据预处理")
        self.btn_train = QtWidgets.QPushButton("训练模型")
        self.btn_analysis = QtWidgets.QPushButton("运行分析")
        self.btn_predict = QtWidgets.QPushButton("加载模型预测")
        self.btn_export = QtWidgets.QPushButton("导出结果（异常）")
        self.btn_open_results = QtWidgets.QPushButton("打开结果目录")
        self.btn_view_logs = QtWidgets.QPushButton("查看日志")
        self.btn_clear = QtWidgets.QPushButton("清空显示")
        self.btn_export_report = QtWidgets.QPushButton("导出 PDF 报告")
        self.btn_config_editor = QtWidgets.QPushButton("编辑全局配置")
        self.btn_online_toggle = QtWidgets.QPushButton("开启在线检测")
        for btn in (
            self.btn_view,
            self.btn_fe,
            self.btn_vector,
            self.btn_train,
            self.btn_analysis,
            self.btn_predict,
            self.btn_export,
            self.btn_open_results,
            self.btn_view_logs,
            self.btn_clear,
            self.btn_export_report,
            self.btn_config_editor,
            self.btn_online_toggle,
        ):
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        data_group, data_layout = self._create_collapsible_group("数据阶段")
        data_layout.addWidget(self.btn_view)
        data_layout.addWidget(self.btn_fe)
        data_layout.addWidget(self.btn_vector)
        data_layout.addStretch(1)
        self.right_layout.addWidget(data_group)

        model_group, model_layout = self._create_collapsible_group("模型阶段")
        model_layout.addWidget(self.btn_train)
        model_layout.addWidget(self.btn_predict)
        model_layout.addWidget(self.btn_analysis)
        model_layout.addStretch(1)
        self.right_layout.addWidget(model_group)

        output_group, output_layout = self._create_collapsible_group("输出管理")
        output_layout.addWidget(self.btn_export)
        output_layout.addWidget(self.btn_open_results)
        output_layout.addWidget(self.btn_view_logs)
        output_layout.addStretch(1)
        self.right_layout.addWidget(output_group)

        utility_group, utility_layout = self._create_collapsible_group("系统与维护")
        utility_layout.addWidget(self.btn_export_report)
        utility_layout.addWidget(self.btn_config_editor)
        utility_layout.addWidget(self.btn_online_toggle)
        self.online_status_label = QtWidgets.QLabel("在线检测未启动")
        self.online_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.online_status_label.setObjectName("OnlineStatusLabel")
        utility_layout.addWidget(self.online_status_label)
        utility_layout.addWidget(self.btn_clear)
        utility_layout.addStretch(1)
        self.right_layout.addWidget(utility_group)

        self.dashboard = ResultsDashboard()
        dashboard_group, dashboard_layout = self._create_collapsible_group("训练监控", spacing=10)
        dashboard_layout.addWidget(self.dashboard)
        self.right_layout.addWidget(dashboard_group)

        self.model_group, mg_layout = self._create_collapsible_group("模型版本管理")
        model_row = QtWidgets.QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(8)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumHeight(38)
        self.model_refresh_btn = QtWidgets.QPushButton("刷新")
        self.model_refresh_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.model_refresh_btn.setObjectName("secondary")
        self.model_refresh_btn.setMinimumWidth(120)
        model_row.addWidget(self.model_combo)
        model_row.addWidget(self.model_refresh_btn)
        mg_layout.addLayout(model_row)
        self.model_info_label = QtWidgets.QLabel("尚未加载模型")
        self.model_info_label.setWordWrap(True)
        mg_layout.addWidget(self.model_info_label)
        self.right_layout.addWidget(self.model_group)

        self.plugin_group, plugin_layout = self._create_collapsible_group("特征插件", spacing=6)
        self.plugin_label = QtWidgets.QLabel("未发现插件")
        self.plugin_label.setWordWrap(True)
        plugin_layout.addWidget(self.plugin_label)
        plugin_layout.addStretch(1)
        self.right_layout.addWidget(self.plugin_group)

        self.right_layout.addStretch(1)
        self.right_frame.setMinimumWidth(240)

        self.splitter.addWidget(self.right_scroll)

    def _update_plugin_summary(self):
        if not hasattr(self, "plugin_label"):
            return
        try:
            info = get_loaded_plugin_info()
        except Exception:
            info = []
        if not info:
            self.plugin_label.setText("未发现插件")
            return
        lines = []
        for item in info:
            module = item.get("module")
            extractors = item.get("extractors")
            if isinstance(extractors, (list, tuple)) and extractors:
                desc = ", ".join(str(e) for e in extractors)
            else:
                desc = "未公开特征"
            lines.append(f"{module}: {desc}")
        self.plugin_label.setText("\n".join(lines))

    def _active_config_path(self) -> Path:
        env_value = os.getenv("MALDET_CONFIG")
        if env_value and str(env_value).strip():
            candidate = Path(str(env_value)).expanduser()
            if candidate.is_dir():
                for name in ("config.yaml", "default.yaml", "settings.yaml"):
                    probe = candidate / name
                    if probe.exists():
                        return probe
                return candidate / "config.yaml"
            return candidate
        default_path = project_root() / "config" / "default.yaml"
        return default_path

    def _open_config_editor_dialog(self) -> None:
        parent_widget = self._parent_widget()
        try:
            config_path = self._active_config_path()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "定位失败", f"无法确定配置文件路径：{exc}")
            return

        config_path_parent = config_path.parent

        text = ""
        warning: Optional[str] = None

        try:
            if config_path.exists():
                try:
                    text = config_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    with open(config_path, "r", encoding="utf-8", errors="replace") as fh:
                        text = fh.read()
                    warning = "检测到非 UTF-8 字符，已自动替换为占位符显示。"
            else:
                sample = load_config() or {}
                if isinstance(sample, dict) and sample:
                    dump_kwargs = {"allow_unicode": True}
                    try:
                        text = yaml.safe_dump(sample, sort_keys=False, **dump_kwargs)
                    except TypeError:
                        text = yaml.safe_dump(sample, **dump_kwargs)
                if not text:
                    text = "# 在此编写全局配置（YAML 格式）\npaths:\n  data_dir: data\n"
        except Exception as exc:
            warning = f"无法加载配置文件：{exc}"
            text = "# 在此编写全局配置（YAML 格式）\npaths:\n  data_dir: data\n"

        try:
            dialog = ConfigEditorDialog(
                title="编辑全局配置",
                text=text,
                path=config_path,
                parent=parent_widget,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "初始化失败", f"无法打开配置编辑器：{exc}")
            return

        if warning:
            QtWidgets.QMessageBox.warning(parent_widget, "读取提示", warning)

        result = dialog.exec_()
        if result != QtWidgets.QDialog.Accepted:
            return

        new_text = dialog.text().strip()
        if not new_text:
            QtWidgets.QMessageBox.warning(parent_widget, "内容为空", "配置内容不能为空。")
            return

        try:
            parsed = yaml.safe_load(new_text) or {}
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "格式错误", f"配置内容不是有效的 YAML：{exc}")
            return

        if not isinstance(parsed, dict):
            QtWidgets.QMessageBox.critical(parent_widget, "格式错误", "配置文件的根节点必须是一个字典。")
            return

        try:
            config_path_parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write(new_text)
                if not new_text.endswith("\n"):
                    fh.write("\n")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "保存失败", f"无法写入配置文件：{exc}")
            return

        try:
            if hasattr(load_config, "cache_clear"):
                load_config.cache_clear()
        except Exception:
            pass

        try:
            refreshed = get_paths(
                {
                    "split": "split_dir",
                    "csv_info": "csv_info_dir",
                    "csv_feature": "csv_feature_dir",
                    "csv_preprocess": "csv_preprocess_dir",
                    "models": "models_dir",
                    "results_analysis": "results_analysis_dir",
                    "results_pred": "results_prediction_dir",
                    "results_abnormal": "results_abnormal_dir",
                    "results": "results_dir",
                    "logs": "logs_dir",
                    "settings": "settings_dir",
                }
            )
            if isinstance(refreshed, dict):
                PATHS.update(refreshed)
        except Exception:
            pass

        self._update_plugin_summary()
        self._refresh_model_versions()
        self._update_status_message()
        self.display_result(f"[INFO] 配置已更新并保存至：{config_path}")

    def _toggle_online_detection(self) -> None:
        parent_widget = self._parent_widget()
        if pd is None:
            QtWidgets.QMessageBox.warning(parent_widget, "缺少依赖", "当前环境未安装 pandas，无法执行在线检测。")
            return

        worker = getattr(self, "_online_worker", None)
        if worker and worker.isRunning():
            self.display_result("[INFO] 正在停止在线检测...")
            worker.stop()
            self.btn_online_toggle.setEnabled(False)
            return

        config = load_config() or {}
        online_cfg = config.get("online_detection") if isinstance(config, dict) else {}
        watch_dir = online_cfg.get("watch_dir") or os.path.join(str(DATA_BASE), "live")
        output_dir = online_cfg.get("output_dir") or os.path.join(self._prediction_out_dir(), "online")

        poll_seconds = None
        ui_cfg = config.get("ui") if isinstance(config, dict) else {}
        if isinstance(ui_cfg, dict):
            poll_seconds = ui_cfg.get("online_detection_poll_seconds")
        if poll_seconds is None:
            poll_seconds = online_cfg.get("poll_seconds", 5)
        try:
            poll_seconds = max(1, int(poll_seconds))
        except Exception:
            poll_seconds = 5

        self._online_output_dir = output_dir
        worker = OnlineDetectionWorker(watch_dir, poll_seconds=poll_seconds)
        worker.new_file.connect(self._on_online_file_detected)
        worker.status.connect(self._on_online_status)
        worker.error.connect(self._on_online_error)
        worker.stopped.connect(self._on_online_stopped)
        self._online_worker = worker
        self.btn_online_toggle.setText("停止在线检测")
        self.online_status_label.setText(f"监控目录：{watch_dir}")
        worker.start()
        self.display_result(f"[INFO] 在线检测已启动，监控目录：{watch_dir}")

    def _on_online_status(self, message: str) -> None:
        if message:
            self.online_status_label.setText(message)

    def _on_online_error(self, message: str) -> None:
        parent_widget = self._parent_widget()
        self.display_result(f"[错误] 在线检测：{message}")
        QtWidgets.QMessageBox.warning(parent_widget, "在线检测错误", message)

    def _on_online_stopped(self) -> None:
        self._online_worker = None
        self._online_output_dir = None
        self.btn_online_toggle.setEnabled(True)
        self.btn_online_toggle.setText("开启在线检测")
        self.online_status_label.setText("在线检测未启动")

    def _on_online_file_detected(self, path: str) -> None:
        if not path:
            return
        self.display_result(f"[INFO] 在线检测发现新文件：{path}")
        metadata = self._selected_metadata if isinstance(self._selected_metadata, dict) else None
        output_dir = self._online_output_dir or self._prediction_out_dir()

        basename = os.path.basename(path)

        def _progress(value: int) -> None:
            self.online_status_label.setText(f"处理 {basename} ({int(value)}%)")

        task = BackgroundTask(
            self._process_online_pcap,
            path,
            output_dir=output_dir,
            metadata=metadata,
        )
        self._start_background_task(
            task,
            self._on_online_prediction_finished,
            self._on_online_prediction_error,
            _progress,
        )

    def _process_online_pcap(
        self,
        pcap_path: str,
        *,
        output_dir: str,
        metadata: Optional[dict],
        progress_cb=None,
    ) -> dict:
        if pd is None:
            raise RuntimeError("pandas 未安装，无法执行在线检测。")

        base = os.path.splitext(os.path.basename(pcap_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        feature_dir = os.path.join(output_dir, "features")
        os.makedirs(feature_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_csv = os.path.join(feature_dir, f"{base}_features_{stamp}.csv")

        def _map_progress(value: int) -> None:
            if not progress_cb:
                return
            try:
                scaled = int(max(0, min(40, float(value) * 0.4)))
            except Exception:
                scaled = 0
            progress_cb(scaled)

        fast_mode = bool((load_config() or {}).get("online_detection", {}).get("fast", True))
        fe_single(pcap_path, feature_csv, progress_cb=_map_progress, fast=fast_mode)
        if progress_cb:
            progress_cb(55)

        try:
            df = read_csv_flexible(feature_csv)
        except Exception as exc:
            raise RuntimeError(f"读取特征失败：{exc}") from exc

        if df.empty:
            raise RuntimeError("提取的特征为空，无法进行检测。")

        result = self._predict_dataframe(
            df,
            source_name=base,
            output_dir=output_dir,
            metadata_override=metadata,
            silent=True,
        )
        if progress_cb:
            progress_cb(100)

        payload = {
            "pcap_path": pcap_path,
            "feature_csv": feature_csv,
            "prediction": result,
        }
        return payload

    def _on_online_prediction_finished(self, payload) -> None:
        if not isinstance(payload, dict):
            return

        prediction = payload.get("prediction") if isinstance(payload.get("prediction"), dict) else {}
        messages = prediction.get("messages") or []
        if messages:
            self.display_result("\n".join(f"[在线检测] {line}" for line in messages))

        output_csv = prediction.get("output_csv")
        if output_csv and os.path.exists(output_csv):
            self._add_output(output_csv)
            self._last_out_csv = output_csv
            self._open_csv_paged(output_csv)
            self._auto_analyze(output_csv)

        dataframe = prediction.get("dataframe")
        if isinstance(dataframe, pd.DataFrame):
            self.populate_table_from_df(dataframe)

        metadata = prediction.get("metadata") if isinstance(prediction.get("metadata"), dict) else self._selected_metadata
        ratio = prediction.get("ratio")
        analysis_stub = None
        if ratio is not None:
            analysis_stub = {
                "metrics": {
                    "malicious_ratio": ratio,
                    "anomaly_count": int(prediction.get("malicious", 0)),
                    "total_count": int(prediction.get("total", 0)),
                }
            }
        self.dashboard.update_metrics(analysis_stub, metadata if isinstance(metadata, dict) else None)

        snapshot = dict(prediction)
        if isinstance(dataframe, pd.DataFrame):
            snapshot["dataframe"] = dataframe
        snapshot["source_pcap"] = payload.get("pcap_path")
        self._latest_prediction_summary = snapshot

        basename = os.path.basename(payload.get("pcap_path") or "")
        self.online_status_label.setText(f"最近完成：{basename}" if basename else "在线检测运行中")
        self.display_result(f"[INFO] 在线检测完成：{payload.get('pcap_path')}")

    def _on_online_prediction_error(self, message: str) -> None:
        parent_widget = self._parent_widget()
        self.display_result(f"[错误] 在线检测任务失败：{message}")
        QtWidgets.QMessageBox.warning(parent_widget, "在线检测任务失败", message)
        self.online_status_label.setText(f"检测任务失败：{message}")

    def _bind_signals(self):
        self.btn_pick_file.clicked.connect(self._choose_file)
        self.btn_pick_dir.clicked.connect(self._choose_dir)
        self.btn_browse.clicked.connect(self._browse_compat)
        self.path_tool_button.clicked.connect(self._browse_compat)

        self.btn_prev.clicked.connect(self._on_prev_batch)
        self.btn_next.clicked.connect(self._on_next_batch)

        # 所有功能均使用顶部路径
        self.btn_view.clicked.connect(self._on_view_info)
        self.btn_export.clicked.connect(self._on_export_results)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_fe.clicked.connect(self._on_extract_features)
        self.btn_vector.clicked.connect(self._on_preprocess_features)
        self.btn_train.clicked.connect(self._on_train_model)
        self.btn_analysis.clicked.connect(self._on_run_analysis)
        self.btn_predict.clicked.connect(self._on_predict)
        self.btn_open_results.clicked.connect(self._open_results_dir)
        self.btn_view_logs.clicked.connect(self._open_logs_dir)
        self.btn_export_report.clicked.connect(self._on_export_report)
        self.btn_config_editor.clicked.connect(self._open_config_editor_dialog)
        self.btn_online_toggle.clicked.connect(self._toggle_online_detection)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.model_refresh_btn.clicked.connect(self._refresh_model_versions)

        self.btn_page_prev.clicked.connect(self._on_page_prev)
        self.btn_page_next.clicked.connect(self._on_page_next)
        self.page_size_spin.valueChanged.connect(self._on_page_size_changed)
        self.btn_show_all.clicked.connect(self._show_full_preview)

        self.output_list.customContextMenuRequested.connect(self._on_output_ctx_menu)
        self.output_list.itemDoubleClicked.connect(self._on_output_double_click)
        self.table_view.doubleClicked.connect(self._on_table_double_click)

    # --------- 路径小工具 ----------
    _PATH_RESOLVERS = {
        "_default_split_dir": "split",
        "_default_results_dir": "results_analysis",
        "_default_models_dir": "models",
        "_default_csv_info_dir": "csv_info",
        "_default_csv_feature_dir": "csv_feature",
        "_analysis_out_dir": "results_analysis",
        "_prediction_out_dir": "results_pred",
        "_abnormal_out_dir": "results_abnormal",
        "_preprocess_out_dir": "csv_preprocess",
    }

    def _project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def __getattr__(self, name):
        key = self._PATH_RESOLVERS.get(name)
        if key:
            return lambda key=key: str(PATHS[key])
        raise AttributeError(name)

    def _refresh_model_versions(self):
        if not hasattr(self, "model_combo"):
            return
        models_dir = Path(self._default_models_dir())
        registry: Dict[str, dict] = {}
        candidates: List[Path] = []
        if models_dir.exists():
            latest_path = models_dir / "latest_iforest_metadata.json"
            if latest_path.exists():
                candidates.append(latest_path)
            pattern_files = sorted(models_dir.glob("iforest_metadata_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(pattern_files)
        for path in candidates:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
                if not isinstance(metadata, dict):
                    continue
            except Exception:
                continue
            pipeline_path = metadata.get("pipeline_latest") or metadata.get("pipeline_path")
            if pipeline_path and not os.path.isabs(pipeline_path):
                pipeline_path = str((models_dir / pipeline_path).resolve())
            display_timestamp = metadata.get("timestamp") or path.stem
            anomaly_ratio = metadata.get("estimated_anomaly_ratio") or metadata.get("training_anomaly_ratio")
            if anomaly_ratio is not None:
                try:
                    display_text = f"{display_timestamp} | 异常占比 {float(anomaly_ratio):.2%}"
                except Exception:
                    display_text = str(display_timestamp)
            else:
                display_text = str(display_timestamp)
            registry[str(path)] = {
                "metadata_path": str(path),
                "metadata": metadata,
                "pipeline_path": pipeline_path,
                "display": display_text,
            }

        self._model_registry = registry
        current_key = self._selected_model_key
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for key, entry in registry.items():
            self.model_combo.addItem(entry["display"], key)
        self.model_combo.blockSignals(False)

        if current_key and current_key in registry:
            idx = self.model_combo.findData(current_key)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self._on_model_combo_changed(idx)
                return

        if self.model_combo.count():
            self.model_combo.setCurrentIndex(0)
            self._on_model_combo_changed(0)
        else:
            self._on_model_combo_changed(-1)

    def _on_model_combo_changed(self, index: int) -> None:
        if not hasattr(self, "model_combo"):
            return
        if index < 0:
            key = None
        else:
            key = self.model_combo.itemData(index)
        entry = self._model_registry.get(key) if key else None
        if not entry:
            self.model_info_label.setText("尚未加载模型")
            self._selected_model_key = None
            self._selected_metadata = None
            self._selected_metadata_path = None
            self._selected_pipeline_path = None
            return

        metadata = entry.get("metadata") or {}
        self._selected_model_key = key
        self._selected_metadata = metadata
        self._selected_metadata_path = entry.get("metadata_path")
        self._selected_pipeline_path = entry.get("pipeline_path")

        info_lines = []
        if metadata.get("timestamp"):
            info_lines.append(f"时间：{metadata['timestamp']}")
        if metadata.get("contamination") is not None:
            info_lines.append(f"训练污染率：{float(metadata['contamination']):.3%}")
        if metadata.get("estimated_precision") is not None:
            info_lines.append(f"估计精度：{float(metadata['estimated_precision']):.2%}")
        if metadata.get("training_anomaly_ratio") is not None:
            info_lines.append(f"训练异常占比：{float(metadata['training_anomaly_ratio']):.2%}")
        if metadata.get("pseudo_labels"):
            info_lines.append("含伪标签增强")
        if metadata.get("manual_annotations"):
            manual = metadata.get("manual_annotations")
            info_lines.append(f"手工标注：{manual.get('total', 0)} 条")

        self.model_info_label.setText("\n".join(info_lines) if info_lines else entry.get("display", ""))
        self.dashboard.update_metrics(self._analysis_summary, metadata if isinstance(metadata, dict) else None)


    def _latest_pipeline_bundle(self):
        if self._selected_pipeline_path and os.path.exists(self._selected_pipeline_path):
            return self._selected_pipeline_path, self._selected_metadata_path

        models_dir = self._default_models_dir()
        latest_path = os.path.join(models_dir, "latest_iforest_pipeline.joblib")
        latest_meta = os.path.join(models_dir, "latest_iforest_metadata.json")
        if os.path.exists(latest_path):
            return latest_path, latest_meta if os.path.exists(latest_meta) else None

        candidates = []
        prefix = "iforest_pipeline_"
        suffix = ".joblib"
        for name in os.listdir(models_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            path = os.path.join(models_dir, name)
            stamp = name[len(prefix):-len(suffix)]
            meta = os.path.join(models_dir, f"iforest_metadata_{stamp}.json")
            candidates.append((os.path.getmtime(path), path, meta if os.path.exists(meta) else None))

        if not candidates:
            return None, None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, path, meta = candidates[0]
        return path, meta

    def _prediction_allowed_extras(self, metadata: Optional[dict]) -> Set[str]:
        allowed: Set[str] = set(TRAIN_META_COLUMNS)
        if not isinstance(metadata, dict):
            return allowed

        def _extend_from(value) -> None:
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                allowed.update(str(col) for col in value if str(col))
            else:
                allowed.add(str(value))

        for key in (
            "meta_columns",
            "meta_fields",
            "reserved_columns",
            "id_columns",
            "keep_columns",
            "drop_columns",
        ):
            _extend_from(metadata.get(key))

        for key in ("ground_truth_column", "label_column", "labels_column", "id_column"):
            value = metadata.get(key)
            if value:
                allowed.add(str(value))

        preprocessor_meta = metadata.get("preprocessor")
        if isinstance(preprocessor_meta, dict):
            for key in (
                "meta_columns",
                "id_columns",
                "keep_columns",
                "drop_columns",
                "reserved_columns",
            ):
                _extend_from(preprocessor_meta.get(key))

        allowed.update({
            "timestamp",
            "time",
            "event_time",
            "frame_time",
            "frame_time_epoch",
            "pcap_name",
            "file_name",
            "file_path",
            "source_file",
            "source_path",
        })
        return {str(col) for col in allowed if str(col)}

    def _resolve_prediction_bundle(
        self,
        df: "pd.DataFrame",
        *,
        metadata_override: Optional[dict] = None,
    ) -> Optional[dict]:
        if pd is None:
            return None

        df_columns = [str(col) for col in df.columns if str(col)]
        if not df_columns:
            return None

        column_set = {col for col in df_columns}
        models_dir = Path(self._default_models_dir())

        candidates: List[dict] = []

        def _register_candidate(
            metadata: Optional[dict],
            metadata_path: Optional[str],
            pipeline_hint: Optional[str],
            priority: int,
            source: str,
        ) -> None:
            if not isinstance(metadata, dict):
                return

            feature_order = _feature_order_from_metadata(metadata)
            if not feature_order:
                return

            feature_set = {str(col) for col in feature_order}
            missing = [col for col in feature_order if col not in column_set]
            if missing:
                return

            pipeline_path = pipeline_hint or metadata.get("pipeline_latest") or metadata.get("pipeline_path")
            if not pipeline_path:
                return
            if not os.path.isabs(pipeline_path):
                pipeline_path = str((models_dir / pipeline_path).resolve())
            if not os.path.exists(pipeline_path):
                return

            allowed_extra = self._prediction_allowed_extras(metadata)
            extras = [col for col in column_set if col not in feature_set and col not in allowed_extra]
            score = (len(extras), priority, -len(feature_set))

            candidates.append(
                {
                    "metadata": metadata,
                    "metadata_path": metadata_path,
                    "pipeline_path": pipeline_path,
                    "feature_order": feature_order,
                    "allowed_extra": allowed_extra,
                    "extras": extras,
                    "score": score,
                    "source": source,
                }
            )

        seen_paths: Set[str] = set()

        if isinstance(metadata_override, dict):
            override_path = metadata_override.get("metadata_path")
            if override_path:
                seen_paths.add(str(override_path))
            pipeline_hint = metadata_override.get("pipeline_latest") or metadata_override.get("pipeline_path")
            if pipeline_hint is None:
                pipeline_hint = self._selected_pipeline_path
            _register_candidate(metadata_override, override_path, pipeline_hint, 0, "override")

        if self._selected_metadata:
            meta_path = self._selected_metadata_path
            if meta_path:
                seen_paths.add(str(meta_path))
            _register_candidate(
                self._selected_metadata,
                meta_path,
                self._selected_pipeline_path,
                1,
                "selected",
            )

        if models_dir.exists():
            latest_meta = models_dir / "latest_iforest_metadata.json"
            if latest_meta.exists():
                resolved = str(latest_meta.resolve())
                if resolved not in seen_paths:
                    try:
                        with open(latest_meta, "r", encoding="utf-8") as fh:
                            metadata = json.load(fh)
                    except Exception:
                        metadata = None
                    _register_candidate(metadata, resolved, None, 2, "latest")
                    seen_paths.add(resolved)

            pattern_files = sorted(
                models_dir.glob("iforest_metadata_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            priority = 3
            for path in pattern_files:
                resolved = str(path.resolve())
                if resolved in seen_paths:
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        metadata = json.load(fh)
                except Exception:
                    continue
                _register_candidate(metadata, resolved, None, priority, "history")
                seen_paths.add(resolved)
                priority += 1

        if not candidates:
            return None

        best = min(candidates, key=lambda item: item["score"])
        return best
    def _browse_compat(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择 pcap 文件", "", "pcap (*.pcap *.pcapng);;所有文件 (*)")
        if not p:
            d = QtWidgets.QFileDialog.getExistingDirectory(None, "选择包含多个小包的目录（如 data/split）", self._default_split_dir())
            if d:
                self.file_edit.setText(d); self.display_result(f"已选择目录: {d}", True)
                self._remember_path(d)
            return
        self.file_edit.setText(p); self.display_result(f"已选择文件: {p}", True)
        self._remember_path(p)

    def _choose_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择 pcap 文件", "", "pcap (*.pcap *.pcapng);;所有文件 (*)")
        if p:
            self.file_edit.setText(p); self.display_result(f"已选择文件: {p}", True)
            self._remember_path(p)

    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(None, "选择包含多个小包的目录（如 data/split）", self._default_split_dir())
        if d:
            self.file_edit.setText(d); self.display_result(f"已选择目录: {d}", True)
            self._remember_path(d)

    def _ask_pcap_input(self):
        current = self.file_edit.text().strip()
        if current and os.path.exists(current):
            start_dir = current if os.path.isdir(current) else os.path.dirname(current)
        else:
            start_dir = self._default_split_dir()

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "选择 PCAP 文件",
            start_dir,
            "pcap (*.pcap *.pcapng);;所有文件 (*)",
        )
        if file_path:
            return file_path

        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "选择包含多个小包的目录",
            start_dir or self._default_split_dir(),
        )
        if dir_path:
            return dir_path
        return None

    def _ask_feature_source(self):
        """兼容旧逻辑，保留文件对话框以便特殊场景下自定义输入。"""
        start_dir = self._default_csv_feature_dir()
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "选择特征 CSV 所在目录", start_dir
        )
        if dir_path:
            return dir_path

        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "选择特征 CSV 文件",
            start_dir,
            "CSV (*.csv);;所有文件 (*)",
        )
        if files:
            return files
        return None

    def _default_feature_csv_files(self) -> List[str]:
        """获取默认特征目录下的全部 CSV 文件列表。"""
        feature_dir = Path(self._default_csv_feature_dir())
        if not feature_dir.exists() or not feature_dir.is_dir():
            return []

        files: Set[str] = set()
        for pattern in ("*.csv", "*.CSV"):
            for path in feature_dir.glob(pattern):
                if path.is_file():
                    try:
                        files.add(str(path.resolve()))
                    except Exception:
                        files.add(str(path))

        return sorted(files)

    def _ask_training_source(self):
        current = self.file_edit.text().strip()
        if current and os.path.exists(current):
            start_dir = current if os.path.isdir(current) else os.path.dirname(current)
        else:
            start_dir = self._preprocess_out_dir()

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "选择训练数据（预处理CSV/NPZ/PCAP）",
            start_dir,
            "预处理CSV (*.csv);;NPZ (*.npz);;PCAP (*.pcap *.pcapng);;所有文件 (*)",
        )
        if file_path:
            return file_path

        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            None,
            "选择训练数据所在目录",
            start_dir or self._default_split_dir(),
        )
        if dir_path:
            return dir_path
        return None

    def _ask_analysis_csv(self):
        start_dir = self._default_results_dir()
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "选择检测结果 CSV",
            start_dir,
            "CSV (*.csv);;所有文件 (*)",
        )
        if csv_path:
            return csv_path
        return None

    def display_result(self, text, append=True):
        (self.results_text.append if append else self.results_text.setPlainText)(text)

    def _update_batch_controls(self):
        is_batch = (self._mode_map.get(self.mode_combo.currentText(), "auto") == "batch")
        for w in (self.batch_spin, self.start_spin, self.btn_prev, self.btn_next):
            w.setEnabled(is_batch)

    def _list_sorted(self, d: str) -> List[str]:
        suffixes = (".pcap", ".pcapng")
        matches: List[str] = []

        if os.path.isfile(d):
            return [d] if d.lower().endswith(suffixes) else []

        try:
            for root, _dirs, files in os.walk(d):
                for name in files:
                    if not name.lower().endswith(suffixes):
                        continue
                    matches.append(os.path.join(root, name))
        except Exception:
            matches = []

        matches.sort()
        return matches

    # ——按钮进度视觉条——
    def set_button_progress(self, button: QtWidgets.QPushButton, progress: int):
        p = max(0, min(100, int(progress))) / 100.0
        button.setStyleSheet(
            f'QPushButton{{border:0;border-radius:10px;padding:8px 12px;'
            f'background:qlineargradient(x1:0,y1:0,x2:1,y2:0,' 
            f'stop:0 #69A1FF, stop:{p:.3f} #69A1FF, stop:{p:.3f} #EEF1F6, stop:1 #EEF1F6);}}'
            'QPushButton:hover{background:#E2E8F0;} QPushButton:pressed{background:#D9E0EA;}'
        )

    def reset_button_progress(self, button: QtWidgets.QPushButton):
        button.setStyleSheet("")

    # --------- 分页器 ----------
    def _open_csv_paged(self, csv_path: str):
        if not os.path.exists(csv_path):
            QtWidgets.QMessageBox.warning(None, "CSV 不存在", csv_path); return
        self._csv_paged_path = csv_path
        total = 0
        with open(csv_path, "rb") as f:
            for _ in f: total += 1
        self._csv_total_rows = max(0, total - 1)
        self._csv_current_page = 1
        self._load_page(self._csv_current_page)

    def _load_page(self, page: int):
        if not self._csv_paged_path or self._csv_total_rows is None: return
        page_size = self.page_size_spin.value()
        total_pages = max(1, math.ceil(self._csv_total_rows / page_size))
        page = max(1, min(total_pages, page))
        skip = (page - 1) * page_size
        try:
            df = read_csv_flexible(
                self._csv_paged_path,
                skiprows=range(1, 1 + skip),
                nrows=page_size,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "分页读取失败", str(e)); return
        self._csv_current_page = page
        self.page_info.setText(f"第 {page}/{total_pages} 页（共 {self._csv_total_rows} 行）")
        self.populate_table_from_df(df)

    def _on_page_prev(self):
        if self._csv_current_page > 1: self._load_page(self._csv_current_page - 1)

    def _on_page_next(self):
        if self._csv_total_rows is None: return
        size = self.page_size_spin.value()
        total_pages = max(1, math.ceil(self._csv_total_rows / size))
        if self._csv_current_page < total_pages:
            self._load_page(self._csv_current_page + 1)

    def _on_page_size_changed(self, _):
        if self._csv_paged_path: self._load_page(1)

    def _show_full_preview(self):
        csv_path = None
        for p in [self._csv_paged_path, self._last_out_csv]:
            if p and os.path.exists(p): csv_path = p; break
        if not csv_path:
            QtWidgets.QMessageBox.warning(None, "没有 CSV", "请先查看流量信息或导出带标注 CSV。"); return
        app = QtWidgets.QApplication.instance()
        app.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            df = read_csv_flexible(csv_path)
            self.populate_table_from_df(df)
            self.display_result(f"[INFO] 已加载全部 {len(df)} 行。")
            self._csv_paged_path = None; self._csv_total_rows = None; self._csv_current_page = 1
            self.page_info.setText("第 1/1 页")
        finally:
            app.restoreOverrideCursor()

    # --------- 表格渲染 ----------
    def populate_table_from_df(self, df: "pd.DataFrame"):
        if pd is None:
            raise RuntimeError("pandas required")
        self._last_preview_df = df
        show_df = df.head(PREVIEW_LIMIT_FOR_TABLE).copy() if (len(df) > PREVIEW_LIMIT_FOR_TABLE and not self._csv_paged_path) else df
        self.table_view.setUpdatesEnabled(False)
        try:
            if getattr(self, "_table_proxy", None) is not None:
                try:
                    self.table_view.setModel(None)
                    self._table_proxy.deleteLater()
                except Exception:
                    pass
                self._table_proxy = None
            if getattr(self, "_table_model", None) is not None:
                try:
                    self._table_model.deleteLater()
                except Exception:
                    pass
                self._table_model = None
            m = PandasFrameModel(show_df, self.table_view)
            proxy = QtCore.QSortFilterProxyModel(self.table_view)
            proxy.setSourceModel(m)
            proxy.setFilterKeyColumn(-1)
            self.table_view.setModel(proxy)
            self.table_view.setSortingEnabled(True)
            self.table_view.resizeColumnsToContents()
            self.display_tabs.setCurrentWidget(self.table_widget)
            self._table_model = m
            self._table_proxy = proxy
        finally:
            self.table_view.setUpdatesEnabled(True)

    def _on_table_double_click(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return

        model = self.table_view.model()
        if model is None:
            return

        source_model = model
        source_index = index
        if isinstance(model, QtCore.QSortFilterProxyModel):
            source_index = model.mapToSource(index)
            source_model = model.sourceModel()

        df = getattr(source_model, "_df", None)
        if df is None or pd is None or not isinstance(df, pd.DataFrame):
            return

        row = source_index.row()
        if row < 0 or row >= len(df):
            return

        record = dict(df.iloc[row])
        dialog = AnomalyDetailDialog(record, self.table_view)
        dialog.annotation_saved.connect(lambda value, rec=record: self._on_manual_annotation_saved(rec, value))
        dialog.exec_()

    def _on_manual_annotation_saved(self, record: Dict[str, object], value: float) -> None:
        try:
            summary = annotation_summary()
        except Exception:
            summary = {}

        key_hint = record.get("flow_id") or record.get("__source_file__") or record.get("pcap_file")
        if key_hint:
            self.display_result(f"[INFO] 已保存人工标注：{key_hint} -> {value}")
        else:
            self.display_result(f"[INFO] 已保存人工标注值：{value}")

        if summary:
            self.display_result(
                "[INFO] 人工标注统计：总 {} 条（异常 {}，正常 {}）".format(
                    int(summary.get("total", 0)),
                    int(summary.get("anomalies", 0)),
                    int(summary.get("normals", 0)),
                )
            )

        if pd is None:
            return

        try:
            if self._last_out_csv and os.path.exists(self._last_out_csv):
                df_full = read_csv_flexible(self._last_out_csv)
                labels = apply_annotations_to_frame(df_full)
                if labels is not None:
                    df_full = df_full.copy()
                    df_full["manual_label"] = labels
                    df_full.to_csv(self._last_out_csv, index=False, encoding="utf-8")

            if isinstance(self._latest_prediction_summary, dict):
                df_latest = self._latest_prediction_summary.get("dataframe")
                if isinstance(df_latest, pd.DataFrame):
                    labels = apply_annotations_to_frame(df_latest)
                    if labels is not None:
                        df_latest = df_latest.copy()
                        df_latest["manual_label"] = labels
                        self._latest_prediction_summary["dataframe"] = df_latest

            if self._last_preview_df is not None and isinstance(self._last_preview_df, pd.DataFrame):
                df_preview = self._last_preview_df.copy()
                labels = apply_annotations_to_frame(df_preview)
                if labels is not None:
                    df_preview["manual_label"] = labels
                self.populate_table_from_df(df_preview)
        except Exception as exc:
            self.display_result(f"[WARN] 刷新标注视图失败：{exc}")

        self.dashboard.update_metrics(
            self._analysis_summary,
            self._selected_metadata if isinstance(self._selected_metadata, dict) else None,
        )

    def _auto_tag_dataframe(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if pd is None or df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df
        try:
            labels = apply_annotations_to_frame(df)
        except Exception as exc:
            self.display_result(f"[WARN] 应用人工标注失败：{exc}")
            return df
        if labels is None:
            return df
        tagged = df.copy()
        tagged["manual_label"] = labels
        return tagged

    # --------- 批次按钮 ----------

    def _on_prev_batch(self):
        if not self.btn_prev.isEnabled(): return
        step = max(1, self.batch_spin.value())
        self.start_spin.setValue(max(0, self.start_spin.value() - step))
        self._on_view_info()

    def _on_next_batch(self):
        if not self.btn_next.isEnabled(): return
        step = max(1, self.batch_spin.value())
        self.start_spin.setValue(self.start_spin.value() + step)
        self._on_view_info()

    # --------- 查看流量 ----------
    def _cancel_running(self):
        if getattr(self, "worker", None) and self.worker.isRunning():
            self.worker.requestInterruption()

    def _on_view_info(self):
        self._cancel_running()
        path = self.file_edit.text().strip()
        if not path:
            self.display_result("请先在顶部选择文件或目录"); return

        mode = self._mode_map.get(self.mode_combo.currentText(), "auto")
        batch = self.batch_spin.value()
        start = self.start_spin.value()
        workers = self.workers_spin.value()
        proto_map = {"TCP+UDP": "both", "仅TCP": "tcp", "仅UDP": "udp"}
        proto = proto_map.get(self.proto_combo.currentText(), "both")
        wl = self.whitelist_edit.text().strip()
        bl = self.blacklist_edit.text().strip()

        file_list = []
        if os.path.isdir(path):
            allf = self._list_sorted(path)
            if mode == "batch":
                s = max(0, start); e = min(len(allf), s + max(1, batch)); file_list = allf[s:e]
            elif mode in ("auto", "all"):
                file_list = allf
            else:
                pick, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择单个 pcap 文件", path, "pcap (*.pcap *.pcapng);;所有文件 (*)")
                if not pick:
                    self.display_result("[INFO] 已取消选择单文件"); return
                path = pick; file_list = [path]
        else:
            file_list = [path]

        self.display_result(f"[INFO] 即将解析的文件（按顺序）共 {len(file_list)} 个：")
        for p in file_list[:100]: self.display_result(" - " + p)
        if len(file_list) > 100: self.display_result(f" ...（其余 {len(file_list) - 100} 个省略显示）")
        self.display_result(f"[INFO] 正在解析 {path} ...", True)

        self._set_action_buttons_enabled(False)
        self.btn_view.setEnabled(False); self.set_button_progress(self.btn_view, 0)

        self.worker = FunctionThread(
            info,
            path=path,
            workers=workers,
            mode=(
                "all"
                if mode == "auto" and os.path.isdir(path)
                else ("file" if mode == "auto" and os.path.isfile(path) else mode)
            ),
            batch_size=batch,
            start_index=start,
            files=file_list if os.path.isdir(path) else None,
            proto_filter=proto,
            port_whitelist_text=wl,
            port_blacklist_text=bl,
            fast=True,
            cancel_arg="cancel_cb",
        )
        self.worker.progress.connect(lambda p: self.set_button_progress(self.btn_view, p))
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_finished(self, df):
        self.btn_view.setEnabled(True)
        self.set_button_progress(self.btn_view, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_view))
        self.worker = None
        self._set_action_buttons_enabled(True)

        if pd is not None and isinstance(df, pd.DataFrame):
            df = self._auto_tag_dataframe(df)

        out_csv = getattr(df, "attrs", {}).get("out_csv", None)
        files_total = getattr(df, "attrs", {}).get("files_total", None)
        errs = getattr(df, "attrs", {}).get("errors", "")

        dst_dir = self._default_csv_info_dir()
        os.makedirs(dst_dir, exist_ok=True)
        try:
            if out_csv and os.path.exists(out_csv):
                dst = os.path.join(dst_dir, f"pcap_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                shutil.copy2(out_csv, dst)
                self._open_csv_paged(dst)
                self._last_out_csv = dst
                self._add_output(dst)
            else:
                tmp_csv = os.path.join(dst_dir, f"pcap_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                df.to_csv(tmp_csv, index=False, encoding="utf-8")
                self._open_csv_paged(tmp_csv)
                self._last_out_csv = tmp_csv
                self._add_output(tmp_csv)
        except Exception as e:
            self.display_result(f"[WARN] 落盘失败：{e}")

        if df is None or (hasattr(df, "empty") and df.empty):
            self.display_result("解析完成，但未找到流量数据。")
        else:
            try:
                head_txt = df.head(20).to_string()
            except Exception:
                head_txt = "(预览生成失败)"
            self.display_result(f"[INFO] 解析完成（表格仅显示前 {PREVIEW_LIMIT_FOR_TABLE} 行；全部已写入 CSV）。\n{head_txt}", append=False)
            rows = len(df) if hasattr(df, "__len__") else 0
            status = f"预览 {min(rows, 20)} 行；共处理文件 {files_total if files_total is not None else '?'} 个"
            self._update_status_message(status)
        if errs:
            for e in errs.split("\n"):
                if e.strip(): self.display_result(e)

    def _on_worker_error(self, msg):
        self.btn_view.setEnabled(True); self.reset_button_progress(self.btn_view)
        self._set_action_buttons_enabled(True)
        self.worker = None
        QtWidgets.QMessageBox.critical(None, "解析失败", msg)
        self.display_result(f"[错误] 解析失败: {msg}")

    # --------- 导出异常 ----------
    def _on_export_results(self):
        out_dir = self._abnormal_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        summary_payload = self._analysis_summary if isinstance(self._analysis_summary, dict) else {}
        export_payload = summary_payload.get("export_payload") if isinstance(summary_payload.get("export_payload"), dict) else None
        metrics_payload = summary_payload.get("metrics") if isinstance(summary_payload.get("metrics"), dict) else None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        written_any = False

        if export_payload:
            anomaly_count = int(export_payload.get("anomaly_count", 0))
            anomaly_files = [str(x) for x in export_payload.get("anomaly_files", []) if x]
            single_status = export_payload.get("single_file_status")
            status_text = export_payload.get("status_text") or ("异常" if anomaly_count > 0 else "正常")

            summary_txt = [f"分析结论: {status_text}", f"异常包数量: {anomaly_count}"]
            if anomaly_files:
                summary_txt.append("异常包名: " + ", ".join(anomaly_files))
            else:
                summary_txt.append("异常包名: 无")
            if single_status:
                summary_txt.append(f"单文件结论: {single_status}")

            summary_txt_display = "\n".join(summary_txt)
            self.display_result(f"[INFO] 导出摘要：\n{summary_txt_display}")

            summary_txt_path = os.path.join(out_dir, f"abnormal_summary_{timestamp}.txt")
            with open(summary_txt_path, "w", encoding="utf-8") as fh:
                fh.write(summary_txt_display)
            self._add_output(summary_txt_path)
            written_any = True

            if metrics_payload and metrics_payload.get("anomalous_files") and pd is not None:
                try:
                    df_files = pd.DataFrame(metrics_payload["anomalous_files"])
                    if not df_files.empty:
                        csv_path = os.path.join(out_dir, f"abnormal_files_{timestamp}.csv")
                        df_files.to_csv(csv_path, index=False, encoding="utf-8")
                        self._add_output(csv_path)
                        written_any = True
                except Exception as exc:
                    self.display_result(f"[WARN] 导出异常文件列表失败：{exc}")

            if anomaly_count == 0:
                QtWidgets.QMessageBox.information(None, "没有异常", "当前分析结果中未检测到异常流量。")
                return

        df = self._last_preview_df
        if df is None or (hasattr(df, "empty") and df.empty):
            if written_any:
                return
            QtWidgets.QMessageBox.warning(None, "没有数据", "请先查看或加载带预测的数据。")
            return

        export_df = None

        def _extract_threshold(source) -> Optional[float]:
            if not isinstance(source, dict):
                return None
            for key in ("score_threshold", "decision_threshold", "threshold"):
                val = source.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    try:
                        return float(val)
                    except ValueError:
                        continue
            return None

        if "prediction_status" in df.columns:
            export_df = df[df["prediction_status"].astype(str) == "异常"].copy()
        elif "prediction_label" in df.columns:
            export_df = df[
                df["prediction_label"].astype(str).str.strip().str.lower().isin(
                    {"异常", "恶意", "malicious", "anomaly", "attack"}
                )
            ].copy()
        elif "malicious_score" in df.columns:
            threshold = _extract_threshold(metrics_payload) if metrics_payload else None
            if threshold is None:
                threshold = _extract_threshold(summary_payload.get("metadata")) if isinstance(summary_payload.get("metadata"), dict) else None
            if threshold is None:
                threshold = _extract_threshold(self._selected_metadata) if isinstance(self._selected_metadata, dict) else None
            if threshold is None and isinstance(self._latest_prediction_summary, dict):
                threshold = _extract_threshold(self._latest_prediction_summary.get("metadata"))
            if threshold is None:
                threshold = 0.5
            scores = pd.to_numeric(df["malicious_score"], errors="coerce") if pd is not None else df["malicious_score"]
            export_df = df[scores >= threshold].copy()
        elif "prediction" in df.columns:
            export_df = df[df["prediction"].isin([1, -1])].copy()
        elif "anomaly_score" in df.columns:
            export_df = df[df["anomaly_score"] > 0].copy()
        else:
            if not written_any:
                QtWidgets.QMessageBox.information(None, "无异常标记", "没有可识别的异常列，无法筛选异常。")
            return
        if export_df.empty:
            if not written_any:
                QtWidgets.QMessageBox.information(None, "没有异常", "当前数据中未检测到异常行。")
            return

        outp = os.path.join(out_dir, f"abnormal_{timestamp}.csv")
        export_df.to_csv(outp, index=False, encoding="utf-8")
        self._add_output(outp)
        self.display_result(f"[INFO] 已导出异常CSV：{outp}")
        self._open_csv_paged(outp)

    def _on_export_report(self):
        analysis = self._analysis_summary if isinstance(self._analysis_summary, dict) else {}
        if not analysis:
            QtWidgets.QMessageBox.warning(None, "暂无分析数据", "请先运行一次分析以生成报告内容。")
            return

        out_dir = self._analysis_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        default_name = os.path.join(out_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "导出分析报告",
            default_name,
            "PDF 文件 (*.pdf)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"

        metadata = (
            self._selected_metadata if isinstance(self._selected_metadata, dict) else analysis.get("metadata")
        )
        prediction = (
            self._latest_prediction_summary if isinstance(self._latest_prediction_summary, dict) else None
        )
        try:
            annot_info = annotation_summary()
        except Exception:
            annot_info = {}

        metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else {}
        summary_text = str(analysis.get("summary_text") or "").strip()
        summary_lines = [line for line in summary_text.splitlines() if line.strip()]
        export_payload = (
            analysis.get("export_payload") if isinstance(analysis.get("export_payload"), dict) else {}
        )

        def _fmt_percent(value: object, digits: int = 2) -> Optional[str]:
            try:
                return format(float(value), f".{digits}%")
            except Exception:
                return None

        def _fmt_float(value: object, digits: int = 3) -> Optional[str]:
            try:
                return format(float(value), f".{digits}f")
            except Exception:
                return None

        try:
            with PdfPages(file_path) as pdf:

                def _new_page() -> Tuple[plt.Figure, plt.Axes]:
                    fig_obj, ax_obj = plt.subplots(figsize=(8.27, 11.69))
                    fig_obj.patch.set_facecolor("#FFFFFF")
                    ax_obj.axis("off")
                    return fig_obj, ax_obj

                fig, ax = _new_page()
                y_pos = 0.94

                ax.text(
                    0.5,
                    y_pos,
                    "恶意流量检测分析报告",
                    ha="center",
                    va="top",
                    fontsize=22,
                    weight="bold",
                    color="#0F172A",
                )
                y_pos -= 0.08
                ax.text(
                    0.5,
                    y_pos,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ha="center",
                    va="top",
                    fontsize=12,
                    color="#4B5563",
                )
                y_pos -= 0.05

                def _ensure_space(lines: float = 1.0) -> None:
                    nonlocal fig, ax, y_pos
                    required = lines * 0.032 + 0.04
                    if y_pos - required < 0.08:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                        fig, ax = _new_page()
                        y_pos = 0.94

                def _add_section(title: str, rows: List[Union[str, Tuple[str, str]]]) -> None:
                    nonlocal fig, ax, y_pos
                    content = [row for row in rows if row]
                    if not content:
                        return
                    estimated_lines = 1.0
                    for row in content:
                        if isinstance(row, tuple):
                            text_value = str(row[1])
                        else:
                            text_value = str(row)
                        wrapped = textwrap.wrap(text_value, width=70) or [text_value]
                        estimated_lines += max(1, len(wrapped))
                    _ensure_space(estimated_lines)
                    ax.text(
                        0.05,
                        y_pos,
                        title,
                        fontsize=14,
                        weight="bold",
                        color="#0F172A",
                        va="top",
                    )
                    y_pos -= 0.045
                    for row in content:
                        if isinstance(row, tuple):
                            label, value = row
                            wrapped = textwrap.wrap(str(value), width=64) or [""]
                            ax.text(
                                0.06,
                                y_pos,
                                f"{label}：",
                                fontsize=12,
                                weight="bold",
                                color="#111827",
                                va="top",
                            )
                            ax.text(
                                0.22,
                                y_pos,
                                wrapped[0],
                                fontsize=12,
                                color="#111827",
                                va="top",
                            )
                            y_pos -= 0.032
                            for extra_line in wrapped[1:]:
                                _ensure_space(1)
                                ax.text(
                                    0.22,
                                    y_pos,
                                    extra_line,
                                    fontsize=12,
                                    color="#111827",
                                    va="top",
                                )
                                y_pos -= 0.032
                        else:
                            wrapped = textwrap.wrap(str(row), width=72) or [""]
                            for line in wrapped:
                                ax.text(
                                    0.06,
                                    y_pos,
                                    line,
                                    fontsize=12,
                                    color="#111827",
                                    va="top",
                                )
                                y_pos -= 0.032
                    y_pos -= 0.02

                base_rows: List[Tuple[str, str]] = []
                if metadata:
                    if metadata.get("timestamp"):
                        base_rows.append(("模型训练时间", str(metadata.get("timestamp"))))
                    if metadata.get("contamination") is not None:
                        percent = _fmt_percent(metadata.get("contamination"))
                        if percent:
                            base_rows.append(("训练污染率", percent))
                    if metadata.get("training_anomaly_ratio") is not None:
                        percent = _fmt_percent(metadata.get("training_anomaly_ratio"))
                        if percent:
                            base_rows.append(("训练异常占比", percent))
                    if metadata.get("estimated_precision") is not None:
                        percent = _fmt_percent(metadata.get("estimated_precision"))
                        if percent:
                            base_rows.append(("模型估计精度", percent))
                if analysis.get("out_dir"):
                    base_rows.append(("分析输出目录", str(analysis.get("out_dir"))))
                if base_rows:
                    _add_section("基础信息", base_rows)

                if summary_lines:
                    _add_section("分析结论摘要", summary_lines)

                metric_rows: List[Union[str, Tuple[str, str]]] = []
                if metrics:
                    if metrics.get("total_rows") is not None:
                        metric_rows.append(("检测样本总数", str(metrics.get("total_rows"))))
                    if metrics.get("malicious_total") is not None:
                        detail = str(metrics.get("malicious_total"))
                        ratio = _fmt_percent(metrics.get("malicious_ratio"))
                        if ratio:
                            detail = f"{detail}（异常占比 {ratio}）"
                        metric_rows.append(("异常样本数量", detail))
                    if metrics.get("anomaly_count") is not None and metrics.get("anomaly_count") != metrics.get("malicious_total"):
                        metric_rows.append(("疑似异常数量", str(metrics.get("anomaly_count"))))
                    if metrics.get("score_threshold") is not None:
                        formatted = _fmt_float(metrics.get("score_threshold"), 6)
                        if formatted:
                            metric_rows.append(("建议异常得分阈值", formatted))
                    if metrics.get("ratio_threshold") is not None:
                        percent = _fmt_percent(metrics.get("ratio_threshold"))
                        if percent:
                            metric_rows.append(("文件级异常阈值", f"恶意占比 ≥ {percent}"))
                    drift_alerts = metrics.get("drift_alerts")
                    if isinstance(drift_alerts, dict) and drift_alerts:
                        metric_rows.append(
                            (
                                "分布漂移告警",
                                ", ".join(f"{k}: {_fmt_percent(v, 2) or v}" for k, v in drift_alerts.items()),
                            )
                        )
                    model_metrics = metrics.get("model_metrics")
                    if isinstance(model_metrics, dict) and model_metrics:
                        parts = []
                        for key, label in (
                            ("precision", "Precision"),
                            ("recall", "Recall"),
                            ("f1", "F1"),
                            ("roc_auc", "ROC-AUC"),
                            ("pr_auc", "PR-AUC"),
                        ):
                            value = model_metrics.get(key)
                            if value is None:
                                continue
                            if key in {"precision", "recall", "f1"}:
                                formatted = _fmt_percent(value)
                            else:
                                formatted = _fmt_float(value, 3)
                            if formatted:
                                parts.append(f"{label} {formatted}")
                        if parts:
                            metric_rows.append(("模型评估", "，".join(parts)))
                if metric_rows:
                    _add_section("关键指标", metric_rows)

                if annot_info and annot_info.get("total"):
                    total = annot_info.get("total", 0)
                    anomalies = annot_info.get("anomalies", 0)
                    normals = annot_info.get("normals", 0)
                    _add_section(
                        "人工标注统计",
                        [
                            (
                                "累计标注",
                                f"共 {total} 条（异常 {anomalies}，正常 {normals}）",
                            )
                        ],
                    )

                if prediction and prediction.get("summary"):
                    lines = [str(item) for item in prediction.get("summary") if item]
                    if lines:
                        _add_section("最近一次模型预测", lines)

                export_rows: List[Tuple[str, str]] = []
                if metrics.get("summary_csv"):
                    export_rows.append(("总体统计 CSV", str(metrics.get("summary_csv"))))
                if metrics.get("top_csv"):
                    export_rows.append(("Top 异常 CSV", str(metrics.get("top_csv"))))
                if analysis.get("summary_json"):
                    export_rows.append(("分析摘要 JSON", str(analysis.get("summary_json"))))
                if export_payload and export_payload.get("status_text"):
                    export_rows.append(("整体判断", str(export_payload.get("status_text"))))
                if export_rows:
                    _add_section("输出文件", export_rows)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                preview_df = None
                top_csv = metrics.get("top_csv") or analysis.get("top20_csv")
                if top_csv and os.path.exists(str(top_csv)) and pd is not None:
                    try:
                        preview_df = read_csv_flexible(top_csv, nrows=8)
                    except Exception:
                        preview_df = None
                if preview_df is not None and not preview_df.empty:
                    fig_table, ax_table = _new_page()
                    ax_table.set_title("Top 异常样本预览", fontsize=16, pad=20)
                    display_df = preview_df.head(8)
                    max_cols = min(6, len(display_df.columns))
                    columns = [str(col) for col in display_df.columns[:max_cols]]
                    data = [
                        [str(value)[:80] for value in row]
                        for row in display_df.iloc[:, :max_cols].itertuples(index=False)
                    ]
                    table = ax_table.table(
                        cellText=data,
                        colLabels=columns,
                        loc="center",
                        cellLoc="left",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1.0, 1.3)
                    ax_table.axis("off")
                    ax_table.text(
                        0.5,
                        0.05,
                        f"数据来源：{os.path.basename(str(top_csv))}",
                        ha="center",
                        fontsize=10,
                        color="#4B5563",
                    )
                    pdf.savefig(fig_table, bbox_inches="tight")
                    plt.close(fig_table)

                plot_candidates: List[str] = []
                timeline_path = analysis.get("timeline_plot")
                if not timeline_path and metrics:
                    timeline_path = metrics.get("timeline_plot")
                if timeline_path:
                    plot_candidates.append(str(timeline_path))
                plots = analysis.get("plots")
                if isinstance(plots, (list, tuple)):
                    plot_candidates.extend(str(p) for p in plots if p)

                seen: Set[str] = set()
                for plot_path in plot_candidates:
                    abs_path = os.path.abspath(plot_path)
                    if abs_path in seen or not os.path.exists(abs_path):
                        continue
                    seen.add(abs_path)
                    fig_plot, ax_plot = _new_page()
                    try:
                        image = plt.imread(abs_path)
                        ax_plot.imshow(image)
                        ax_plot.axis("off")
                        ax_plot.set_title(os.path.basename(abs_path), fontsize=12)
                    except Exception as exc:
                        ax_plot.axis("off")
                        ax_plot.text(
                            0.5,
                            0.5,
                            f"无法加载图像：{os.path.basename(abs_path)}\n{exc}",
                            ha="center",
                            va="center",
                        )
                    pdf.savefig(fig_plot, bbox_inches="tight")
                    plt.close(fig_plot)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(None, "导出失败", f"生成报告时出错：{exc}")
            return

        self.display_result(f"[INFO] 报告已导出：{file_path}")
        self._add_output(file_path)

    # --------- 提取特征（按顶部路径） ----------
    def _on_extract_features(self):
        selection = self._ask_pcap_input()
        if not selection:
            self.display_result("[INFO] 已取消特征提取。")
            return

        path = selection
        if isinstance(path, str) and os.path.exists(path):
            self.file_edit.setText(path)

        out_dir = self._default_csv_feature_dir()
        os.makedirs(out_dir, exist_ok=True)

        self._set_action_buttons_enabled(False)
        if os.path.isdir(path):
            self.display_result(f"[INFO] 目录特征提取：{path} -> {out_dir}")
            self.btn_fe.setEnabled(False); self.set_button_progress(self.btn_fe, 1)
            self.dir_fe_worker = FunctionThread(
                fe_dir,
                path,
                out_dir,
                workers=self.workers_spin.value(),
            )
            self.dir_fe_worker.progress.connect(lambda p: self.set_button_progress(self.btn_fe, p))
            self.dir_fe_worker.finished.connect(self._on_fe_dir_finished)
            self.dir_fe_worker.error.connect(self._on_fe_error)
            self.dir_fe_worker.start()
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            csv = os.path.join(out_dir, f"{base}_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.display_result(f"[INFO] 单文件特征提取：{path} -> {csv}")
            self.btn_fe.setEnabled(False); self.set_button_progress(self.btn_fe, 1)
            self.fe_worker = FunctionThread(fe_single, path, csv)
            self.fe_worker.progress.connect(lambda p: self.set_button_progress(self.btn_fe, p))
            self.fe_worker.finished.connect(self._on_fe_finished)
            self.fe_worker.error.connect(self._on_fe_error)
            self.fe_worker.start()

    def _on_fe_finished(self, csv):
        self.btn_fe.setEnabled(True)
        self.set_button_progress(self.btn_fe, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_fe))
        self._set_action_buttons_enabled(True)
        self.display_result(f"[INFO] 特征提取完成，CSV已保存: {csv}")
        self._add_output(csv)
        try:
            df = read_csv_flexible(csv, nrows=50)
            self.populate_table_from_df(df)
            self._last_out_csv = csv
            self._open_csv_paged(csv)
        except Exception:
            pass

    def _on_fe_dir_finished(self, csv_list: List[str]):
        self.btn_fe.setEnabled(True)
        self.set_button_progress(self.btn_fe, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_fe))
        self._set_action_buttons_enabled(True)
        self.display_result(f"[INFO] 目录特征提取完成：共 {len(csv_list)} 个 CSV")
        for p in csv_list:
            if os.path.exists(p): self._add_output(p)
        if csv_list:
            first = csv_list[0]
            try:
                df = read_csv_flexible(first, nrows=50)
                self.populate_table_from_df(df)
                self._last_out_csv = first
                self._open_csv_paged(first)
            except Exception:
                pass

    def _on_fe_error(self, msg):
        self.btn_fe.setEnabled(True); self.reset_button_progress(self.btn_fe)
        self._set_action_buttons_enabled(True)
        QtWidgets.QMessageBox.critical(None, "特征提取失败", msg)
        self.display_result(f"[错误] 特征提取失败: {msg}")

    # --------- 数据预处理（基于特征 CSV） ----------
    def _on_preprocess_features(self):
        # 默认直接使用配置中的特征目录与输出目录，确保与用户目录保持一致。
        default_feature_files = self._default_feature_csv_files()
        feature_source: Optional[FeatureSource]

        # 先弹出文件选择器，允许用户手动指定要预处理的特征 CSV。
        dialog_dir = self._default_csv_feature_dir()
        if not dialog_dir or not os.path.isdir(dialog_dir):
            dialog_dir = self.file_edit.text() or os.getcwd()

        selected_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.centralwidget,
            "选择预处理特征 CSV",
            dialog_dir,
            "CSV 文件 (*.csv);;所有文件 (*)",
        )

        if selected_files:
            feature_source = list(selected_files)
            preview = f"手动选择的 {len(selected_files)} 个 CSV"
            first_path = selected_files[0]
            self.file_edit.setText(first_path)
            self._remember_path(first_path)
        elif default_feature_files:
            feature_source = list(default_feature_files)
            preview = f"{self._default_csv_feature_dir()} 中的 {len(default_feature_files)} 个 CSV"
            self.file_edit.setText(self._default_csv_feature_dir())
            self._remember_path(self._default_csv_feature_dir())
        else:
            # 回退到手动选择，兼容自定义路径场景。
            feature_source = self._ask_feature_source()
            if not feature_source:
                self.display_result("[INFO] 已取消数据预处理。")
                return
            if isinstance(feature_source, (list, tuple, set)):
                files_list = list(feature_source)
                preview = f"{len(files_list)} 个CSV文件"
                if files_list:
                    first_path = str(files_list[0])
                    self.file_edit.setText(first_path)
                    self._remember_path(first_path)
            else:
                preview = str(feature_source)
                if os.path.exists(preview):
                    self.file_edit.setText(preview)
                    self._remember_path(preview)

        out_dir = self._preprocess_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        if isinstance(feature_source, (list, tuple, set)) and not feature_source:
            QtWidgets.QMessageBox.warning(
                None,
                "未找到特征 CSV",
                "默认特征目录中未检测到任何 CSV 文件，请先执行特征提取。",
            )
            self.display_result("[WARN] 默认特征目录中没有可预处理的 CSV 文件。")
            return

        self.display_result(f"[INFO] 数据预处理：{preview} -> {out_dir}")
        self._set_action_buttons_enabled(False)
        self.btn_vector.setEnabled(False); self.set_button_progress(self.btn_vector, 1)
        self.preprocess_worker = FunctionThread(
            preprocess_dir,
            feature_source,
            out_dir,
        )
        self.preprocess_worker.progress.connect(lambda p: self.set_button_progress(self.btn_vector, p))
        self.preprocess_worker.finished.connect(self._on_preprocess_finished)
        self.preprocess_worker.error.connect(self._on_preprocess_error)
        self.preprocess_worker.start()

    def _on_preprocess_finished(self, result):
        self.btn_vector.setEnabled(True)
        self.set_button_progress(self.btn_vector, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_vector))
        self.preprocess_worker = None
        self._set_action_buttons_enabled(True)

        data = result if isinstance(result, dict) else {}
        dataset = data.get("dataset_path")
        manifest = data.get("manifest_path")
        meta_path = data.get("meta_path")
        total_rows = data.get("total_rows")
        total_cols = data.get("total_cols")

        if dataset:
            self._add_output(dataset)
        if manifest:
            self._add_output(manifest)
            try:
                df = read_csv_flexible(manifest, nrows=50)
                self.populate_table_from_df(df)
                self._last_out_csv = manifest
                self._open_csv_paged(manifest)
            except Exception:
                pass
        if meta_path:
            self._add_output(meta_path)

        summary = f"[INFO] 数据预处理完成：{total_rows or 0} 条记录，{total_cols or 0} 个特征。"
        if dataset:
            summary += f" 数据集：{dataset}"
        self.display_result(summary)

    def _on_preprocess_error(self, msg):
        self.btn_vector.setEnabled(True)
        self.reset_button_progress(self.btn_vector)
        self._set_action_buttons_enabled(True)
        QtWidgets.QMessageBox.critical(None, "数据预处理失败", msg)
        self.display_result(f"[错误] 数据预处理失败: {msg}")
        self.preprocess_worker = None

    # --------- 训练模型（按顶部路径） ----------
    def _on_train_model(self):
        selection = self._ask_training_source()
        if not selection:
            self.display_result("[INFO] 已取消模型训练。")
            return

        path = selection
        if os.path.exists(path):
            self.file_edit.setText(path)
            self._remember_path(path)

        res_dir = self._default_results_dir()
        mdl_dir = self._default_models_dir()
        os.makedirs(res_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)

        self.display_result(f"[INFO] 开始训练，输入: {path}")
        self._set_action_buttons_enabled(False)
        self.btn_train.setEnabled(False); self.set_button_progress(self.btn_train, 1)
        train_task = BackgroundTask(
            run_train,
            path,
            res_dir,
            mdl_dir,
        )
        self._start_background_task(
            train_task,
            self._on_train_finished,
            self._on_train_error,
            lambda p: self.set_button_progress(self.btn_train, p),
        )

    def _on_train_finished(self, res):
        self.btn_train.setEnabled(True)
        self.set_button_progress(self.btn_train, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_train))
        self._set_action_buttons_enabled(True)
        if not isinstance(res, dict):
            res = {} if res is None else {"result": res}
        threshold = res.get("threshold")
        vote_thr = res.get("vote_threshold")
        thr_parts = []
        if threshold is not None:
            thr_parts.append(f"score_thr={threshold:.4f}")
        if vote_thr is not None:
            thr_parts.append(f"vote_thr={vote_thr:.3f}")
        if thr_parts:
            self.display_result(f"[阈值] {', '.join(thr_parts)}")
        breakdown = res.get("threshold_breakdown") or {}
        if isinstance(breakdown, dict) and breakdown:
            try:
                tb_text = json.dumps(breakdown, ensure_ascii=False, indent=2)
            except Exception:
                tb_text = str(breakdown)
            self.display_result(f"[阈值溯源] {tb_text}")

        data_quality = res.get("data_quality") or {}
        empty_cols = list(data_quality.get("empty_columns") or [])
        const_cols = list(data_quality.get("constant_columns") or [])
        wins_cols = data_quality.get("winsorized_columns") or {}
        guard_info = res.get("memory_guard") or data_quality.get("memory_guard")
        if isinstance(guard_info, dict):
            dropped = int(guard_info.get("dropped_features", 0) or 0)
            kept = int(guard_info.get("kept_features", 0) or 0)
            budget = guard_info.get("budget_features")
            if guard_info.get("triggered") and dropped:
                msg = f"内存守护生效：{dropped} 列被裁剪，保留 {kept} 列"
                if budget:
                    msg += f"（预算上限 {int(budget)} 列）"
                self.display_result(f"[内存守护] {msg}")
            elif not guard_info.get("triggered") and budget:
                self.display_result(
                    f"[内存守护] 预算允许 {int(budget)} 列，本次未触发裁剪。"
                )
        if empty_cols:
            preview = ", ".join(empty_cols[:5])
            if len(empty_cols) > 5:
                preview += f" ...(+{len(empty_cols) - 5})"
            self.display_result(f"[数据质量] 空列已填充: {preview}")
        if const_cols:
            preview = ", ".join(const_cols[:5])
            if len(const_cols) > 5:
                preview += f" ...(+{len(const_cols) - 5})"
            self.display_result(f"[数据质量] 常量列检测: {preview}")
        if isinstance(wins_cols, dict) and wins_cols:
            win_keys = list(wins_cols.keys())
            preview = ", ".join(win_keys[:5])
            if len(win_keys) > 5:
                preview += f" ...(+{len(win_keys) - 5})"
            self.display_result(f"[数据质量] 已稳健裁剪: {preview}")

        msg_lines = [
            "results:",
            f"- {res.get('results_csv')}",
            f"- {res.get('summary_csv')}",
            "models:",
            f"- 最新管线: {res.get('pipeline_latest') or res.get('model_path')}",
            f"- 元数据: {res.get('metadata_latest') or res.get('metadata_path')}",
            f"- 标准化器: {res.get('scaler_path')}",
            f"样本总数={res.get('flows')} 异常数={res.get('malicious')}",
        ]
        if threshold is not None:
            msg_lines.append(f"得分阈值={threshold:.6f}")
        if vote_thr is not None:
            msg_lines.append(f"投票阈值={vote_thr:.3f}")
        shapes = res.get("representation_shapes") or {}
        if isinstance(shapes, dict) and shapes:
            parts = []
            base_shape = shapes.get("base")
            expanded_shape = shapes.get("expanded")
            reduced_shape = shapes.get("reduced")
            if isinstance(base_shape, (list, tuple)) and len(base_shape) == 2:
                parts.append(f"基础:{int(base_shape[0])}×{int(base_shape[1])}")
            if isinstance(expanded_shape, (list, tuple)) and len(expanded_shape) == 2:
                parts.append(f"展开:{int(expanded_shape[0])}×{int(expanded_shape[1])}")
            if isinstance(reduced_shape, (list, tuple)) and len(reduced_shape) == 2:
                parts.append(f"降维:{int(reduced_shape[0])}×{int(reduced_shape[1])}")
            if parts:
                msg_lines.append("特征形状: " + " -> ".join(parts))
        svd_info = res.get("svd_info") or {}
        if isinstance(svd_info, dict) and svd_info.get("components"):
            top_ratio = svd_info.get("top_variance_ratio") or []
            if top_ratio:
                cumulative = float(svd_info.get("cumulative_top", sum(top_ratio)))
                msg_lines.append(
                    f"SVD前{len(top_ratio)}维累计解释≈{cumulative:.1%}"
                )
            else:
                msg_lines.append(f"SVD保留维度={int(svd_info.get('components'))}")
        if empty_cols:
            msg_lines.append(f"空列填充={len(empty_cols)}")
        if const_cols:
            msg_lines.append(f"常量列={len(const_cols)}")
        if isinstance(wins_cols, dict) and wins_cols:
            msg_lines.append(f"稳健裁剪列={len(wins_cols)}")
        if res.get("estimated_precision") is not None:
            msg_lines.append(f"异常置信度均值≈{res.get('estimated_precision'):.2%}")
        if res.get("active_learning_csv"):
            msg_lines.append(f"主动学习候选: {res['active_learning_csv']}")
        pseudo_info = res.get("pseudo_labels") or {}
        if isinstance(pseudo_info, dict) and pseudo_info.get("total"):
            msg_lines.append(
                "半监督样本：人工 {human} 条 | 伪标签 异常 {pseudo_anomaly} 条 / 正常 {pseudo_normal} 条".format(
                    human=int(pseudo_info.get("human", 0)),
                    pseudo_anomaly=int(pseudo_info.get("pseudo_anomaly", 0)),
                    pseudo_normal=int(pseudo_info.get("pseudo_normal", 0)),
                )
            )
        feature_importances = res.get("feature_importances_topk") or []
        if feature_importances:
            preview_items = []
            for item in feature_importances[:5]:
                feat = item.get("feature")
                imp = item.get("importance")
                if feat is None or imp is None:
                    continue
                try:
                    preview_items.append(f"{feat}({float(imp):.3f})")
                except Exception:
                    continue
            if preview_items:
                msg_lines.append("Top特征重要性: " + ", ".join(preview_items))
        msg = "\n".join(msg_lines)
        self.display_result(f"[INFO] 训练完成：\n{msg}")
        for k in ("results_csv", "summary_csv", "model_path", "scaler_path", "pipeline_latest", "metadata_latest", "metadata_path"):
            p = res.get(k)
            if p and os.path.exists(p): self._add_output(p)
        if res.get("results_csv") and os.path.exists(res["results_csv"]):
            self._open_csv_paged(res["results_csv"])
        if res.get("active_learning_csv") and os.path.exists(res["active_learning_csv"]):
            self._add_output(res["active_learning_csv"])

    def _on_train_error(self, msg):
        self.btn_train.setEnabled(True); self.reset_button_progress(self.btn_train)
        self._set_action_buttons_enabled(True)
        QtWidgets.QMessageBox.critical(None, "训练失败", msg)
        self.display_result(f"[错误] 训练失败: {msg}")

    def _auto_analyze(self, csv_path: str) -> bool:
        btn = getattr(self, "btn_analysis", None)
        if not csv_path or not os.path.exists(csv_path) or btn is None:
            return False

        settings = getattr(self, "_settings", None)
        auto_enabled = False
        if settings is not None:
            try:
                auto_enabled = bool(settings.get("auto_analyze_after_predict"))
            except Exception:
                auto_enabled = False

        if not auto_enabled or not btn.isEnabled():
            if not getattr(self, "_auto_analyze_tip_shown", False):
                self.display_result(
                    "[INFO] 预测结果已生成，如需分析请点击右侧的“运行分析”按钮。"
                )
                self._auto_analyze_tip_shown = True
            return False

        out_dir = self._analysis_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        meta_path = getattr(self, "_selected_metadata_path", None)
        if not meta_path or not os.path.exists(meta_path):
            _, latest_meta = self._latest_pipeline_bundle()
            if latest_meta and os.path.exists(latest_meta):
                meta_path = latest_meta

        if not meta_path or not os.path.exists(meta_path):
            return False

        self.display_result(f"[INFO] 自动分析预测结果 -> {out_dir}")
        self._analysis_summary = None
        self._set_action_buttons_enabled(False)
        btn.setEnabled(False)
        self.set_button_progress(btn, 1)

        def _finished(result):
            payload = result if isinstance(result, dict) else {"out_dir": out_dir}
            if isinstance(payload, dict) and "out_dir" not in payload:
                payload["out_dir"] = out_dir
            self._on_analysis_finished(payload)

        analysis_task = BackgroundTask(
            run_analysis,
            csv_path,
            out_dir,
            metadata_path=meta_path,
        )
        self._start_background_task(
            analysis_task,
            _finished,
            self._on_analysis_error,
            lambda p: self.set_button_progress(btn, p),
        )
        return True

    # --------- 运行分析 ----------
    def _on_run_analysis(self):
        out_dir = self._analysis_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        csv = self._ask_analysis_csv()
        if not csv:
            auto = os.path.join(self._default_results_dir(), "iforest_results.csv")
            if os.path.exists(auto):
                csv = auto
                self.display_result(f"[INFO] 未选择文件，自动使用最新结果：{csv}")
            else:
                self.display_result("请先选择结果CSV"); return

        self.display_result(f"[INFO] 正在分析结果 -> {out_dir}")
        self._analysis_summary = None
        self._set_action_buttons_enabled(False)
        self.btn_analysis.setEnabled(False); self.set_button_progress(self.btn_analysis, 1)
        meta_path = self._selected_metadata_path
        if not meta_path or not os.path.exists(meta_path):
            _, meta_path = self._latest_pipeline_bundle()
        def _analysis_finished(result):
            payload = result
            if not isinstance(payload, dict):
                payload = {"out_dir": out_dir}
            else:
                payload.setdefault("out_dir", out_dir)
            self._on_analysis_finished(payload)

        analysis_task = BackgroundTask(
            run_analysis,
            csv,
            out_dir,
            metadata_path=meta_path,
        )
        self._start_background_task(
            analysis_task,
            _analysis_finished,
            self._on_analysis_error,
            lambda p: self.set_button_progress(self.btn_analysis, p),
        )

    def _on_analysis_finished(self, result):
        self.btn_analysis.setEnabled(True)
        self.set_button_progress(self.btn_analysis, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_analysis))
        self._set_action_buttons_enabled(True)
        out_dir = None
        plot_paths: List[str] = []
        top20_csv: Optional[str] = None
        summary_csv: Optional[str] = None
        summary_text: Optional[str] = None
        summary_json: Optional[str] = None

        if isinstance(result, dict):
            self._analysis_summary = result
            out_dir = result.get("out_dir") or None
            plots = result.get("plots") or []
            if isinstance(plots, (list, tuple)):
                plot_paths = [p for p in plots if isinstance(p, str)]
            elif isinstance(plots, str):
                plot_paths = [plots]
            top20_csv = result.get("top20_csv") if isinstance(result.get("top20_csv"), str) else None
            summary_csv = result.get("summary_csv") if isinstance(result.get("summary_csv"), str) else None
            summary_text = result.get("summary_text") if isinstance(result.get("summary_text"), str) else None
            summary_json = result.get("summary_json") if isinstance(result.get("summary_json"), str) else None
            metrics_csv = result.get("metrics_csv") if isinstance(result.get("metrics_csv"), str) else None
            metrics_info = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        else:
            out_dir = str(result) if result is not None else None

        if not out_dir:
            out_dir = self._analysis_out_dir()

        base_msg = f"[INFO] 分析完成，图表保存在 {out_dir}"
        if summary_text:
            self.display_result(f"{base_msg}\n{summary_text}")
        else:
            self.display_result(base_msg)

        if "metrics_info" not in locals():
            metrics_info = {}
        if isinstance(metrics_info, dict) and metrics_info.get("drift_retrain"):
            reasons = metrics_info.get("drift_retrain_reasons") or []
            reason_text = "; ".join(str(item) for item in reasons if item)
            if not reason_text:
                reason_text = "分布漂移指标超出阈值"
            QtWidgets.QMessageBox.warning(
                None,
                "建议重新训练",
                f"检测到显著数据漂移：{reason_text}\n建议重新训练模型以适应最新数据。",
            )

        added_paths: Set[str] = set()

        def _mark(path: str):
            if path and os.path.exists(path) and path not in added_paths:
                self._add_output(path)
                added_paths.add(path)

        for p in plot_paths:
            _mark(p)

        if top20_csv:
            _mark(top20_csv)
            if pd is not None and os.path.exists(top20_csv):
                try:
                    df = read_csv_flexible(top20_csv)
                    if not df.empty:
                        self.populate_table_from_df(df)
                        self._last_out_csv = top20_csv
                        self._open_csv_paged(top20_csv)
                except Exception:
                    pass

        if locals().get("metrics_csv"):
            _mark(metrics_csv)

        fallback_names = [
            "top10_malicious_ratio.png",
            "anomaly_score_distribution.png",
            "top20_packets.csv",
        ]
        for name in fallback_names:
            _mark(os.path.join(out_dir, name))

        if summary_csv:
            _mark(summary_csv)
        else:
            default_summary = os.path.join(self._default_results_dir(), "summary_by_file.csv")
            if os.path.exists(default_summary):
                _mark(default_summary)

        if summary_json:
            _mark(summary_json)

        if out_dir and os.path.isdir(out_dir):
            _mark(out_dir)

        metadata_for_dashboard = self._selected_metadata if isinstance(self._selected_metadata, dict) else (result.get("metadata") if isinstance(result, dict) else None)
        self.dashboard.update_metrics(
            self._analysis_summary,
            metadata_for_dashboard if isinstance(metadata_for_dashboard, dict) else None,
        )

    def _on_analysis_error(self, msg):
        self.btn_analysis.setEnabled(True); self.reset_button_progress(self.btn_analysis)
        self._set_action_buttons_enabled(True)
        QtWidgets.QMessageBox.critical(None, "分析失败", msg)
        self.display_result(f"[错误] 分析失败: {msg}")
        self._analysis_summary = None

    # --------- 模型预测（支持 Pipeline / 模型+scaler / 仅模型） ----------
    def _predict_dataframe(
        self,
        df: "pd.DataFrame",
        *,
        source_name: str,
        output_dir: Optional[str] = None,
        metadata_override: Optional[dict] = None,
        silent: bool = False,
    ) -> dict:
        if pd is None:
            raise RuntimeError("pandas 未安装，无法执行预测。")

        def _format_row_messages(frame: "pd.DataFrame") -> List[str]:
            if pd is None or frame is None or not isinstance(frame, pd.DataFrame):
                return []
            max_preview = 200
            if len(frame) > max_preview:
                return []

            identifier_columns = [
                "flow_id",
                "__source_file__",
                "__source_path__",
                "pcap_file",
                "pcap_name",
                "file_name",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
            ]

            messages: List[str] = []
            for idx, row in frame.iterrows():
                identifier = None
                for column in identifier_columns:
                    if column not in frame.columns:
                        continue
                    value = row.get(column)
                    try:
                        is_valid = pd.notna(value)
                    except Exception:
                        is_valid = value not in (None, "")
                    if is_valid:
                        text_value = str(value).strip()
                        if text_value:
                            identifier = f"{column}={text_value}"
                            break
                if identifier is None:
                    identifier = f"第{idx + 1}行"

                label = row.get("prediction_label")
                if not label:
                    label = row.get("prediction")

                score_value = row.get("malicious_score")
                if score_value is None:
                    score_value = row.get("anomaly_score")
                try:
                    score_float = float(score_value)
                except Exception:
                    score_float = None

                if score_float is not None and not math.isnan(score_float):
                    score_text = f" 分数={score_float:.4f}"
                else:
                    score_text = ""

                messages.append(f"{identifier} -> 预测={label}{score_text}")

            return messages

        bundle = self._resolve_prediction_bundle(df, metadata_override=metadata_override)
        if not bundle:
            raise RuntimeError(
                "未找到与所选特征CSV匹配的模型，请确认已选择训练时对应的特征文件。"
            )

        pipeline_path = bundle.get("pipeline_path")
        metadata_obj = bundle.get("metadata") or {}
        metadata = dict(metadata_obj) if isinstance(metadata_obj, dict) else {}
        metadata_path = bundle.get("metadata_path")
        allowed_extra = set(bundle.get("allowed_extra") or set())
        extras_detected = list(bundle.get("extras") or [])
        source = bundle.get("source")

        if not pipeline_path or not os.path.exists(pipeline_path):
            raise RuntimeError("未找到可用的模型管线，请先训练模型。")

        try:
            pipeline = joblib_load(pipeline_path)
        except Exception as exc:
            raise RuntimeError(f"模型管线加载失败：{exc}")

        # 记录所选模型，确保后续操作保持一致
        if metadata_path:
            self._selected_metadata = metadata
            self._selected_metadata_path = metadata_path
            self._selected_pipeline_path = pipeline_path
            if hasattr(self, "model_combo"):
                idx = self.model_combo.findData(metadata_path)
                if idx < 0:
                    self._refresh_model_versions()
                    idx = self.model_combo.findData(metadata_path)
                if idx >= 0:
                    self.model_combo.blockSignals(True)
                    self.model_combo.setCurrentIndex(idx)
                    self.model_combo.blockSignals(False)
                    self._on_model_combo_changed(idx)

        feature_df_raw, align_info = _align_input_features(
            df,
            metadata,
            strict=False,
            allow_extra=allowed_extra.union(extras_detected),
        )

        messages: List[str] = []
        if source and source not in {"selected", "override"}:
            messages.append("已根据特征列自动匹配模型版本。")
        messages.append(f"使用模型管线: {os.path.basename(pipeline_path)}")
        if extras_detected:
            sample = ", ".join(sorted(extras_detected)[:8])
            more = " ..." if len(extras_detected) > 8 else ""
            messages.append(f"忽略了 {len(extras_detected)} 个额外列: {sample}{more}")
        missing_after_align = align_info.get("missing_filled") if isinstance(align_info, dict) else None
        if missing_after_align:
            missing_list = list(missing_after_align)
            sample_missing = ", ".join(missing_list[:8])
            more_missing = " ..." if len(missing_list) > 8 else ""
            raise RuntimeError(
                "检测到特征 CSV 缺少模型所需列，请确认选择的特征文件正确。\n缺少列: "
                f"{sample_missing}{more_missing}"
            )
        schema_version = align_info.get("schema_version")
        if schema_version and schema_version != MODEL_SCHEMA_VERSION:
            messages.append(f"模型 schema_version={schema_version} 与当前 {MODEL_SCHEMA_VERSION} 不一致")

        expected_order = align_info.get("feature_order") or []
        pipeline_features = getattr(pipeline, "feature_names_in_", None)
        if pipeline_features is not None:
            pipeline_cols = [str(col) for col in pipeline_features]
            if list(pipeline_cols) != list(expected_order):
                raise RuntimeError(
                    "模型管线的特征列顺序与训练时不一致，请重新训练或重新选择模型。"
                )

        if isinstance(pipeline, dict) and "model" in pipeline and "feature_names" in pipeline:
            feature_names = [str(name) for name in pipeline.get("feature_names", [])]
            if not feature_names:
                feature_names = list(expected_order)
            if feature_names and expected_order and list(feature_names) != list(expected_order):
                feature_df_raw = feature_df_raw.loc[:, feature_names]
            matrix = feature_df_raw.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)

            model = pipeline["model"]
            positive_keywords = {"异常", "恶意", "malicious", "anomaly", "attack", "1", "-1"}
            classes = list(getattr(model, "classes_", []))
            pos_idx = None
            meta_pos_label = ""
            meta_pos_class = None
            raw_label_mapping = pipeline.get("label_mapping")
            mapping: Optional[Dict[Union[int, str], str]] = None
            if isinstance(raw_label_mapping, dict):
                converted: Dict[Union[int, str], str] = {}
                for key, value in raw_label_mapping.items():
                    text_value = str(value)
                    converted[str(key)] = text_value
                    try:
                        converted[int(key)] = text_value
                    except (TypeError, ValueError):
                        continue
                mapping = converted
            if isinstance(metadata, dict):
                meta_pos_label = str(metadata.get("positive_label", "")).strip().lower()
                meta_pos_class = metadata.get("positive_class")

            def _label_for(cls_val):
                if mapping:
                    try:
                        mapped = mapping.get(int(cls_val))
                    except (TypeError, ValueError):
                        mapped = mapping.get(str(cls_val))
                    if mapped is not None:
                        return str(mapped).strip()
                return str(cls_val).strip()

            if meta_pos_class is not None and classes:
                for idx, cls_val in enumerate(classes):
                    if cls_val == meta_pos_class:
                        pos_idx = idx
                        break
                    try:
                        if isinstance(meta_pos_class, str) and isinstance(cls_val, (int, np.integer)):
                            if int(meta_pos_class) == int(cls_val):
                                pos_idx = idx
                                break
                    except (TypeError, ValueError):
                        pass
                    try:
                        if isinstance(meta_pos_class, (int, np.integer)) and int(meta_pos_class) == int(cls_val):
                            pos_idx = idx
                            break
                    except (TypeError, ValueError):
                        pass

            if pos_idx is None and meta_pos_label:
                for idx, cls_val in enumerate(classes):
                    if _label_for(cls_val).lower() == meta_pos_label:
                        pos_idx = idx
                        break

            if pos_idx is None and classes:
                for idx, cls_val in enumerate(classes):
                    if _label_for(cls_val).lower() in positive_keywords:
                        pos_idx = idx
                        break

            if pos_idx is None and classes:
                try:
                    if 1 in classes:
                        pos_idx = classes.index(1)
                    elif -1 in classes:
                        pos_idx = classes.index(-1)
                except ValueError:
                    pos_idx = None

            if pos_idx is None and classes:
                pos_idx = len(classes) - 1

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(matrix)
                if np.ndim(proba) == 2 and proba.shape[1] >= 2:
                    if pos_idx is not None and 0 <= pos_idx < proba.shape[1]:
                        scores = proba[:, pos_idx]
                    else:
                        scores = proba[:, -1]
                else:
                    scores = proba
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(matrix)
                if np.ndim(decision) == 1:
                    scores = 1.0 / (1.0 + np.exp(-decision))
                else:
                    if pos_idx is not None and decision.shape[1] > pos_idx:
                        margin = decision[:, pos_idx]
                    else:
                        margin = decision.max(axis=1)
                    scores = 1.0 / (1.0 + np.exp(-margin))
            else:
                raw_pred = model.predict(matrix)
                scores = np.asarray(raw_pred, dtype=float)

            preds = model.predict(matrix)

            if summarize_prediction_labels is not None:
                labels, anomaly_count, normal_count, status_text = summarize_prediction_labels(
                    preds,
                    mapping,
                )
            else:
                labels = [
                    mapping.get(int(value), str(value)) if mapping is not None else str(value)
                    for value in preds
                ]
                anomaly_count = None
                normal_count = None
                status_text = None
                if labels:
                    abnormal = sum(1 for label in labels if str(label) == "异常")
                    normal = sum(1 for label in labels if str(label) == "正常")
                    if abnormal:
                        anomaly_count = abnormal
                        normal_count = normal
                        status_text = "异常"
                    elif normal:
                        anomaly_count = 0
                        normal_count = normal
                        status_text = "正常"

            out_df = df.copy()
            out_df["prediction"] = [int(value) if isinstance(value, (int, np.integer)) else value for value in preds]
            out_df["prediction_label"] = labels
            out_df["malicious_score"] = [float(value) for value in np.asarray(scores, dtype=float)]
            if status_text is not None:
                out_df["prediction_status"] = [
                    label if label in {"异常", "正常"} else status_text for label in labels
                ]

            if output_dir is None:
                output_dir = self._prediction_out_dir()
            os.makedirs(output_dir, exist_ok=True)
            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_name)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = os.path.join(output_dir, f"prediction_{safe_name}_{stamp}.csv")
            out_df.to_csv(out_csv, index=False, encoding="utf-8")

            total_rows = int(len(out_df))
            summary_lines = [
                f"模型预测完成：{source_name}",
                f"输出行数：{total_rows}",
            ]
            if status_text is not None:
                summary_lines.append(f"分析结论：{status_text}")
            ratio: Optional[float] = None
            if anomaly_count is not None:
                summary_line = f"异常数量：{int(anomaly_count)}"
                if normal_count is not None:
                    summary_line += f"，正常数量：{int(normal_count)}"
                summary_lines.append(summary_line)
                if total_rows:
                    ratio = float(anomaly_count) / float(total_rows)
            messages.extend(summary_lines)
            if not silent:
                for msg in summary_lines:
                    self.display_result(f"[INFO] {msg}")

            row_messages = _format_row_messages(out_df)

            return {
                "output_csv": out_csv,
                "dataframe": out_df,
                "summary": summary_lines,
                "messages": messages,
                "metadata": metadata,
                "malicious": int(anomaly_count) if anomaly_count is not None else None,
                "total": total_rows,
                "ratio": ratio,
                "status_text": status_text,
                "anomaly_count": int(anomaly_count) if anomaly_count is not None else None,
                "normal_count": int(normal_count) if normal_count is not None else None,
                "predictions": [int(value) if isinstance(value, (int, np.integer)) else value for value in preds],
                "scores": [float(value) for value in np.asarray(scores, dtype=float)],
                "labels": labels,
                "row_messages": row_messages,
            }

        models_dir = self._default_models_dir()
        preproc_candidates: List[str] = []
        if isinstance(metadata, dict):
            if metadata.get("preprocessor_latest"):
                preproc_candidates.append(metadata.get("preprocessor_latest"))
            if metadata.get("preprocessor_path"):
                preproc_candidates.append(metadata.get("preprocessor_path"))
        preproc_candidates.append(os.path.join(models_dir, "latest_preprocessor.joblib"))

        loaded_preprocessor = None
        for path in preproc_candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                loaded_preprocessor = joblib_load(path)
                break
            except Exception:
                continue
        if loaded_preprocessor is None and hasattr(pipeline, "named_steps"):
            loaded_preprocessor = pipeline.named_steps.get("preprocessor")
        if loaded_preprocessor is None:
            raise RuntimeError("模型缺少特征预处理器。")

        try:
            feature_df_aligned = loaded_preprocessor.transform(feature_df_raw)
        except Exception as exc:
            raise RuntimeError(f"特征预处理失败：{exc}")

        named_steps = getattr(pipeline, "named_steps", {}) if hasattr(pipeline, "named_steps") else {}
        detector = named_steps.get("detector") if isinstance(named_steps, dict) else None
        if detector is None:
            raise RuntimeError("当前模型缺少集成检测器，请重新训练。")

        transformed = feature_df_aligned
        for name, step in pipeline.steps[1:-1]:
            try:
                transformed = step.transform(transformed)
            except Exception as exc:
                raise RuntimeError(f"特征变换失败（{name}）：{exc}")

        try:
            preds = detector.predict(transformed)
        except Exception as exc:
            raise RuntimeError(f"预测失败：{exc}")

        scores = getattr(detector, "last_combined_scores_", None)
        if scores is None:
            try:
                scores = detector.score_samples(transformed)
            except Exception:
                scores = np.zeros(len(feature_df_aligned), dtype=float)
        scores = np.asarray(scores, dtype=float)

        vote_ratio = getattr(detector, "last_vote_ratio_", None)
        if vote_ratio is None:
            vote_blocks = []
            for info in getattr(detector, "detectors_", {}).values():
                est = getattr(info, "estimator", None)
                if est is None or not hasattr(est, "predict"):
                    continue
                try:
                    sub_pred = est.predict(transformed)
                    vote_blocks.append(np.where(sub_pred == -1, 1.0, 0.0))
                except Exception:
                    continue
            if vote_blocks:
                vote_ratio = np.vstack(vote_blocks).mean(axis=0)
        if vote_ratio is None:
            vote_mean_meta = metadata.get("vote_mean") if isinstance(metadata, dict) else None
            fallback_vote = 0.5 if vote_mean_meta is None else float(vote_mean_meta)
            vote_ratio = np.full(len(feature_df_aligned), fallback_vote, dtype=float)
        vote_ratio = np.asarray(vote_ratio, dtype=float)

        threshold_breakdown_meta = metadata.get("threshold_breakdown") if isinstance(metadata, dict) else None
        threshold = None
        if isinstance(threshold_breakdown_meta, dict) and threshold_breakdown_meta.get("adaptive") is not None:
            threshold = threshold_breakdown_meta.get("adaptive")
        elif isinstance(metadata, dict):
            threshold = metadata.get("threshold")
        if threshold is None:
            threshold = getattr(detector, "threshold_", None)
        if threshold is None:
            threshold = float(np.quantile(scores, 0.05)) if len(scores) else 0.0

        vote_threshold = metadata.get("vote_threshold") if isinstance(metadata, dict) else None
        if vote_threshold is None:
            vote_threshold = getattr(detector, "vote_threshold_", None)
        if vote_threshold is None:
            vote_threshold = float(np.mean(vote_ratio)) if len(vote_ratio) else 0.5
        vote_threshold = float(np.clip(vote_threshold, 0.0, 1.0))

        score_std = metadata.get("score_std") if isinstance(metadata, dict) else None
        if score_std is None:
            score_std = float(np.std(scores) or 1.0)

        conf_from_score = 1.0 / (1.0 + np.exp((scores - threshold) / (score_std + 1e-6)))
        vote_component = np.clip((vote_ratio - vote_threshold) / max(1e-6, (1.0 - vote_threshold)), 0.0, 1.0)
        risk_score = np.clip(0.6 * conf_from_score + 0.4 * vote_component, 0.0, 1.0)

        supervised_scores = getattr(detector, "last_supervised_scores_", None)
        if supervised_scores is not None:
            risk_score = np.clip(0.5 * risk_score + 0.5 * supervised_scores.astype(float), 0.0, 1.0)
        elif getattr(detector, "last_calibrated_scores_", None) is not None:
            risk_score = np.clip(
                0.6 * risk_score + 0.4 * detector.last_calibrated_scores_.astype(float),
                0.0,
                1.0,
            )

        out_df = df.copy()
        out_df["prediction"] = preds
        out_df["is_malicious"] = (preds == -1).astype(int)
        out_df["anomaly_score"] = scores
        out_df["anomaly_confidence"] = risk_score
        out_df["vote_ratio"] = vote_ratio
        out_df["risk_score"] = risk_score

        malicious = int(out_df["is_malicious"].sum())
        total = int(len(out_df))
        ratio = (malicious / total) if total else 0.0

        score_min = float(np.min(scores)) if len(scores) else 0.0
        score_max = float(np.max(scores)) if len(scores) else 0.0

        summary_lines = [
            f"模型预测完成：{source_name}",
            f"检测包数：{total}",
            f"异常包数：{malicious} ({ratio:.2%})",
            f"自动阈值：{float(threshold):.6f}",
            f"投票阈值：{vote_threshold:.2f}",
            f"分数范围：{score_min:.4f} ~ {score_max:.4f}",
            f"平均风险分：{float(risk_score.mean()):.2%}",
        ]

        if output_dir is None:
            output_dir = self._prediction_out_dir()
        os.makedirs(output_dir, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(output_dir, f"prediction_{safe_name}_{stamp}.csv")
        out_df.to_csv(out_csv, index=False, encoding="utf-8")

        messages.extend(summary_lines)
        if not silent:
            for msg in messages:
                self.display_result(f"[INFO] {msg}")

        row_messages = _format_row_messages(out_df)

        return {
            "output_csv": out_csv,
            "dataframe": out_df,
            "summary": summary_lines,
            "messages": messages,
            "metadata": metadata,
            "malicious": malicious,
            "total": total,
            "ratio": ratio,
            "row_messages": row_messages,
        }


    def _present_prediction_result(
        self,
        result: Optional[dict],
        *,
        source_name: str,
        metadata_override: Optional[dict] = None,
        source_csv: Optional[str] = None,
        source_pcap: Optional[str] = None,
        show_dialog: bool = True,
    ) -> None:
        if not isinstance(result, dict):
            return

        parent_widget = self._parent_widget()

        messages = result.get("messages") or []
        if messages:
            lines = [f"[INFO] [{source_name}] {line}" for line in messages]
            self.display_result("\n".join(lines))

        row_messages = result.get("row_messages") or []
        if row_messages:
            detailed_lines = [f"[INFO] [{source_name}] {line}" for line in row_messages]
            self.display_result("\n".join(detailed_lines))

        output_csv = result.get("output_csv")
        if output_csv and os.path.exists(output_csv):
            self._add_output(output_csv)
            self._last_out_csv = output_csv
            if show_dialog:
                self._open_csv_paged(output_csv)
            self._auto_analyze(output_csv)

        dataframe = result.get("dataframe")
        if isinstance(dataframe, pd.DataFrame):
            self.populate_table_from_df(dataframe)

        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else metadata_override
        ratio = result.get("ratio")
        analysis_stub = None
        if ratio is not None:
            analysis_stub = {
                "metrics": {
                    "malicious_ratio": ratio,
                    "anomaly_count": int(result.get("malicious", 0)),
                    "total_count": int(result.get("total", 0)),
                }
            }

        self.dashboard.update_metrics(analysis_stub, metadata)

        snapshot = {key: value for key, value in result.items() if key != "dataframe"}
        if isinstance(dataframe, pd.DataFrame):
            snapshot["dataframe"] = dataframe
        if source_csv:
            snapshot["source_csv"] = source_csv
        if source_pcap:
            snapshot["source_pcap"] = source_pcap
        if metadata and "metadata" not in snapshot:
            snapshot["metadata"] = metadata
        snapshot["source_name"] = source_name
        self._latest_prediction_summary = snapshot

        if show_dialog:
            QtWidgets.QMessageBox.information(
                parent_widget,
                "预测完成",
                "\n".join(result.get("summary") or ["模型预测已完成并写出结果。"]),
            )

    def _predict_pcap_batch(
        self,
        paths: List[str],
        *,
        metadata_override: Optional[dict],
    ) -> None:
        parent_widget = self._parent_widget()
        if not paths:
            QtWidgets.QMessageBox.information(parent_widget, "无有效文件", "所选目录中没有可处理的 PCAP 文件。")
            return

        button = getattr(self, "btn_predict", None)
        if button is not None:
            button.setEnabled(False)
            try:
                self.set_button_progress(button, 1)
            except Exception:
                pass
            QtWidgets.QApplication.processEvents()

        output_dir = self._prediction_out_dir()
        os.makedirs(output_dir, exist_ok=True)

        total = len(paths)
        successes = 0
        total_rows = 0
        total_malicious = 0
        failed: List[Tuple[str, str]] = []

        try:
            for index, pcap_path in enumerate(paths, start=1):
                self.display_result(f"[INFO] 处理 PCAP ({index}/{total}): {pcap_path}")

                def _update_progress(local_pct: int) -> None:
                    if button is None:
                        return
                    try:
                        local_value = max(0, min(100, int(local_pct)))
                    except Exception:
                        local_value = 0
                    completed = (index - 1) + (local_value / 100.0)
                    overall = int(max(1, min(100, (completed / float(total)) * 100.0)))
                    try:
                        self.set_button_progress(button, overall)
                    except Exception:
                        pass
                    QtWidgets.QApplication.processEvents()

                try:
                    payload = self._process_online_pcap(
                        pcap_path,
                        output_dir=output_dir,
                        metadata=metadata_override,
                        progress_cb=_update_progress,
                    )
                except Exception as exc:
                    reason = str(exc)
                    failed.append((pcap_path, reason))
                    self.display_result(f"[错误] PCAP 处理失败：{pcap_path} -> {reason}")
                    continue

                prediction = payload.get("prediction") if isinstance(payload, dict) else None
                source_name = os.path.basename(pcap_path)
                self._present_prediction_result(
                    prediction,
                    source_name=source_name,
                    metadata_override=metadata_override,
                    source_pcap=pcap_path,
                    show_dialog=False,
                )

                successes += 1
                try:
                    total_rows += int(prediction.get("total", 0))  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    total_malicious += int(prediction.get("malicious", 0))  # type: ignore[arg-type]
                except Exception:
                    pass

            if successes == 0:
                error_lines = [f"- {os.path.basename(path)}: {reason}" for path, reason in failed[:5]]
                if len(failed) > 5:
                    error_lines.append("...")
                QtWidgets.QMessageBox.critical(
                    parent_widget,
                    "预测失败",
                    "所有 PCAP 文件处理均失败。\n" + "\n".join(error_lines),
                )
                return

            summary_lines = [
                f"共处理 PCAP 文件：{total} 个",
                f"成功：{successes} 个，失败：{total - successes} 个",
                f"累计检测流量：{total_rows} 条",
                f"检测到异常：{total_malicious} 条",
            ]
            if failed:
                sample = "; ".join(
                    f"{os.path.basename(path)}: {reason}" for path, reason in failed[:3]
                )
                if len(failed) > 3:
                    sample += " ..."
                summary_lines.append(f"失败样例：{sample}")

            QtWidgets.QMessageBox.information(parent_widget, "预测完成", "\n".join(summary_lines))
        finally:
            if button is not None:
                button.setEnabled(True)
                if successes > 0:
                    try:
                        self.set_button_progress(button, 100)
                    except Exception:
                        pass
                    QtCore.QTimer.singleShot(300, lambda b=button: self.reset_button_progress(b))
                else:
                    self.reset_button_progress(button)

    def _on_predict(self):
        parent_widget = self._parent_widget()
        if pd is None:
            QtWidgets.QMessageBox.warning(parent_widget, "缺少依赖", "当前环境未安装 pandas，无法执行预测。")
            return

        selected_df: Optional["pd.DataFrame"] = None
        selection_model = getattr(self.table_view, "selectionModel", None)
        if callable(selection_model):
            selection_model = selection_model()
        if selection_model is not None:
            try:
                selected_indexes = selection_model.selectedRows()
            except Exception:
                selected_indexes = []
            if selected_indexes:
                model = self.table_view.model()
                source_model = model
                if isinstance(model, QtCore.QSortFilterProxyModel):
                    source_model = model.sourceModel()
                df_source = getattr(source_model, "_df", None)
                if isinstance(df_source, pd.DataFrame) and not df_source.empty:
                    row_numbers: List[int] = []
                    for index in selected_indexes:
                        source_index = index
                        if isinstance(model, QtCore.QSortFilterProxyModel):
                            source_index = model.mapToSource(index)
                        if source_index.isValid():
                            row_numbers.append(source_index.row())
                    if row_numbers:
                        valid_rows = sorted({row for row in row_numbers if 0 <= row < len(df_source)})
                        if valid_rows:
                            selected_df = df_source.iloc[valid_rows].copy()
                            drop_columns = {
                                "prediction",
                                "prediction_label",
                                "prediction_status",
                                "malicious_score",
                                "anomaly_score",
                                "anomaly_confidence",
                                "vote_ratio",
                                "risk_score",
                                "is_malicious",
                                "manual_label",
                            }
                            try:
                                selected_df.drop(columns=[col for col in drop_columns if col in selected_df], inplace=True, errors="ignore")
                            except TypeError:
                                # pandas<1.0 无 errors 参数
                                for col in list(drop_columns):
                                    if col in selected_df:
                                        selected_df.drop(columns=[col], inplace=True)
                            selected_df.reset_index(drop=True, inplace=True)

        if selected_df is not None and not selected_df.empty:
            metadata_override = None
            if isinstance(self._selected_metadata, dict):
                metadata_override = self._selected_metadata
            elif isinstance(self._latest_prediction_summary, dict):
                meta_candidate = self._latest_prediction_summary.get("metadata")
                if isinstance(meta_candidate, dict):
                    metadata_override = meta_candidate

            self.display_result(f"[INFO] 使用已选流量 {len(selected_df)} 条执行模型预测。")

            button = getattr(self, "btn_predict", None)
            if button is not None:
                button.setEnabled(False)
                try:
                    self.set_button_progress(button, 1)
                except Exception:
                    pass
                QtWidgets.QApplication.processEvents()

            try:
                result = self._predict_dataframe(
                    selected_df,
                    source_name=f"选中流量({len(selected_df)})",
                    metadata_override=metadata_override,
                    silent=True,
                )
            except Exception as exc:
                if button is not None:
                    button.setEnabled(True)
                    self.reset_button_progress(button)
                QtWidgets.QMessageBox.critical(parent_widget, "预测失败", str(exc))
                return

            if button is not None:
                button.setEnabled(True)
                try:
                    self.set_button_progress(button, 100)
                except Exception:
                    pass
                QtCore.QTimer.singleShot(300, lambda b=button: self.reset_button_progress(b))

            self._present_prediction_result(
                result,
                source_name=f"选中流量({len(selected_df)})",
                metadata_override=metadata_override,
                show_dialog=True,
            )
            return

        preferred_pcap_dir = r"D:\pythonProject8\data\split"
        start_dir = preferred_pcap_dir if os.path.exists(preferred_pcap_dir) else self._default_split_dir()
        selected_files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "选择 PCAP 流量（可多选）",
            start_dir,
            "PCAP (*.pcap *.pcapng);;所有文件 (*)",
        )

        chosen_path: Optional[str]
        if selected_files:
            normalized_files = [os.path.normpath(p.strip()) for p in selected_files if p]
            valid_files = [
                p
                for p in normalized_files
                if p
                and os.path.exists(p)
                and os.path.isfile(p)
                and p.lower().endswith((".pcap", ".pcapng"))
            ]

            if not valid_files:
                QtWidgets.QMessageBox.warning(
                    parent_widget,
                    "无有效文件",
                    "请选择存在的 PCAP/PCAPNG 文件。",
                )
                return

            metadata_override = (
                self._selected_metadata if isinstance(self._selected_metadata, dict) else None
            )

            deduped_files: List[str] = []
            seen: Set[str] = set()
            for path in valid_files:
                if path not in seen:
                    deduped_files.append(path)
                    seen.add(path)

            valid_files = deduped_files

            if len(valid_files) > 1:
                self._remember_path(os.path.dirname(valid_files[0]))
                self._predict_pcap_batch(
                    valid_files,
                    metadata_override=metadata_override,
                )
                return

            chosen_path = valid_files[0]
        else:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                None, "选择 PCAP 所在目录", start_dir
            )
            if not dir_path:
                return
            chosen_path = dir_path

        chosen_path = chosen_path.strip()
        if not chosen_path:
            return

        if not os.path.exists(chosen_path):
            QtWidgets.QMessageBox.warning(parent_widget, "路径不存在", chosen_path)
            return

        metadata_override = (
            self._selected_metadata if isinstance(self._selected_metadata, dict) else None
        )

        if os.path.isdir(chosen_path):
            pcap_candidates = self._list_sorted(chosen_path)

            if not pcap_candidates:
                QtWidgets.QMessageBox.information(
                    parent_widget,
                    "未发现数据",
                    "所选目录没有可用的 PCAP 文件。",
                )
                return

            self._remember_path(chosen_path)
            self._predict_pcap_batch(
                pcap_candidates,
                metadata_override=metadata_override,
            )
            return

        if not os.path.isdir(chosen_path):
            self._remember_path(chosen_path)

        if chosen_path.lower().endswith((".pcap", ".pcapng")):
            button = getattr(self, "btn_predict", None)
            if button is not None:
                button.setEnabled(False)
                try:
                    self.set_button_progress(button, 1)
                except Exception:
                    pass
                QtWidgets.QApplication.processEvents()

            def _single_progress(local_pct: int) -> None:
                if button is None:
                    return
                try:
                    pct_value = max(0, min(100, int(local_pct)))
                except Exception:
                    pct_value = 0
                pct_value = max(1, pct_value)
                try:
                    self.set_button_progress(button, pct_value)
                except Exception:
                    pass
                QtWidgets.QApplication.processEvents()

            try:
                payload = self._process_online_pcap(
                    chosen_path,
                    output_dir=self._prediction_out_dir(),
                    metadata=metadata_override,
                    progress_cb=_single_progress,
                )
            except Exception as exc:
                if button is not None:
                    button.setEnabled(True)
                    self.reset_button_progress(button)
                QtWidgets.QMessageBox.critical(parent_widget, "预测失败", str(exc))
                return

            if button is not None:
                button.setEnabled(True)
                try:
                    self.set_button_progress(button, 100)
                except Exception:
                    pass
                QtCore.QTimer.singleShot(300, lambda b=button: self.reset_button_progress(b))

            prediction = payload.get("prediction") if isinstance(payload, dict) else None
            source_name = os.path.basename(chosen_path)
            self._present_prediction_result(
                prediction,
                source_name=source_name,
                metadata_override=metadata_override,
                source_pcap=chosen_path,
                show_dialog=True,
            )
            return

        try:
            df = read_csv_flexible(chosen_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "读取失败", f"无法读取 CSV：{exc}")
            return

        if df.empty:
            QtWidgets.QMessageBox.information(parent_widget, "空数据", "该 CSV 没有数据行。")
            return

        source_name = os.path.splitext(os.path.basename(chosen_path))[0] or os.path.basename(chosen_path)

        try:
            result = self._predict_dataframe(
                df,
                source_name=source_name,
                metadata_override=metadata_override,
                silent=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(parent_widget, "预测失败", str(exc))
            return

        self._present_prediction_result(
            result,
            source_name=source_name,
            metadata_override=metadata_override,
            source_csv=chosen_path,
            show_dialog=True,
        )

    # --------- 输出列表 ----------

    def _add_output(self, path):
        if not path: return
        it = QtWidgets.QListWidgetItem(os.path.basename(path))
        it.setToolTip(path); it.setData(QtCore.Qt.UserRole, path)
        self.output_list.addItem(it); self.output_list.scrollToBottom()

    def _open_results_dir(self):
        self._reveal_in_folder(self._default_results_dir())

    def _open_logs_dir(self):
        parent_widget = self._parent_widget()
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            sample_file = next((p for p in LOGS_DIR.rglob("*") if p.is_file()), None)
        except Exception:
            sample_file = None

        try:
            dialog = LogViewerDialog(
                LOGS_DIR,
                reveal_callback=self._reveal_in_folder,
                parent=parent_widget,
            )
            if dialog.exec_() == 0 and sample_file is None:
                QtWidgets.QMessageBox.information(
                    None,
                    "暂无日志",
                    "当前日志目录为空，已自动为您打开日志目录。",
                )
                self._reveal_in_folder(str(LOGS_DIR))
            return
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                None,
                "查看失败",
                f"内置日志查看器无法启动：{exc}\n已尝试直接打开日志目录。",
            )

        self._reveal_in_folder(str(LOGS_DIR))

    def _on_output_double_click(self, it):
        self._reveal_in_folder(it.data(QtCore.Qt.UserRole))

    def _on_output_ctx_menu(self, pos):
        it = self.output_list.itemAt(pos)
        if not it: return
        p = it.data(QtCore.Qt.UserRole)
        m = QtWidgets.QMenu(self.output_list)
        a1 = m.addAction("复制完整路径")
        a2 = m.addAction("将此路径写入上方输入框")
        a3 = m.addAction("在资源管理器中显示")
        act = m.exec_(self.output_list.mapToGlobal(pos))
        if act == a1: QtWidgets.QApplication.clipboard().setText(p)
        elif act == a2: self.file_edit.setText(p)
        elif act == a3: self._reveal_in_folder(p)

    def _reveal_in_folder(self, path):
        try:
            sysname = platform.system().lower()
            if sysname.startswith("win"):
                if os.path.isdir(path): os.startfile(path)
                else: subprocess.run(["explorer", "/select,", os.path.normpath(path)])
            elif sysname == "darwin":
                subprocess.run(["open", "-R", path])
            else:
                subprocess.run(["xdg-open", path if os.path.isdir(path) else os.path.dirname(path)])
        except Exception as e:
            QtWidgets.QMessageBox.warning(None, "打开失败", f"无法打开资源管理器：{e}")

    def _remember_path(self, path: str) -> None:
        if not path or not hasattr(self, "_settings"):
            return
        try:
            self._settings.set("last_input_path", path)
        except Exception:
            pass

    def _on_training_settings_changed(self):
        return

    def _apply_saved_preferences(self):
        if not hasattr(self, "_settings"):
            self._settings_ready = True
            return
        self._loading_settings = True
        try:
            last_path = self._settings.get("last_input_path")
            if isinstance(last_path, str) and last_path:
                self.file_edit.setText(last_path)
        finally:
            self._loading_settings = False
            self._settings_ready = True

    def shutdown(self) -> None:
        try:
            self._cancel_running()
        except Exception:
            pass

        if getattr(self, "preprocess_worker", None) and self.preprocess_worker.isRunning():
            try:
                self.preprocess_worker.requestInterruption()
                self.preprocess_worker.wait(1000)
            except Exception:
                pass

        worker = getattr(self, "_online_worker", None)
        if worker and worker.isRunning():
            try:
                worker.stop()
                worker.wait(2000)
            except Exception:
                pass
        self._online_worker = None
        self._online_output_dir = None

    # --------- 清空 ----------
    def _on_clear(self):
        if getattr(self, "worker", None) and self.worker.isRunning():
            self.worker.requestInterruption()
        if getattr(self, "preprocess_worker", None) and self.preprocess_worker.isRunning():
            self.preprocess_worker.requestInterruption()

        self.results_text.setUpdatesEnabled(False)
        self.table_view.setUpdatesEnabled(False)
        self.output_list.setUpdatesEnabled(False)
        try:
            self.results_text.clear()
            self.table_view.setSortingEnabled(False)
            self.table_view.setModel(None)
            if getattr(self, "_table_proxy", None) is not None:
                try:
                    self._table_proxy.deleteLater()
                except Exception:
                    pass
                self._table_proxy = None
            if getattr(self, "_table_model", None) is not None:
                try:
                    self._table_model.deleteLater()
                except Exception:
                    pass
                self._table_model = None
            self.display_tabs.setCurrentWidget(self.results_widget)
            self.output_list.clear()
            self._update_status_message("@2025 恶意流量检测系统")
            self._csv_paged_path = None
            self._csv_total_rows = None
            self._csv_current_page = 1
            self.page_info.setText("第 0/0 页")
            self._last_preview_df = None
            self._last_out_csv = None
            self.reset_button_progress(self.btn_view)
            self.reset_button_progress(self.btn_fe)
            self.reset_button_progress(self.btn_vector)
            self.reset_button_progress(self.btn_train)
            self.reset_button_progress(self.btn_analysis)
        finally:
            self.results_text.setUpdatesEnabled(True)
            self.table_view.setUpdatesEnabled(True)
            self.output_list.setUpdatesEnabled(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "恶意流量检测系统 — 主功能页面"))


class MainWindow(QtWidgets.QMainWindow):
    """PyQt 主窗口封装，方便直接运行该脚本启动 GUI。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._ui_settings = QtCore.QSettings("Maldet", "UI")

        geometry = self._ui_settings.value("geometry")
        if geometry:
            try:
                self.restoreGeometry(geometry)
            except TypeError:
                self.restoreGeometry(QtCore.QByteArray(geometry))

        splitter_state = self._ui_settings.value("splitter")
        if splitter_state:
            try:
                self.ui.splitter.restoreState(splitter_state)
            except TypeError:
                self.ui.splitter.restoreState(QtCore.QByteArray(splitter_state))

    def closeEvent(self, event):
        settings = getattr(self, "_ui_settings", None)
        if settings is None:
            settings = QtCore.QSettings("Maldet", "UI")
        try:
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("splitter", self.ui.splitter.saveState())
            settings.sync()
        except Exception:
            pass
        try:
            self.ui.shutdown()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    """启动 Qt 应用并展示主窗口。"""
    try:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        policy = getattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy", None)
        if policy:
            QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(policy.PassThrough)
    except Exception:
        pass

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True
    else:
        owns_app = False

    window = MainWindow()
    window.showMaximized()

    if owns_app:
        return app.exec_()

    # 若外部已有 QApplication，则仅返回 0，保持兼容
    return 0


if __name__ == "__main__":
    sys.exit(main())