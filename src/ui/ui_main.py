# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os, platform, subprocess, math, shutil, json, io, time, textwrap
from pathlib import Path
import numpy as np
from typing import Collection, Dict, List, Optional, Set
from datetime import datetime

import yaml
import matplotlib
matplotlib.use("Agg")
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
try:
    from src.functions.unsupervised_train import (
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
from src.functions.analyze_results import analyze_results as run_analysis
from src.functions.preprocess import preprocess_feature_dir as preprocess_dir
from src.functions.annotations import (
    upsert_annotation,
    annotation_summary,
    apply_annotations_to_frame,
)

try:
    import pandas as pd
except Exception:
    pd = None

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


APP_STYLE = """
/* 全局 */
QWidget {
  background-color: #F5F6FA;
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI";
  font-size: 14px;
  color: #1F1F1F;
}

/* 顶部标题栏 */
#TitleBar {
  background: #F5F6FA;
  border-bottom: 1px solid #E5E7EB;
}
#pageTitle {
  font-family: "Microsoft YaHei UI", "PingFang SC", "Segoe UI";
  font-size: 18px;
  font-weight: 700;
  color: #111827;
}

/* 卡片/分组 */
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

/* 输入控件 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
  background: #FFFFFF;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  padding: 8px 10px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
  border: 1px solid #60A5FA;
}

/* 主按钮（执行类） */
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

/* 次要按钮/分页按钮 */
QPushButton#secondary {
  background: #EEF1F6;
  color: #111827;
  border-radius: 10px;
  padding: 8px 16px;
  min-height: 38px;
}
QPushButton#secondary:hover { background: #E5EAF1; }
QPushButton#secondary:pressed { background: #D9E0EA; }

/* 表格 */
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

/* 分隔条 */
QSplitter::handle {
  background-color: #E5E7EB;
  width: 2px;
}

/* 状态栏 */
QStatusBar {
  background: #F3F4F6;
  border-top: 1px solid #E5E7EB;
  font-size: 12px;
  color: #6B7280;
  padding: 4px 8px;
}

/* 列表/文本区域 */
QTextEdit, QPlainTextEdit, QListWidget {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
}

/* 工具按钮 & 滚动区 */
QToolButton {
  background: transparent;
  border: none;
  padding: 4px;
}
QToolButton:hover {
  background: rgba(14, 165, 233, 0.1);
  border-radius: 6px;
}
QScrollArea {
  border: none;
}

/* 在线状态徽标 */
#OnlineStatusLabel {
  font-size: 12px;
  color: #5F6368;
  padding: 6px 8px;
  border-radius: 6px;
  background-color: #F3F4F6;
}
"""

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
    }
)
if "results" in PATHS and "results_analysis" not in PATHS:
    PATHS["results_analysis"] = PATHS["results"] / "analysis"
if "results" in PATHS and "results_pred" not in PATHS:
    PATHS["results_pred"] = PATHS["results"] / "modelprediction"
if "results" in PATHS and "results_abnormal" not in PATHS:
    PATHS["results_abnormal"] = PATHS["results"] / "abnormal"
for key in ("split", "csv_info", "csv_feature", "csv_preprocess", "models", "results_analysis", "results_pred", "results_abnormal"):
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
        }[key]
    PATHS[key].mkdir(parents=True, exist_ok=True)

default_logs = PATHS.get("logs", DATA_BASE / "logs")
logs_env = os.getenv("MALDET_LOG_DIR")
LOGS_DIR = Path(logs_env).expanduser().resolve() if logs_env else Path(default_logs).expanduser().resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = DATA_BASE / "settings.json"

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


class AppSettings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, object] = {}
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

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self._save()

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

# =============== 表格模型与行高亮 ===============
class PandasFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, df: "pd.DataFrame", parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, p=QtCore.QModelIndex()):
        return 0 if p.isValid() else len(self._df)

    def columnCount(self, p=QtCore.QModelIndex()):
        return 0 if p.isValid() else len(self._df.columns)

    def data(self, idx, role=QtCore.Qt.DisplayRole):
        if not idx.isValid():
            return None
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.ToolTipRole):
            try:
                v = self._df.iat[idx.row(), idx.column()]
                if pd is not None and pd.isna(v):
                    return ""
                return "" if v is None else str(v)
            except Exception:
                return ""
        return None

    def headerData(self, sec, ori, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return None
        if ori == QtCore.Qt.Horizontal:
            try:
                return str(self._df.columns[sec])
            except Exception:
                return str(sec)
        return str(sec)


class RowHighlighter(QtWidgets.QStyledItemDelegate):
    def __init__(self, df_provider, parent=None):
        super().__init__(parent)
        self.df_provider = df_provider

    def paint(self, painter, option, index):
        df = self.df_provider()
        if df is not None and "__TAG__" in df.columns:
            r = index.row()
            try:
                tag = str(df.iloc[r].get("__TAG__", "")).strip()
            except Exception:
                tag = ""
            if tag:
                painter.save()
                painter.fillRect(option.rect, QtGui.QColor(255, 204, 204))
                painter.restore()
        super().paint(painter, option, index)

    def helpEvent(self, event, view, option, index):
        df = self.df_provider()
        if df is not None and "__TAG__" in df.columns:
            r = index.row()
            try:
                tag = str(df.iloc[r].get("__TAG__", "")).strip()
            except Exception:
                tag = ""
            if tag:
                QtWidgets.QToolTip.showText(event.globalPos(), f"自动标注：{tag}")
                return True
        return super().helpEvent(event, view, option, index)


# =============== 后台线程 ===============
class InfoWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        try:
            kw = dict(self.kwargs)
            kw["progress_cb"] = self.progress.emit
            kw["cancel_cb"] = self.isInterruptionRequested
            df = info(**kw)
            self.progress.emit(100)
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class FeatureWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, pcap_path, csv_path):
        super().__init__()
        self.pcap_path = pcap_path
        self.csv_path = csv_path

    def run(self):
        try:
            fe_single(self.pcap_path, self.csv_path, progress_cb=self.progress.emit)
            self.finished.emit(self.csv_path)
        except Exception as e:
            self.error.emit(str(e))


class DirFeatureWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, split_dir, out_dir, workers=8):
        super().__init__()
        self.split_dir = split_dir
        self.out_dir = out_dir
        self.workers = workers

    def run(self):
        try:
            csvs = fe_dir(self.split_dir, self.out_dir, workers=self.workers, progress_cb=self.progress.emit)
            self.finished.emit(csvs)
        except Exception as e:
            self.error.emit(str(e))


class PreprocessWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, feature_source, out_dir):
        super().__init__()
        self.feature_source = feature_source
        self.out_dir = out_dir

    def run(self):
        try:
            result = preprocess_dir(self.feature_source, self.out_dir, progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class WorkerSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)


class BackgroundTask(QtCore.QRunnable):
    def __init__(self, fn, *args, **kwargs):
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

    def run(self):
        kwargs = dict(self._kwargs)

        def _emit_progress(value):
            try:
                value_int = int(float(value))
            except Exception:
                value_int = 0
            self.signals.progress.emit(max(0, min(100, value_int)))

        kwargs["progress_cb"] = _emit_progress
        try:
            result = self._fn(*self._args, **kwargs)
        except Exception as exc:
            self.signals.error.emit(str(exc))
            return

        self.signals.finished.emit(result)

# ======= 交互式列选择（用于 scaler+model 场景）=======
class FeaturePickDialog(QtWidgets.QDialog):
    def __init__(self, all_columns: List[str], need_k: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"选择并排序特征列（需要 {need_k} 列）")
        self.resize(700, 420)
        lay = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel("左侧可用列 → 添加到右侧并拖动排序；必须与训练时列数一致。")
        lay.addWidget(info)

        body = QtWidgets.QHBoxLayout()
        lay.addLayout(body)

        self.list_all = QtWidgets.QListWidget()
        self.list_all.addItems(all_columns)
        self.list_all.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        mid = QtWidgets.QVBoxLayout()
        btn_add = QtWidgets.QPushButton("→ 添加")
        btn_remove = QtWidgets.QPushButton("← 移除")
        btn_up = QtWidgets.QPushButton("上移")
        btn_down = QtWidgets.QPushButton("下移")
        mid.addStretch(1); mid.addWidget(btn_add); mid.addWidget(btn_remove); mid.addSpacing(10)
        mid.addWidget(btn_up); mid.addWidget(btn_down); mid.addStretch(1)

        self.list_sel = QtWidgets.QListWidget()
        self.list_sel.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        body.addWidget(self.list_all, 5)
        body.addLayout(mid, 1)
        body.addWidget(self.list_sel, 5)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        lay.addWidget(btns)

        self.need_k = need_k
        btn_add.clicked.connect(self._add)
        btn_remove.clicked.connect(self._remove)
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down.clicked.connect(lambda: self._move(1))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def _add(self):
        for it in self.list_all.selectedItems():
            self.list_sel.addItem(it.text())

    def _remove(self):
        for it in self.list_sel.selectedItems():
            self.list_sel.takeItem(self.list_sel.row(it))

    def _move(self, d):
        rows = [self.list_sel.row(it) for it in self.list_sel.selectedItems()]
        if not rows: return
        r = rows[0]
        nr = r + d
        if 0 <= nr < self.list_sel.count():
            it = self.list_sel.takeItem(r)
            self.list_sel.insertItem(nr, it)
            self.list_sel.setCurrentRow(nr)

    def selected_columns(self) -> List[str]:
        return [self.list_sel.item(i).text() for i in range(self.list_sel.count())]

    def accept(self):
        sel = self.selected_columns()
        if len(sel) != self.need_k:
            QtWidgets.QMessageBox.warning(self, "列数不一致", f"当前选择 {len(sel)} 列，需要 {self.need_k} 列。")
            return
        super().accept()


class ResultsDashboard(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("结果仪表盘", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(6)

        self.anomaly_bar = QtWidgets.QProgressBar()
        self.anomaly_bar.setRange(0, 100)
        self.anomaly_bar.setFormat("尚无结果")
        layout.addWidget(QtWidgets.QLabel("异常占比"))
        layout.addWidget(self.anomaly_bar)

        self.training_compare = QtWidgets.QLabel("训练/当前比较尚未加载")
        self.training_compare.setWordWrap(True)
        layout.addWidget(self.training_compare)

        self.metrics_label = QtWidgets.QLabel("Precision / Recall / F1 待更新")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)

        self.timeline_label = QtWidgets.QLabel("暂无风险趋势数据")
        self.timeline_label.setAlignment(QtCore.Qt.AlignCenter)
        self.timeline_label.setMinimumHeight(140)
        self.timeline_label.setStyleSheet("QLabel { border:1px solid #E6E9EF; border-radius:6px; background:#FFFFFF; }")
        self.timeline_label.setScaledContents(True)
        layout.addWidget(self.timeline_label)

        self._timeline_path: Optional[str] = None

    def update_metrics(self, analysis: Optional[dict], metadata: Optional[dict]) -> None:
        ratio: Optional[float] = None
        metrics = None
        if isinstance(analysis, dict):
            metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else None
            if metrics:
                ratio = metrics.get("malicious_ratio")
        if ratio is not None:
            self.anomaly_bar.setValue(int(np.clip(ratio, 0.0, 1.0) * 100))
            self.anomaly_bar.setFormat(f"{ratio:.2%}")
        else:
            self.anomaly_bar.reset()
            self.anomaly_bar.setFormat("尚无结果")

        train_ratio = None
        if isinstance(metadata, dict):
            train_ratio = metadata.get("training_anomaly_ratio")
        compare_lines = []
        if train_ratio is not None:
            compare_lines.append(f"训练异常占比：{float(train_ratio):.2%}")
        if ratio is not None:
            compare_lines.append(f"当前检测异常占比：{ratio:.2%}")
        if isinstance(metadata, dict) and metadata.get("timestamp"):
            compare_lines.append(f"模型时间：{metadata.get('timestamp')}")
        self.training_compare.setText("\n".join(compare_lines) if compare_lines else "训练/当前比较尚未加载")

        eval_block = None
        if isinstance(metadata, dict) and isinstance(metadata.get("evaluation"), dict):
            eval_block = metadata.get("evaluation")
        if eval_block is None and metrics and isinstance(metrics.get("model_metrics"), dict):
            eval_block = metrics.get("model_metrics")
        if eval_block:
            precision = float(eval_block.get("precision", 0.0))
            recall = float(eval_block.get("recall", 0.0))
            f1 = float(eval_block.get("f1", 0.0))
            text = f"Precision：{precision:.2%}  Recall：{recall:.2%}  F1：{f1:.2%}"
        else:
            text = "Precision / Recall / F1 待更新"
        if metrics:
            roc_val = metrics.get("roc_auc")
            pr_val = metrics.get("pr_auc")
            if roc_val is not None or pr_val is not None:
                parts = [text]
                if roc_val is not None:
                    parts.append(f"ROC AUC：{float(roc_val):.3f}")
                if pr_val is not None:
                    parts.append(f"PR AUC：{float(pr_val):.3f}")
                text = "\n".join(parts)
        self.metrics_label.setText(text)

        timeline_path = None
        if analysis:
            timeline_path = analysis.get("timeline_plot")
            if not timeline_path and metrics:
                timeline_path = metrics.get("timeline_plot")
        if timeline_path and os.path.exists(timeline_path):
            if timeline_path != self._timeline_path:
                pixmap = QtGui.QPixmap(timeline_path)
                if not pixmap.isNull():
                    self.timeline_label.setPixmap(pixmap)
                    self.timeline_label.setText("")
                    self._timeline_path = timeline_path
        else:
            self.timeline_label.setPixmap(QtGui.QPixmap())
            self.timeline_label.setText("暂无风险趋势数据")
            self._timeline_path = None


class AnomalyDetailDialog(QtWidgets.QDialog):
    annotation_saved = QtCore.pyqtSignal(float)

    def __init__(self, record: Dict[str, object], parent=None):
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑全局配置 (YAML)")
        self.resize(720, 540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.path = project_root() / "config" / "default.yaml"
        self.path.parent.mkdir(parents=True, exist_ok=True)

        layout.addWidget(QtWidgets.QLabel(f"配置文件：{self.path}"))
        self.editor = QtWidgets.QPlainTextEdit()
        layout.addWidget(self.editor, 1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(btns)

        btn_reload = QtWidgets.QPushButton("重新加载")
        btns.addButton(btn_reload, QtWidgets.QDialogButtonBox.ResetRole)

        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        btn_reload.clicked.connect(self._load)

        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except FileNotFoundError:
            text = yaml.safe_dump(load_config() or {}, allow_unicode=True, sort_keys=False)
        self.editor.setPlainText(text)

    def _on_save(self) -> None:
        text = self.editor.toPlainText()
        try:
            yaml.safe_load(text or "{}")
        except yaml.YAMLError as exc:
            QtWidgets.QMessageBox.critical(self, "格式错误", f"YAML 解析失败：{exc}")
            return
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(text)
            if hasattr(load_config, "cache_clear"):
                load_config.cache_clear()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "保存失败", f"无法写入配置：{exc}")
            return
        QtWidgets.QMessageBox.information(self, "已保存", "配置已保存。部分修改可能需重启生效。")
        self.accept()


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

    def run(self) -> None:
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


# =============== 主 UI ===============
class Ui_MainWindow(object):
    # --------- 基本结构 ----------
    def setupUi(self, MainWindow):
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

        # 分页状态
        self._csv_paged_path: Optional[str] = None
        self._csv_total_rows: Optional[int] = None
        self._csv_current_page: int = 1

        # worker
        self.worker: Optional[InfoWorker] = None
        self.preprocess_worker: Optional[PreprocessWorker] = None
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
        if not hasattr(self, "pipeline_checks"):
            return {}
        return {key: checkbox.isChecked() for key, checkbox in self.pipeline_checks.items()}

    def _collect_speed_config(self) -> Dict[str, object]:
        if not hasattr(self, "speed_mode_checkbox"):
            return {}
        ratio = float(self.two_stage_ratio_spin.value()) / 100.0 if hasattr(self, "two_stage_ratio_spin") else 0.03
        return {
            "enabled": bool(self.speed_mode_checkbox.isChecked()),
            "two_stage_refine": bool(self.two_stage_checkbox.isChecked()),
            "refine_ratio": ratio,
        }

    def _update_two_stage_controls(self) -> None:
        if not hasattr(self, "two_stage_checkbox"):
            return
        enabled = bool(self.speed_mode_checkbox.isChecked())
        self.two_stage_checkbox.setEnabled(enabled)
        self.two_stage_ratio_spin.setEnabled(enabled and self.two_stage_checkbox.isChecked())

    def _on_pipeline_option_toggled(self) -> None:
        if getattr(self, "_loading_settings", False):
            return
        if not getattr(self, "_settings_ready", False):
            return
        if not hasattr(self, "_settings"):
            return
        try:
            self._settings.set("pipeline_components", self._collect_pipeline_config())
        except Exception:
            pass

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
        model_layout.addWidget(self.btn_analysis)
        model_layout.addWidget(self.btn_predict)
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

        pipeline_options = [
            ("feature_weighter", "特征加权"),
            ("variance_filter", "低方差过滤"),
            ("scaler", "标准化"),
            ("deep_features", "深度表征 (AutoEncoder)"),
            ("gaussianizer", "分位数正态化"),
            ("rbf_expander", "RBF 特征扩展"),
        ]
        self.pipeline_labels = {key: label for key, label in pipeline_options}
        self.pipeline_group, pipeline_layout = self._create_collapsible_group(
            "Pipeline 组件", spacing=6
        )
        self.pipeline_checks = {}
        for key, label in pipeline_options:
            checkbox = QtWidgets.QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_pipeline_option_toggled)
            self.pipeline_checks[key] = checkbox
            pipeline_layout.addWidget(checkbox)
        pipeline_layout.addStretch(1)
        self.right_layout.addWidget(self.pipeline_group)

        self.advanced_group, ag_layout = self._create_collapsible_group(
            "模型高级设置", QtWidgets.QFormLayout, spacing=10
        )

        self.speed_mode_checkbox = QtWidgets.QCheckBox("启用极速模式（IF 提速）")
        self.speed_mode_checkbox.setChecked(True)
        self.two_stage_checkbox = QtWidgets.QCheckBox("两阶段精排（建议）")
        self.two_stage_checkbox.setChecked(True)
        self.two_stage_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.two_stage_ratio_spin.setRange(0.5, 10.0)
        self.two_stage_ratio_spin.setDecimals(1)
        self.two_stage_ratio_spin.setSingleStep(0.5)
        self.two_stage_ratio_spin.setSuffix("%")
        self.two_stage_ratio_spin.setValue(3.0)
        two_stage_row = QtWidgets.QWidget()
        two_stage_layout = QtWidgets.QHBoxLayout(two_stage_row)
        two_stage_layout.setContentsMargins(0, 0, 0, 0)
        two_stage_layout.setSpacing(6)
        two_stage_layout.addWidget(self.two_stage_checkbox)
        two_stage_layout.addWidget(self.two_stage_ratio_spin)
        two_stage_row.setMinimumHeight(38)

        self.rbf_components_spin = QtWidgets.QSpinBox()
        self.rbf_components_spin.setRange(0, 2048)
        self.rbf_components_spin.setSingleStep(32)
        self.rbf_components_spin.setSpecialValueText("自动")
        self.rbf_components_spin.setValue(384)
        self.rbf_gamma_spin = QtWidgets.QDoubleSpinBox()
        self.rbf_gamma_spin.setRange(0.0, 5.0)
        self.rbf_gamma_spin.setDecimals(4)
        self.rbf_gamma_spin.setSingleStep(0.05)
        self.rbf_gamma_spin.setSpecialValueText("自动 (≈1/√d)")
        self.rbf_gamma_spin.setValue(0.0)
        self.fusion_checkbox = QtWidgets.QCheckBox("启用半监督融合")
        self.fusion_checkbox.setChecked(True)
        self.fusion_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.fusion_alpha_spin.setRange(0.0, 1.0)
        self.fusion_alpha_spin.setDecimals(2)
        self.fusion_alpha_spin.setSingleStep(0.05)
        self.fusion_alpha_spin.setValue(0.50)
        self.fusion_alpha_spin.setToolTip("α 越大越偏向无监督风险分数")
        self.memory_ceiling_combo = QtWidgets.QComboBox()
        self.memory_ceiling_combo.addItems(
            [
                "自动 (物理内存 35%)",
                "512 MB",
                "1 GB",
                "2 GB",
                "4 GB",
                "8 GB",
            ]
        )
        self.memory_ceiling_combo.setMinimumHeight(38)
        self.feature_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.feature_slider.setRange(0, 100)
        self.feature_slider.setSingleStep(5)
        self.feature_slider.setPageStep(10)
        self.feature_slider.setValue(0)
        self.feature_slider.setToolTip("0 表示保留全部特征，其它值表示按重要性保留前 N% 的特征")
        feature_slider_row = QtWidgets.QWidget()
        feature_slider_layout = QtWidgets.QHBoxLayout(feature_slider_row)
        feature_slider_layout.setContentsMargins(0, 0, 0, 0)
        feature_slider_layout.setSpacing(6)
        feature_slider_layout.addWidget(self.feature_slider)
        self.feature_slider_value = QtWidgets.QLabel("全部")
        feature_slider_layout.addWidget(self.feature_slider_value)
        feature_slider_row.setMinimumHeight(38)
        ag_layout.addRow("极速模式：", self.speed_mode_checkbox)
        ag_layout.addRow("两阶段精排：", two_stage_row)
        ag_layout.addRow("RBF 维度：", self.rbf_components_spin)
        ag_layout.addRow("RBF γ：", self.rbf_gamma_spin)
        ag_layout.addRow("半监督融合：", self.fusion_checkbox)
        ag_layout.addRow("融合权重 α：", self.fusion_alpha_spin)
        ag_layout.addRow("特征筛选阈值：", feature_slider_row)
        ag_layout.addRow("内存上限", self.memory_ceiling_combo)
        self.fusion_alpha_spin.setEnabled(self.fusion_checkbox.isChecked())
        self._on_feature_slider_changed(self.feature_slider.value())
        self._update_two_stage_controls()
        for widget in (
            self.rbf_components_spin,
            self.rbf_gamma_spin,
            self.fusion_alpha_spin,
            self.memory_ceiling_combo,
            self.two_stage_ratio_spin,
        ):
            widget.setMinimumHeight(38)
        for field in (
            self.speed_mode_checkbox,
            two_stage_row,
            self.rbf_components_spin,
            self.rbf_gamma_spin,
            self.fusion_checkbox,
            self.fusion_alpha_spin,
            self.memory_ceiling_combo,
            feature_slider_row,
        ):
            label = ag_layout.labelForField(field)
            if isinstance(label, QtWidgets.QLabel):
                label.setMinimumWidth(140)
                label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.right_layout.addWidget(self.advanced_group)

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

    def _open_config_editor_dialog(self) -> None:
        dialog = ConfigEditorDialog(self.right_frame)
        result = dialog.exec_()
        if result != QtWidgets.QDialog.Accepted:
            return

        try:
            if hasattr(load_config, "cache_clear"):
                load_config.cache_clear()
        except Exception:
            pass

        self._update_plugin_summary()
        self._refresh_model_versions()
        self._update_status_message()
        self.display_result("[INFO] 配置已更新，相关目录和插件信息已刷新。")

    def _toggle_online_detection(self) -> None:
        if pd is None:
            QtWidgets.QMessageBox.warning(None, "缺少依赖", "当前环境未安装 pandas，无法执行在线检测。")
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
        self.display_result(f"[错误] 在线检测：{message}")
        QtWidgets.QMessageBox.warning(None, "在线检测错误", message)

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
            df = pd.read_csv(feature_csv, encoding="utf-8")
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
        self.display_result(f"[错误] 在线检测任务失败：{message}")
        QtWidgets.QMessageBox.warning(None, "在线检测任务失败", message)
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
        self.rbf_components_spin.valueChanged.connect(self._on_training_settings_changed)
        self.rbf_gamma_spin.valueChanged.connect(self._on_training_settings_changed)
        self.fusion_checkbox.toggled.connect(lambda checked: self.fusion_alpha_spin.setEnabled(checked))
        self.fusion_checkbox.toggled.connect(self._on_training_settings_changed)
        self.fusion_alpha_spin.valueChanged.connect(self._on_training_settings_changed)
        self.speed_mode_checkbox.toggled.connect(self._on_training_settings_changed)
        self.two_stage_checkbox.toggled.connect(self._on_training_settings_changed)
        self.two_stage_ratio_spin.valueChanged.connect(self._on_training_settings_changed)
        self.speed_mode_checkbox.toggled.connect(self._update_two_stage_controls)
        self.two_stage_checkbox.toggled.connect(self._update_two_stage_controls)
        self.feature_slider.valueChanged.connect(self._on_feature_slider_changed)

        self.output_list.customContextMenuRequested.connect(self._on_output_ctx_menu)
        self.output_list.itemDoubleClicked.connect(self._on_output_double_click)
        self.table_view.doubleClicked.connect(self._on_table_double_click)

    # --------- 路径小工具 ----------
    def _project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    def _default_split_dir(self):
        return str(PATHS["split"])

    def _default_results_dir(self):
        return str(PATHS["results_analysis"])

    def _default_models_dir(self):
        return str(PATHS["models"])

    def _default_csv_info_dir(self):
        return str(PATHS["csv_info"])

    def _default_csv_feature_dir(self):
        return str(PATHS["csv_feature"])

    def _analysis_out_dir(self):
        return str(PATHS["results_analysis"])

    def _prediction_out_dir(self):
        return str(PATHS["results_pred"])

    def _abnormal_out_dir(self):
        return str(PATHS["results_abnormal"])

    def _preprocess_out_dir(self):
        return str(PATHS["csv_preprocess"])

    def _current_memory_budget_bytes(self) -> Optional[int]:
        idx = self.memory_ceiling_combo.currentIndex()
        if idx <= 0:
            return None
        mapping = {
            1: 512,
            2: 1024,
            3: 2048,
            4: 4096,
            5: 8192,
        }
        value_mb = mapping.get(idx)
        if value_mb is None:
            return None
        return int(value_mb) * 1024 * 1024

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
        try:
            names = [n for n in os.listdir(d) if n.lower().endswith((".pcap", ".pcapng"))]
        except Exception:
            names = []
        names.sort()
        return [os.path.join(d, n) for n in names]

    def _on_feature_slider_changed(self, value: int) -> None:
        if value <= 0:
            self.feature_slider_value.setText("全部")
        else:
            self.feature_slider_value.setText(f"{int(value)}%")

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
            df = pd.read_csv(self._csv_paged_path, skiprows=range(1, 1 + skip), nrows=page_size, encoding="utf-8")
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
            df = pd.read_csv(csv_path, encoding="utf-8")
            self.populate_table_from_df(df)
            self.display_result(f"[INFO] 已加载全部 {len(df)} 行。")
            self._csv_paged_path = None; self._csv_total_rows = None; self._csv_current_page = 1
            self.page_info.setText("第 1/1 页")
        finally:
            app.restoreOverrideCursor()

    # --------- 表格渲染 ----------
    def _auto_tag_dataframe(self, df: "pd.DataFrame"):
        if pd is None or df is None:
            return df
        if df.empty:
            return df

        tag_col = "__TAG__"
        df[tag_col] = ""

        def _ensure_mask(mask):
            if isinstance(mask, pd.Series):
                return mask.fillna(False)
            return pd.Series(mask, index=df.index).fillna(False)

        def _append_reason(mask, reason: str):
            mask = _ensure_mask(mask)
            if not mask.any():
                return
            current = df.loc[mask, tag_col].astype(str)
            df.loc[mask, tag_col] = [
                reason if not s or not s.strip() else f"{s};{reason}"
                for s in current
            ]

        def _append_reason_from_values(mask, values: "pd.Series", prefix: str):
            mask = _ensure_mask(mask)
            if not mask.any():
                return
            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=df.index)
            vals = values.loc[mask].astype(str).str.strip().str[:40]
            current = df.loc[mask, tag_col].astype(str)
            df.loc[mask, tag_col] = [
                (f"{prefix}{val}" if not s or not s.strip() else f"{s};{prefix}{val}")
                for s, val in zip(current, vals)
            ]

        if "prediction" in df.columns:
            series = df["prediction"]
            try:
                if pd.api.types.is_numeric_dtype(series):
                    mask_pred = pd.to_numeric(series, errors="coerce").fillna(0) < 0
                else:
                    text = series.astype(str).str.lower()
                    mask_pred = text.isin({"-1", "anomaly", "abnormal", "malicious", "恶意", "异常"}) | text.str.contains(
                        "attack|threat|anomaly|异常|恶意", case=False, na=False
                    )
            except Exception:
                mask_pred = pd.Series(False, index=df.index)
            _append_reason(mask_pred, "模型预测异常")

        if "anomaly_score" in df.columns:
            series = pd.to_numeric(df["anomaly_score"], errors="coerce")
            finite = series.dropna()
            if not finite.empty:
                if (finite <= 0).any() and (finite >= 0).any():
                    threshold = 0
                else:
                    threshold = finite.quantile(0.98) if len(finite) > 10 else finite.max()
                mask_score = series > threshold
                _append_reason(mask_score, "异常得分偏高")

        bool_like_names = {
            "is_anomaly",
            "anomaly",
            "is_attack",
            "is_malicious",
            "malicious",
            "threat",
        }
        for col in df.columns:
            if str(col).lower() in bool_like_names:
                series = df[col].astype(str).str.strip().str.lower()
                mask_bool = series.isin({"1", "true", "yes", "y", "异常", "恶意", "attack", "malicious", "是"})
                _append_reason(mask_bool, f"{col} 指示异常")

        suspicious_column_keywords = ("label", "result", "status", "type", "attack", "threat", "alert", "category", "tag")
        safe_values = {"", "normal", "benign", "none", "ok", "-", "合法", "正常", "无"}
        safe_values = {s.lower() for s in safe_values}
        suspicious_values = [
            "异常", "攻击", "恶意", "malicious", "threat", "suspicious", "可疑", "beacon", "flood",
            "bot", "c2", "command", "shell", "scan", "exploit", "入侵", "泄露", "exfil"
        ]

        for col in df.columns:
            lower = str(col).lower()
            if lower == tag_col.lower():
                continue
            if any(key in lower for key in suspicious_column_keywords):
                try:
                    text_series = df[col].astype(str).str.strip()
                except Exception:
                    continue

                def _is_suspicious(value: str) -> bool:
                    if not value:
                        return False
                    lv = value.lower()
                    if lv in safe_values:
                        return False
                    return any(keyword in lv for keyword in suspicious_values)

                mask = text_series.apply(_is_suspicious)
                _append_reason_from_values(mask, text_series, f"{col}: ")

        return df

    def populate_table_from_df(self, df: "pd.DataFrame"):
        if pd is None:
            raise RuntimeError("pandas required")
        if pd is not None and isinstance(df, pd.DataFrame):
            df = self._auto_tag_dataframe(df)
        self._last_preview_df = df
        show_df = df.head(PREVIEW_LIMIT_FOR_TABLE).copy() if (len(df) > PREVIEW_LIMIT_FOR_TABLE and not self._csv_paged_path) else df
        self.table_view.setUpdatesEnabled(False)
        try:
            m = PandasFrameModel(show_df, self.table_view)
            proxy = QtCore.QSortFilterProxyModel(self.table_view)
            proxy.setSourceModel(m)
            proxy.setFilterKeyColumn(-1)
            self.table_view.setModel(proxy)
            self.table_view.setSortingEnabled(True)
            self.table_view.resizeColumnsToContents()

            def _current_df():
                src = proxy.sourceModel()
                return getattr(src, "_df", None)

            self.table_view.setItemDelegate(RowHighlighter(_current_df, self.table_view))
            self.display_tabs.setCurrentWidget(self.table_widget)
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
                df_full = pd.read_csv(self._last_out_csv, encoding="utf-8")
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

        self.worker = InfoWorker(
            path=path, workers=workers,
            mode=("all" if mode == "auto" and os.path.isdir(path) else ("file" if mode == "auto" and os.path.isfile(path) else mode)),
            batch_size=batch, start_index=start,
            files=file_list if os.path.isdir(path) else None,
            proto_filter=proto, port_whitelist_text=wl, port_blacklist_text=bl,
            fast=True,
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
        if "prediction" in df.columns:
            export_df = df[df["prediction"] == -1].copy()
        elif "anomaly_score" in df.columns:
            export_df = df[df["anomaly_score"] > 0].copy()
        else:
            if not written_any:
                QtWidgets.QMessageBox.information(None, "无异常标记", "没有 prediction 或 anomaly_score 列，无法筛选异常。")
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

        metadata = self._selected_metadata if isinstance(self._selected_metadata, dict) else analysis.get("metadata")
        prediction = self._latest_prediction_summary if isinstance(self._latest_prediction_summary, dict) else None
        try:
            annot_info = annotation_summary()
        except Exception:
            annot_info = {}

        try:
            with PdfPages(file_path) as pdf:
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                y_pos = 0.95
                ax.text(0.5, y_pos, "恶意流量检测自动化报告", ha="center", va="top", fontsize=20, weight="bold")
                y_pos -= 0.06

                def add_line(text_line: str, *, indent: float = 0.0, fontsize: int = 12) -> None:
                    nonlocal fig, ax, y_pos
                    if y_pos < 0.08:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                        fig, ax = plt.subplots(figsize=(8.27, 11.69))
                        ax.axis("off")
                        y_pos = 0.95
                    ax.text(0.05 + indent, y_pos, text_line, fontsize=fontsize, va="top")
                    y_pos -= 0.035

                add_line(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if metadata:
                    if metadata.get("timestamp"):
                        add_line(f"模型训练时间：{metadata['timestamp']}")
                    if metadata.get("contamination") is not None:
                        add_line(f"训练污染率：{float(metadata['contamination']):.2%}")
                    if metadata.get("training_anomaly_ratio") is not None:
                        add_line(f"训练异常占比：{float(metadata['training_anomaly_ratio']):.2%}")
                    if metadata.get("estimated_precision") is not None:
                        add_line(f"估计精度：{float(metadata['estimated_precision']):.2%}")
                if analysis.get("out_dir"):
                    add_line(f"分析输出目录：{analysis.get('out_dir')}")

                summary_text = analysis.get("summary_text")
                if summary_text:
                    add_line("分析摘要：")
                    for line in textwrap.wrap(str(summary_text), width=68):
                        add_line(line, indent=0.02)

                metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else {}
                if metrics:
                    add_line("关键指标：")
                    if metrics.get("malicious_ratio") is not None:
                        add_line(f"当前异常占比：{float(metrics['malicious_ratio']):.2%}", indent=0.02)
                    if metrics.get("anomaly_count") is not None:
                        add_line(f"异常样本数量：{int(metrics['anomaly_count'])}", indent=0.02)
                    if metrics.get("total_count") is not None:
                        add_line(f"总样本数量：{int(metrics['total_count'])}", indent=0.02)
                    drift_block = metrics.get("drift") if isinstance(metrics.get("drift"), dict) else {}
                    if drift_block:
                        add_line("漂移检测：", indent=0.02)
                        for key, label in (
                            ("kl_divergence", "KL 散度"),
                            ("psi", "PSI"),
                            ("p_value", "p-value"),
                        ):
                            if drift_block.get(key) is not None:
                                add_line(f"{label}：{float(drift_block[key]):.4f}", indent=0.04)
                    eval_block = metrics.get("model_metrics") if isinstance(metrics.get("model_metrics"), dict) else {}
                    if eval_block:
                        add_line("模型评估：", indent=0.02)
                        for key in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
                            if eval_block.get(key) is None:
                                continue
                            value = eval_block[key]
                            if key in {"precision", "recall", "f1"}:
                                add_line(f"{key.upper()}：{float(value):.2%}", indent=0.04)
                            else:
                                add_line(f"{key.upper()}：{float(value):.3f}", indent=0.04)

                if annot_info and annot_info.get("total"):
                    add_line(
                        "人工标注累计：{} 条（异常 {}，正常 {}）".format(
                            int(annot_info.get("total", 0)),
                            int(annot_info.get("anomalies", 0)),
                            int(annot_info.get("normals", 0)),
                        ),
                        indent=0.02,
                    )

                if prediction and prediction.get("summary"):
                    add_line("最近一次预测：")
                    for line in prediction.get("summary") or []:
                        add_line(str(line), indent=0.02)

                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

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
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis("off")
                    try:
                        image = plt.imread(abs_path)
                        ax.imshow(image)
                        ax.set_title(os.path.basename(abs_path), fontsize=12)
                    except Exception as exc:
                        ax.text(0.5, 0.5, f"无法加载图像：{os.path.basename(abs_path)}\n{exc}", ha="center", va="center")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
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
            self.dir_fe_worker = DirFeatureWorker(path, out_dir, workers=self.workers_spin.value())
            self.dir_fe_worker.progress.connect(lambda p: self.set_button_progress(self.btn_fe, p))
            self.dir_fe_worker.finished.connect(self._on_fe_dir_finished)
            self.dir_fe_worker.error.connect(self._on_fe_error)
            self.dir_fe_worker.start()
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            csv = os.path.join(out_dir, f"{base}_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            self.display_result(f"[INFO] 单文件特征提取：{path} -> {csv}")
            self.btn_fe.setEnabled(False); self.set_button_progress(self.btn_fe, 1)
            self.fe_worker = FeatureWorker(path, csv)
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
            df = pd.read_csv(csv, nrows=50, encoding="utf-8")
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
                df = pd.read_csv(first, nrows=50, encoding="utf-8")
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
        feature_source = self._ask_feature_source()
        if not feature_source:
            self.display_result("[INFO] 已取消数据预处理。")
            return

        out_dir = self._preprocess_out_dir()
        os.makedirs(out_dir, exist_ok=True)

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

        self.display_result(f"[INFO] 数据预处理：{preview} -> {out_dir}")
        self._set_action_buttons_enabled(False)
        self.btn_vector.setEnabled(False); self.set_button_progress(self.btn_vector, 1)
        self.preprocess_worker = PreprocessWorker(feature_source, out_dir)
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
                df = pd.read_csv(manifest, nrows=50, encoding="utf-8")
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
        comp = self.rbf_components_spin.value()
        gamma = self.rbf_gamma_spin.value()
        fusion_enabled = self.fusion_checkbox.isChecked()
        fusion_alpha = self.fusion_alpha_spin.value()
        feature_ratio = self.feature_slider.value()
        feature_ratio = (float(feature_ratio) / 100.0) if feature_ratio > 0 else None
        pipeline_config = self._collect_pipeline_config()
        memory_budget = self._current_memory_budget_bytes()
        speed_config = self._collect_speed_config()
        train_task = BackgroundTask(
            run_train,
            path,
            res_dir,
            mdl_dir,
            rbf_components=int(comp) if comp > 0 else None,
            rbf_gamma=float(gamma) if gamma > 0 else None,
            enable_supervised_fusion=bool(fusion_enabled),
            fusion_alpha=float(fusion_alpha),
            feature_selection_ratio=feature_ratio,
            pipeline_components=pipeline_config,
            memory_budget_bytes=memory_budget,
            speed_config=speed_config,
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
        if res.get("rbf_components"):
            msg_lines.append(f"RBF 维度={int(res['rbf_components'])}")
        if res.get("expanded_dim"):
            msg_lines.append(f"展开后维度={int(res['expanded_dim'])}")
        gamma_val = res.get("rbf_gamma")
        if gamma_val is not None:
            gamma_line = f"RBF γ={float(gamma_val):.3f}"
            source = res.get("rbf_gamma_source")
            if source == "auto":
                gamma_line += "（自动）"
            elif source == "manual":
                gamma_line += "（手动）"
            msg_lines.append(gamma_line)
        if res.get("rbf_gamma_auto") is not None:
            msg_lines.append(f"自动γ估计≈{float(res['rbf_gamma_auto']):.3f}")
        fusion_enabled = res.get("fusion_enabled")
        if fusion_enabled is not None:
            if fusion_enabled:
                alpha_val = float(res.get("fusion_alpha", 0.5))
                fusion_source = res.get("fusion_source")
                if fusion_source:
                    readable = {
                        "supervised_model": "监督模型",
                        "calibration": "校准",
                        "supervised_proba": "监督概率",
                    }.get(fusion_source, str(fusion_source))
                    source_txt = f", 来源={readable}"
                else:
                    source_txt = ""
                msg_lines.append(f"半监督融合=开启(α={alpha_val:.2f}{source_txt})")
            else:
                msg_lines.append("半监督融合=关闭")
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
        weight_info = res.get("feature_weighting") or {}
        if isinstance(weight_info, dict) and weight_info.get("ratio"):
            ratio_val = float(weight_info.get("ratio", 0.0))
            selected = weight_info.get("selected_features") or []
            total_feats = len(res.get("feature_columns") or [])
            msg_lines.append(
                f"特征筛选≈{ratio_val * 100:.0f}% (保留 {len(selected)}/{total_feats})"
            )
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
        pipeline_cfg = res.get("pipeline_components") or {}
        if isinstance(pipeline_cfg, dict) and pipeline_cfg:
            enabled_labels = [
                self.pipeline_labels.get(key, key)
                for key, flag in pipeline_cfg.items()
                if flag and key in self.pipeline_labels
            ]
            disabled_labels = [
                self.pipeline_labels.get(key, key)
                for key, flag in pipeline_cfg.items()
                if not flag and key in self.pipeline_labels
            ]
            if enabled_labels:
                msg_lines.append("启用组件: " + ", ".join(enabled_labels))
            if disabled_labels:
                msg_lines.append("停用组件: " + ", ".join(disabled_labels))
        speed_cfg = res.get("speed_config") or {}
        if isinstance(speed_cfg, dict) and speed_cfg:
            status = "开启" if speed_cfg.get("enabled", True) else "关闭"
            ratio_val = speed_cfg.get("refine_ratio")
            if ratio_val is not None:
                try:
                    ratio_pct = float(ratio_val) * 100.0
                except Exception:
                    ratio_pct = 0.0
                msg_lines.append(f"极速模式={status}（精排Top≈{ratio_pct:.1f}%）")
            else:
                msg_lines.append(f"极速模式={status}")
        refine_report = res.get("refinement_report") or {}
        if isinstance(refine_report, dict) and refine_report.get("refined"):
            subset = refine_report.get("subset_size")
            ratio_val = refine_report.get("ratio")
            overlap_map = refine_report.get("topk_overlap") or {}
            overlap100 = overlap_map.get("100")
            overlap_alert = refine_report.get("topk_overlap_alert") or {}
            try:
                ratio_pct = float(ratio_val) * 100.0 if ratio_val is not None else None
            except Exception:
                ratio_pct = None
            line = "两阶段精排: "
            if ratio_pct is not None:
                line += f"Top≈{ratio_pct:.1f}%"
            if subset is not None:
                line += f" ({int(subset)} 条)"
            if overlap100 is not None:
                try:
                    line += f" @100重叠≈{float(overlap100) * 100:.1f}%"
                except Exception:
                    pass
            failing_entries: List[str] = []
            if isinstance(overlap_alert, dict):
                failing = overlap_alert.get("failing") or {}
                threshold = overlap_alert.get("threshold")
                for key, value in (failing.items() if isinstance(failing, dict) else []):
                    try:
                        failing_entries.append(f"@{key}≈{float(value) * 100:.1f}%")
                    except Exception:
                        continue
                if failing_entries:
                    if isinstance(threshold, (int, float)):
                        try:
                            line += f" ⚠️重叠<{float(threshold) * 100:.0f}%: "
                        except Exception:
                            line += " ⚠️重叠不足: "
                    else:
                        line += " ⚠️重叠不足: "
                    line += ", ".join(failing_entries)
            spearman = refine_report.get("spearman")
            if spearman is not None:
                try:
                    line += f" ρ≈{float(spearman):.3f}"
                except Exception:
                    pass
            msg_lines.append(line)
        compute_device = res.get("compute_device") or res.get("deep_features", {}).get("device")
        backend_name = None
        deep_info = res.get("deep_features") or {}
        if isinstance(deep_info, dict):
            backend_name = deep_info.get("backend") or res.get("deep_backend")
            loss_val = deep_info.get("training_loss")
            if loss_val is not None:
                msg_lines.append(f"自编码器训练误差≈{float(loss_val):.4f}")
        if compute_device or backend_name:
            msg_lines.append(
                f"深度特征运行于 {backend_name or 'numpy'} @ {compute_device or 'CPU'}"
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
                    df = pd.read_csv(top20_csv, encoding="utf-8")
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

        return {
            "output_csv": out_csv,
            "dataframe": out_df,
            "summary": summary_lines,
            "messages": messages,
            "metadata": metadata,
            "malicious": malicious,
            "total": total,
            "ratio": ratio,
        }


    def _on_predict(self):
        if pd is None:
            QtWidgets.QMessageBox.warning(None, "缺少依赖", "当前环境未安装 pandas，无法执行预测。")
            return

        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择特征CSV", self._default_csv_feature_dir(), "CSV (*.csv)"
        )
        if not csv_path:
            return

        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(None, "读取失败", f"无法读取 CSV：{exc}")
            return

        if df.empty:
            QtWidgets.QMessageBox.information(None, "空数据", "该 CSV 没有数据行。")
            return

        self._remember_path(csv_path)

        source_name = os.path.splitext(os.path.basename(csv_path))[0] or os.path.basename(csv_path)
        metadata_override = self._selected_metadata if isinstance(self._selected_metadata, dict) else None

        try:
            result = self._predict_dataframe(
                df,
                source_name=source_name,
                metadata_override=metadata_override,
                silent=True,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(None, "预测失败", str(exc))
            return

        messages = result.get("messages") or []
        if messages:
            log_text = "\n".join(f"[INFO] {line}" for line in messages)
            self.display_result(log_text)

        output_csv = result.get("output_csv")
        if output_csv and os.path.exists(output_csv):
            self._add_output(output_csv)
            self._last_out_csv = output_csv
            self._open_csv_paged(output_csv)

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
        snapshot["dataframe"] = dataframe
        snapshot["source_csv"] = csv_path
        if metadata and "metadata" not in snapshot:
            snapshot["metadata"] = metadata
        self._latest_prediction_summary = snapshot

        QtWidgets.QMessageBox.information(
            None,
            "预测完成",
            "\n".join(result.get("summary") or ["模型预测已完成并写出结果。"]),
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
        if getattr(self, "_loading_settings", False):
            return
        if not getattr(self, "_settings_ready", False):
            return
        if not hasattr(self, "_settings"):
            return
        self.fusion_alpha_spin.setEnabled(self.fusion_checkbox.isChecked())
        self._update_two_stage_controls()
        try:
            self._settings.set("rbf_components", int(self.rbf_components_spin.value()))
            self._settings.set("rbf_gamma", float(self.rbf_gamma_spin.value()))
            self._settings.set("fusion_enabled", bool(self.fusion_checkbox.isChecked()))
            self._settings.set("fusion_alpha", float(self.fusion_alpha_spin.value()))
            self._settings.set("memory_ceiling_idx", int(self.memory_ceiling_combo.currentIndex()))
            self._settings.set("speed_enabled", bool(self.speed_mode_checkbox.isChecked()))
            self._settings.set("two_stage_refine", bool(self.two_stage_checkbox.isChecked()))
            self._settings.set("two_stage_ratio", float(self.two_stage_ratio_spin.value()))
        except Exception:
            pass

    def _apply_saved_preferences(self):
        if not hasattr(self, "_settings"):
            self._settings_ready = True
            return
        self._loading_settings = True
        try:
            last_path = self._settings.get("last_input_path")
            if isinstance(last_path, str) and last_path:
                self.file_edit.setText(last_path)
            saved_components = self._settings.get("rbf_components", 0) or 0
            saved_gamma = self._settings.get("rbf_gamma", 0.0) or 0.0
            try:
                self.rbf_components_spin.setValue(int(saved_components))
            except Exception:
                self.rbf_components_spin.setValue(0)
            try:
                self.rbf_gamma_spin.setValue(float(saved_gamma))
            except Exception:
                self.rbf_gamma_spin.setValue(0.0)
            fusion_enabled = self._settings.get("fusion_enabled", True)
            self.fusion_checkbox.setChecked(bool(fusion_enabled))
            saved_alpha = self._settings.get("fusion_alpha", 0.5)
            try:
                self.fusion_alpha_spin.setValue(float(saved_alpha))
            except Exception:
                self.fusion_alpha_spin.setValue(0.5)
            speed_enabled = self._settings.get("speed_enabled", True)
            self.speed_mode_checkbox.setChecked(bool(speed_enabled))
            two_stage_enabled = self._settings.get("two_stage_refine", True)
            self.two_stage_checkbox.setChecked(bool(two_stage_enabled))
            saved_ratio = self._settings.get("two_stage_ratio", 3.0)
            try:
                self.two_stage_ratio_spin.setValue(float(saved_ratio))
            except Exception:
                self.two_stage_ratio_spin.setValue(3.0)
            saved_memory_idx = self._settings.get("memory_ceiling_idx", 0) or 0
            try:
                idx = int(saved_memory_idx)
                idx = max(0, min(self.memory_ceiling_combo.count() - 1, idx))
                self.memory_ceiling_combo.setCurrentIndex(idx)
            except Exception:
                self.memory_ceiling_combo.setCurrentIndex(0)
            self.fusion_alpha_spin.setEnabled(self.fusion_checkbox.isChecked())
            saved_pipeline = self._settings.get("pipeline_components", {})
            if isinstance(saved_pipeline, dict) and hasattr(self, "pipeline_checks"):
                for key, checkbox in self.pipeline_checks.items():
                    if key in saved_pipeline:
                        try:
                            checkbox.setChecked(bool(saved_pipeline[key]))
                        except Exception:
                            pass
        finally:
            self._loading_settings = False
            self._settings_ready = True
            self._update_two_stage_controls()

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