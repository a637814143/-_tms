# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os, platform, subprocess, math, shutil, json
from pathlib import Path
import numpy as np
from typing import List, Optional, Set
from datetime import datetime

# ---- 业务函数（保持导入路径）----
from src.functions.info import get_pcap_features as info
from src.functions.feature_extractor import (
    extract_features as fe_single,
    extract_features_dir as fe_dir,
)
from src.functions.unsupervised_train import train_unsupervised_on_split as run_train
from src.functions.analyze_results import analyze_results as run_analysis
from src.functions.preprocess import preprocess_feature_dir as preprocess_dir

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
        proj_root = Path(__file__).resolve().parents[2]
        data_dir = proj_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir.resolve()
    except Exception:
        fallback = Path.home() / "maldet_data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


APP_STYLE = """
QWidget { background-color: #F7F8FA; font-family: "Microsoft YaHei", "PingFang SC","SF Pro Text"; color:#1d1d1f; font-size:13px; }
QTextEdit, QTableView { background-color: #FFFFFF; border:1px solid #E3E6EB; border-radius:8px; }
QLineEdit, QSpinBox, QComboBox { background:#FFFFFF; border:1px solid #D7DBE2; border-radius:8px; padding:6px; }
QPushButton { background:#EEF1F6; border:0; border-radius:10px; padding:8px 12px; min-height:30px;}
QPushButton:hover { background:#E2E8F0; } QPushButton:pressed { background:#D9E0EA; }
QHeaderView::section { background:#F3F5F9; padding:6px; border:1px solid #E6E9EF; }
QGroupBox { border:1px solid #E6E9EF; border-radius:10px; margin-top:10px; }
QGroupBox::title { subcontrol-origin: margin; left:12px; padding:0 6px; background:transparent; }
"""

# 仅表格预览上限（全部数据都会落盘）
PREVIEW_LIMIT_FOR_TABLE = 50

DATA_BASE = _resolve_data_base()
PATHS = {
    "split": DATA_BASE / "split",
    "csv_info": DATA_BASE / "CSV" / "info",
    "csv_feature": DATA_BASE / "CSV" / "feature",
    "csv_preprocess": DATA_BASE / "CSV" / "DP",
    "models": DATA_BASE / "models",
    "results_analysis": DATA_BASE / "results" / "analysis",
    "results_pred": DATA_BASE / "results" / "modelprediction",
    "results_abnormal": DATA_BASE / "results" / "abnormal",
}
for _path in PATHS.values():
    _path.mkdir(parents=True, exist_ok=True)

LOGS_DIR = Path(os.getenv("MALDET_LOG_DIR", DATA_BASE / "logs")).expanduser().resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = DATA_BASE / "settings.json"

MODEL_SCHEMA_VERSION = "2025.10"


def _align_input_features(df: "pd.DataFrame", metadata: dict) -> tuple["pd.DataFrame", dict]:
    info: dict[str, object] = {}
    if not isinstance(metadata, dict):
        raise ValueError("模型缺少有效的元数据。")

    schema_version = metadata.get("schema_version")
    if schema_version is None:
        raise ValueError("模型元数据缺少 schema_version 字段，请重新训练模型。")
    info["schema_version"] = schema_version

    feature_order = metadata.get("feature_order")
    preprocessor_meta = metadata.get("preprocessor") if isinstance(metadata.get("preprocessor"), dict) else None
    if not feature_order and preprocessor_meta:
        feature_order = preprocessor_meta.get("feature_order") or preprocessor_meta.get("input_columns")
    if not feature_order:
        feature_order = metadata.get("feature_columns")
    if not feature_order:
        raise ValueError("模型元数据缺少 feature_order 描述，无法校验列。")

    feature_order = list(feature_order)
    if not feature_order:
        raise ValueError("模型元数据的特征列为空。")

    default_fill = 0.0
    fill_values = {}
    if isinstance(metadata.get("fill_values"), dict):
        fill_values.update(metadata["fill_values"])
    if isinstance(preprocessor_meta, dict):
        default_fill = float(preprocessor_meta.get("fill_value", default_fill))
        fill_values.update(preprocessor_meta.get("fill_values") or {})
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

    extra_columns = [col for col in working.columns if col not in feature_order]
    info["missing_filled"] = missing_columns
    info["extra_columns"] = extra_columns

    aligned = working.loc[:, feature_order].copy()
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


class TrainWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(
        self,
        input_path,
        results_dir,
        models_dir,
        rbf_components=None,
        rbf_gamma=None,
        fusion_enabled=True,
        fusion_alpha=0.5,
    ):
        super().__init__()
        self.input_path = input_path
        self.results_dir = results_dir
        self.models_dir = models_dir
        self.rbf_components = rbf_components
        self.rbf_gamma = rbf_gamma
        self.fusion_enabled = fusion_enabled
        self.fusion_alpha = fusion_alpha

    def run(self):
        try:
            res = run_train(
                self.input_path,
                self.results_dir,
                self.models_dir,
                progress_cb=self.progress.emit,
                rbf_components=self.rbf_components,
                rbf_gamma=self.rbf_gamma,
                enable_supervised_fusion=self.fusion_enabled,
                fusion_alpha=float(self.fusion_alpha),
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class AnalysisWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, results_csv, out_dir, metadata_path=None):
        super().__init__()
        self.results_csv = results_csv
        self.out_dir = out_dir
        self.metadata_path = metadata_path

    def run(self):
        try:
            result = run_analysis(
                self.results_csv,
                self.out_dir,
                metadata_path=self.metadata_path,
                progress_cb=self.progress.emit,
            )
            if not isinstance(result, dict):
                result = {"out_dir": self.out_dir}
            elif "out_dir" not in result:
                result["out_dir"] = self.out_dir
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

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


# =============== 主 UI ===============
class Ui_MainWindow(object):
    # --------- 基本结构 ----------
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1100, 680)
        MainWindow.setStyleSheet(APP_STYLE)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.centralwidget)
        self.main_layout.addWidget(self.splitter)

        # 左侧滚动区
        self.left_scroll = QtWidgets.QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_container = QtWidgets.QWidget()
        self.left_scroll.setWidget(self.left_container)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_container)
        self.left_layout.setContentsMargins(4, 4, 4, 4)
        self.left_layout.setSpacing(8)

        self._build_path_bar()
        self._build_param_panel()
        self._build_center_tabs()
        self._build_paging_toolbar()
        self._build_output_list()
        self._build_footer()
        self._update_status_message("@2025 恶意流量检测系统")

        self.splitter.addWidget(self.left_scroll)

        # 右侧按钮列
        self._build_right_panel()
        self.splitter.addWidget(self.right_frame)
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)

        MainWindow.setCentralWidget(self.centralwidget)
        self._bind_signals()

        # 状态缓存
        self._last_preview_df: Optional["pd.DataFrame"] = None
        self._last_out_csv: Optional[str] = None
        self._analysis_summary: Optional[dict] = None

        # 分页状态
        self._csv_paged_path: Optional[str] = None
        self._csv_total_rows: Optional[int] = None
        self._csv_current_page: int = 1

        # worker
        self.worker: Optional[InfoWorker] = None
        self.preprocess_worker: Optional[PreprocessWorker] = None

        # 用户偏好
        self._settings = AppSettings(SETTINGS_PATH)
        self._loading_settings = False
        self._apply_saved_preferences()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # --------- UI 子构建 ----------
    def _build_path_bar(self):
        self.file_bar = QtWidgets.QWidget(self.left_container)
        fb = QtWidgets.QHBoxLayout(self.file_bar)
        fb.setContentsMargins(8, 0, 8, 0)
        fb.setSpacing(8)
        self.file_label = QtWidgets.QLabel("选择流量文件或目录 (.pcap/.pcapng 或 split 目录):")
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("请选择文件或目录路径")
        self.btn_pick_file = QtWidgets.QPushButton("选文件")
        self.btn_pick_dir = QtWidgets.QPushButton("选目录")
        self.btn_browse = QtWidgets.QPushButton("浏览(文件或目录)")
        fb.addWidget(self.file_label, 2)
        fb.addWidget(self.file_edit, 8)
        fb.addWidget(self.btn_pick_file, 1)
        fb.addWidget(self.btn_pick_dir, 1)
        fb.addWidget(self.btn_browse, 2)
        self.left_layout.addWidget(self.file_bar)

    def _build_param_panel(self):
        self.param_group = QtWidgets.QGroupBox("查看流量信息参数")
        pg = QtWidgets.QGridLayout(self.param_group)
        pg.setContentsMargins(10, 6, 10, 6)
        pg.setHorizontalSpacing(12)
        pg.setVerticalSpacing(6)

        self.mode_label = QtWidgets.QLabel("查看模式：")
        self.mode_combo = QtWidgets.QComboBox()
        self._mode_map = {
            "自动(文件=单文件/目录=全部)": "auto",
            "单文件": "file",
            "整个目录(从上到下)": "all",
            "分批(按文件名排序)": "batch"
        }
        self.mode_combo.addItems(list(self._mode_map.keys()))

        self.batch_label = QtWidgets.QLabel("batch_size：")
        self.batch_spin = QtWidgets.QSpinBox()
        self.batch_spin.setRange(1, 99999)
        self.batch_spin.setValue(10)

        self.start_label = QtWidgets.QLabel("start_index：")
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, 10**9)
        self.start_spin.setSingleStep(10)
        self.start_spin.setValue(0)

        self.workers_label = QtWidgets.QLabel("并发数：")
        self.workers_spin = QtWidgets.QSpinBox()
        self.workers_spin.setRange(1, 32)
        self.workers_spin.setValue(8)

        self.proto_label = QtWidgets.QLabel("协议：")
        self.proto_combo = QtWidgets.QComboBox()
        self.proto_combo.addItems(["TCP+UDP", "仅TCP", "仅UDP"])

        self.whitelist_label = QtWidgets.QLabel("端口白名单(任一命中保留)：")
        self.whitelist_edit = QtWidgets.QLineEdit()
        self.whitelist_edit.setPlaceholderText("例: 80,443,53")

        self.blacklist_label = QtWidgets.QLabel("端口黑名单(任一命中排除)：")
        self.blacklist_edit = QtWidgets.QLineEdit()
        self.blacklist_edit.setPlaceholderText("例: 135,137,138,139")

        self.fast_check = QtWidgets.QCheckBox("极速模式（UI开关保留）")
        self.fast_check.setChecked(True)

        self.btn_prev = QtWidgets.QPushButton("上一批")
        self.btn_next = QtWidgets.QPushButton("下一批")

        r = 0
        pg.addWidget(self.mode_label, r, 0); pg.addWidget(self.mode_combo, r, 1, 1, 5); r += 1
        pg.addWidget(self.batch_label, r, 0); pg.addWidget(self.batch_spin, r, 1)
        pg.addWidget(self.start_label, r, 2); pg.addWidget(self.start_spin, r, 3)
        pg.addWidget(self.workers_label, r, 4); pg.addWidget(self.workers_spin, r, 5); r += 1
        pg.addWidget(self.proto_label, r, 0); pg.addWidget(self.proto_combo, r, 1)
        pg.addWidget(self.whitelist_label, r, 2); pg.addWidget(self.whitelist_edit, r, 3, 1, 3); r += 1
        pg.addWidget(self.blacklist_label, r, 0); pg.addWidget(self.blacklist_edit, r, 1, 1, 4)
        pg.addWidget(self.fast_check, r, 5); r += 1
        pg.addWidget(self.btn_prev, r, 0, 1, 3); pg.addWidget(self.btn_next, r, 3, 1, 3); r += 1

        self.left_layout.addWidget(self.param_group)
        self.mode_combo.currentIndexChanged.connect(self._update_batch_controls)
        self._update_batch_controls()

    def _build_center_tabs(self):
        self.display_tabs = QtWidgets.QTabWidget(self.left_container)

        self.results_widget = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(self.results_widget)
        rl.setContentsMargins(6, 6, 6, 6)
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        rl.addWidget(self.results_text)
        self.display_tabs.addTab(self.results_widget, "结果（文本）")

        self.table_widget = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(self.table_widget)
        tl.setContentsMargins(6, 6, 6, 6)
        self.table_view = QtWidgets.QTableView()
        self.table_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.verticalHeader().setDefaultSectionSize(22)
        self.table_widget.setMinimumHeight(22 * 16 + 40)
        tl.addWidget(self.table_view)
        self.display_tabs.addTab(self.table_widget, "流量表格")

        self.left_layout.addWidget(self.display_tabs, stretch=3)

    def _build_paging_toolbar(self):
        bar = QtWidgets.QWidget(self.left_container)
        hb = QtWidgets.QHBoxLayout(bar)
        hb.setContentsMargins(4, 0, 4, 0); hb.setSpacing(8)

        self.btn_page_prev = QtWidgets.QPushButton("上一页")
        self.btn_page_next = QtWidgets.QPushButton("下一页")
        self.page_info = QtWidgets.QLabel("第 0/0 页")
        self.page_size_label = QtWidgets.QLabel("每页行数：")
        self.page_size_spin = QtWidgets.QSpinBox()
        self.page_size_spin.setRange(20, 200000)
        self.page_size_spin.setSingleStep(10)
        self.page_size_spin.setValue(50)
        self.btn_show_all = QtWidgets.QPushButton("显示全部（可能较慢）")

        hb.addWidget(self.btn_page_prev); hb.addWidget(self.btn_page_next)
        hb.addSpacing(8); hb.addWidget(self.page_info); hb.addStretch(1)
        hb.addWidget(self.page_size_label); hb.addWidget(self.page_size_spin)
        hb.addSpacing(8); hb.addWidget(self.btn_show_all)

        self.left_layout.addWidget(bar)

    def _build_output_list(self):
        self.out_group = QtWidgets.QGroupBox("输出文件（双击打开所在目录，右键复制路径）")
        og = QtWidgets.QVBoxLayout(self.out_group)
        og.setContentsMargins(8, 6, 8, 8)
        self.output_list = QtWidgets.QListWidget()
        self.output_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        og.addWidget(self.output_list)
        self.left_layout.addWidget(self.out_group, stretch=1)

    def _build_footer(self):
        self.bottom_bar = QtWidgets.QWidget(self.left_container)
        self.bottom_bar.setFixedHeight(22)
        bb = QtWidgets.QHBoxLayout(self.bottom_bar)
        bb.setContentsMargins(6, 0, 6, 0); bb.addStretch()
        self.bottom_label = QtWidgets.QLabel()
        self.bottom_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        bb.addWidget(self.bottom_label)
        self.left_layout.addWidget(self.bottom_bar)

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

    def _update_status_message(self, message: Optional[str] = None) -> None:
        base = f"数据目录: {DATA_BASE}"
        if message:
            self.bottom_label.setText(f"{message} | {base}")
        else:
            self.bottom_label.setText(base)

    def _build_right_panel(self):
        self.right_frame = QtWidgets.QFrame(self.centralwidget)
        self.right_layout = QtWidgets.QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(6, 6, 6, 6)
        self.right_layout.setSpacing(10)

        self.advanced_group = QtWidgets.QGroupBox("高级设置")
        ag_layout = QtWidgets.QFormLayout(self.advanced_group)
        ag_layout.setContentsMargins(10, 8, 10, 8)
        self.rbf_components_spin = QtWidgets.QSpinBox()
        self.rbf_components_spin.setRange(0, 5000)
        self.rbf_components_spin.setSingleStep(50)
        self.rbf_components_spin.setSpecialValueText("自动")
        self.rbf_components_spin.setValue(0)
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
        ag_layout.addRow("RBF 维度：", self.rbf_components_spin)
        ag_layout.addRow("RBF γ：", self.rbf_gamma_spin)
        ag_layout.addRow("半监督融合：", self.fusion_checkbox)
        ag_layout.addRow("融合权重 α：", self.fusion_alpha_spin)
        self.fusion_alpha_spin.setEnabled(self.fusion_checkbox.isChecked())
        self.right_layout.addWidget(self.advanced_group)

        self.btn_view = QtWidgets.QPushButton("查看流量信息")
        self.btn_fe = QtWidgets.QPushButton("提取特征")
        self.btn_vector = QtWidgets.QPushButton("数据预处理")
        self.btn_train = QtWidgets.QPushButton("训练模型")
        self.btn_analysis = QtWidgets.QPushButton("运行分析")
        self.btn_predict = QtWidgets.QPushButton("加载模型预测")
        self.btn_export = QtWidgets.QPushButton("导出结果（异常）")
        self.btn_clear = QtWidgets.QPushButton("清空显示")
        self.btn_open_results = QtWidgets.QPushButton("打开结果目录")
        self.btn_view_logs = QtWidgets.QPushButton("查看日志")

        for b in [
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
        ]:
            self.right_layout.addWidget(b); self.right_layout.addSpacing(2)

        self.right_layout.addStretch(1)
        self.right_frame.setMinimumWidth(200)

    def _bind_signals(self):
        self.btn_pick_file.clicked.connect(self._choose_file)
        self.btn_pick_dir.clicked.connect(self._choose_dir)
        self.btn_browse.clicked.connect(self._browse_compat)

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

        self.btn_page_prev.clicked.connect(self._on_page_prev)
        self.btn_page_next.clicked.connect(self._on_page_next)
        self.page_size_spin.valueChanged.connect(self._on_page_size_changed)
        self.btn_show_all.clicked.connect(self._show_full_preview)
        self.rbf_components_spin.valueChanged.connect(self._on_rbf_settings_changed)
        self.rbf_gamma_spin.valueChanged.connect(self._on_rbf_settings_changed)
        self.fusion_checkbox.toggled.connect(lambda checked: self.fusion_alpha_spin.setEnabled(checked))
        self.fusion_checkbox.toggled.connect(self._on_rbf_settings_changed)
        self.fusion_alpha_spin.valueChanged.connect(self._on_rbf_settings_changed)

        self.output_list.customContextMenuRequested.connect(self._on_output_ctx_menu)
        self.output_list.itemDoubleClicked.connect(self._on_output_double_click)

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

    def _latest_pipeline_bundle(self):
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
        if pd is None: raise RuntimeError("pandas required")
        if pd is not None and isinstance(df, pd.DataFrame):
            df = self._auto_tag_dataframe(df)
        self._last_preview_df = df
        show_df = df.head(PREVIEW_LIMIT_FOR_TABLE).copy() if (len(df) > PREVIEW_LIMIT_FOR_TABLE and not self._csv_paged_path) else df
        self.table_view.setUpdatesEnabled(False)
        try:
            m = PandasFrameModel(show_df, self.table_view)
            proxy = QtCore.QSortFilterProxyModel(self.table_view)
            proxy.setSourceModel(m); proxy.setFilterKeyColumn(-1)
            self.table_view.setModel(proxy)
            self.table_view.setSortingEnabled(True)
            self.table_view.resizeColumnsToContents()
            def _current_df():
                src = proxy.sourceModel(); return getattr(src, "_df", None)
            self.table_view.setItemDelegate(RowHighlighter(_current_df, self.table_view))
            self.display_tabs.setCurrentWidget(self.table_widget)
        finally:
            self.table_view.setUpdatesEnabled(True)
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
        self.train_worker = TrainWorker(
            path,
            res_dir,
            mdl_dir,
            rbf_components=int(comp) if comp > 0 else None,
            rbf_gamma=float(gamma) if gamma > 0 else None,
            fusion_enabled=bool(fusion_enabled),
            fusion_alpha=float(fusion_alpha),
        )
        self.train_worker.progress.connect(lambda p: self.set_button_progress(self.btn_train, p))
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.start()

    def _on_train_finished(self, res):
        self.btn_train.setEnabled(True)
        self.set_button_progress(self.btn_train, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_train))
        self._set_action_buttons_enabled(True)
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
        _, meta_path = self._latest_pipeline_bundle()
        self.analysis_worker = AnalysisWorker(csv, out_dir, metadata_path=meta_path)
        self.analysis_worker.progress.connect(lambda p: self.set_button_progress(self.btn_analysis, p))
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.start()

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
        else:
            out_dir = str(result) if result is not None else None

        if not out_dir:
            out_dir = self._analysis_out_dir()

        base_msg = f"[INFO] 分析完成，图表保存在 {out_dir}"
        if summary_text:
            self.display_result(f"{base_msg}\n{summary_text}")
        else:
            self.display_result(base_msg)

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

    def _on_analysis_error(self, msg):
        self.btn_analysis.setEnabled(True); self.reset_button_progress(self.btn_analysis)
        self._set_action_buttons_enabled(True)
        QtWidgets.QMessageBox.critical(None, "分析失败", msg)
        self.display_result(f"[错误] 分析失败: {msg}")
        self._analysis_summary = None

    # --------- 模型预测（支持 Pipeline / 模型+scaler / 仅模型） ----------
    def _on_predict(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择特征CSV", self._default_csv_feature_dir(), "CSV (*.csv)"
        )
        if not csv_path:
            return
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "读取失败", f"无法读取 CSV：{e}")
            return
        if df.empty:
            QtWidgets.QMessageBox.information(None, "空数据", "该 CSV 没有数据行。")
            return

        self._remember_path(csv_path)

        from joblib import load as joblib_load

        pipeline_path, meta_path = self._latest_pipeline_bundle()
        if not pipeline_path or not os.path.exists(pipeline_path):
            QtWidgets.QMessageBox.warning(None, "缺少模型", "未找到最新的管线模型，请先完成一次训练。")
            return

        try:
            pipeline = joblib_load(pipeline_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "加载失败", f"无法加载模型：{e}")
            return

        metadata = {}
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    metadata = json.load(fh)
            except Exception:
                metadata = {}

        threshold_breakdown_meta = metadata.get("threshold_breakdown") if isinstance(metadata, dict) else None
        threshold = None
        if isinstance(threshold_breakdown_meta, dict) and threshold_breakdown_meta.get("adaptive") is not None:
            threshold = threshold_breakdown_meta.get("adaptive")
        elif isinstance(metadata, dict):
            threshold = metadata.get("threshold")
        score_std = metadata.get("score_std") if isinstance(metadata, dict) else None
        vote_mean_meta = metadata.get("vote_mean") if isinstance(metadata, dict) else None
        vote_threshold_meta = metadata.get("vote_threshold") if isinstance(metadata, dict) else None

        preprocessor_meta = metadata.get("preprocessor") if isinstance(metadata, dict) else None

        try:
            feature_df_raw, align_info = _align_input_features(df, metadata)
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(None, "特征校验失败", str(exc))
            return

        missing_cols = align_info.get("missing_filled") or []
        extra_cols = align_info.get("extra_columns") or []
        schema_version = align_info.get("schema_version")
        if missing_cols:
            msg = "以下列在输入数据中缺失，已按训练时填充值自动补齐：\n" + ", ".join(missing_cols)
            QtWidgets.QMessageBox.information(None, "已自动补齐特征列", msg)
            self.display_result(f"[WARN] 自动补齐缺失列: {', '.join(missing_cols)}")
        if extra_cols:
            self.display_result(f"[INFO] 预测时忽略未使用列: {', '.join(extra_cols[:10])}{' ...' if len(extra_cols) > 10 else ''}")
        if schema_version and schema_version != MODEL_SCHEMA_VERSION:
            self.display_result(
                f"[WARN] 模型 schema_version={schema_version} 与当前期望 {MODEL_SCHEMA_VERSION} 不一致，请确认兼容性。"
            )

        models_dir = self._default_models_dir()
        preproc_candidates = []
        if isinstance(metadata, dict):
            if metadata.get("preprocessor_latest"):
                preproc_candidates.append(metadata.get("preprocessor_latest"))
            if metadata.get("preprocessor_path"):
                preproc_candidates.append(metadata.get("preprocessor_path"))
        preproc_candidates.append(os.path.join(models_dir, "latest_preprocessor.joblib"))

        loaded_preprocessor = None
        last_preproc_error = None
        for path in preproc_candidates:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            try:
                loaded_preprocessor = joblib_load(path)
                break
            except Exception as exc:
                last_preproc_error = exc
                continue

        if loaded_preprocessor is None:
            loaded_preprocessor = pipeline.named_steps.get("preprocessor") if hasattr(pipeline, "named_steps") else None

        if loaded_preprocessor is None:
            detail = f"（最近一次错误：{last_preproc_error}）" if last_preproc_error else ""
            QtWidgets.QMessageBox.critical(None, "模型不完整", f"缺少可用的特征预处理器，请重新训练。{detail}")
            return

        try:
            feature_df_aligned = loaded_preprocessor.transform(feature_df_raw)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "转换失败", f"特征预处理失败：{e}")
            return

        named_steps = getattr(pipeline, "named_steps", {}) if hasattr(pipeline, "named_steps") else {}
        detector = named_steps.get("detector") if isinstance(named_steps, dict) else None

        if detector is None:
            QtWidgets.QMessageBox.critical(None, "模型不完整", "当前模型缺少集成检测器，请重新训练。")
            return

        transformed = feature_df_aligned
        for name, step in pipeline.steps[1:-1]:
            try:
                transformed = step.transform(transformed)
            except Exception as e:
                QtWidgets.QMessageBox.critical(None, "转换失败", f"特征变换失败（{name}）：{e}")
                return

        try:
            preds = detector.predict(transformed)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "预测失败", f"预测错误：{e}")
            return

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
            fallback_vote = 0.5 if vote_mean_meta is None else float(vote_mean_meta)
            vote_ratio = np.full(len(feature_df_aligned), fallback_vote, dtype=float)
        vote_ratio = np.asarray(vote_ratio, dtype=float)

        if threshold is None:
            threshold = getattr(detector, "threshold_", None)
        if threshold is None:
            threshold = float(np.quantile(scores, 0.05)) if len(scores) else 0.0

        vote_threshold = vote_threshold_meta
        if vote_threshold is None:
            vote_threshold = getattr(detector, "vote_threshold_", None)
        if vote_threshold is None:
            vote_threshold = float(np.mean(vote_ratio)) if len(vote_ratio) else 0.5
        vote_threshold = float(np.clip(vote_threshold, 0.0, 1.0))

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

        anomaly_confidence = risk_score.copy()

        out_df = df.copy()
        out_df["prediction"] = preds
        out_df["is_malicious"] = (preds == -1).astype(int)
        out_df["anomaly_score"] = scores
        out_df["anomaly_confidence"] = anomaly_confidence
        out_df["vote_ratio"] = vote_ratio
        out_df["risk_score"] = risk_score

        malicious = int(out_df["is_malicious"].sum())
        total = int(len(out_df))
        ratio = (malicious / total) if total else 0.0
        status = "异常" if malicious else "正常"

        out_dir = self._prediction_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_csv = os.path.join(out_dir, f"prediction_{base}_{stamp}.csv")
        try:
            out_df.to_csv(out_csv, index=False, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "保存失败", f"无法写出预测结果：{e}")
            return

        score_min = float(np.min(scores)) if len(scores) else 0.0
        score_max = float(np.max(scores)) if len(scores) else 0.0
        msg = [
            f"模型预测完成：{out_csv}",
            f"整体判定：{status}",
            f"检测包数：{total}",
            f"异常包数：{malicious} ({ratio:.2%})",
            f"自动阈值：{threshold:.6f}",
            f"投票阈值：{vote_threshold:.2f}",
            f"分数范围：{score_min:.4f} ~ {score_max:.4f}",
            f"平均风险分：{float(risk_score.mean()):.2%}",
        ]
        if missing_cols:
            msg.append(
                "已自动补齐缺失列: "
                + ", ".join(missing_cols[:10])
                + (" ..." if len(missing_cols) > 10 else "")
            )
        if extra_cols:
            msg.append(
                "已忽略未在训练中使用的列: "
                + ", ".join(extra_cols[:10])
                + (" ..." if len(extra_cols) > 10 else "")
            )
        if schema_version and schema_version != MODEL_SCHEMA_VERSION:
            msg.append(
                f"⚠ 模型 schema_version={schema_version} 与期望 {MODEL_SCHEMA_VERSION} 不一致"
            )
        if isinstance(threshold_breakdown_meta, dict) and threshold_breakdown_meta.get("adaptive") is not None:
            msg.append(
                "阈值拆解: 自适应 {:.6f} | 分位 {:.6f} | 鲁棒 {:.6f}".format(
                    float(threshold_breakdown_meta.get("adaptive")),
                    float(threshold_breakdown_meta.get("quantile", threshold)),
                    float(threshold_breakdown_meta.get("robust", threshold)),
                )
            )
        self.display_result("\n".join(msg))

        self._add_output(out_csv)
        self._last_out_csv = out_csv
        self._open_csv_paged(out_csv)

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

    def _on_rbf_settings_changed(self):
        if getattr(self, "_loading_settings", False):
            return
        if not hasattr(self, "_settings"):
            return
        try:
            self._settings.set("rbf_components", int(self.rbf_components_spin.value()))
            self._settings.set("rbf_gamma", float(self.rbf_gamma_spin.value()))
            self._settings.set("fusion_enabled", bool(self.fusion_checkbox.isChecked()))
            self._settings.set("fusion_alpha", float(self.fusion_alpha_spin.value()))
        except Exception:
            pass

    def _apply_saved_preferences(self):
        if not hasattr(self, "_settings"):
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
            self.fusion_alpha_spin.setEnabled(self.fusion_checkbox.isChecked())
        finally:
            self._loading_settings = False

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


def main() -> int:
    """启动 Qt 应用并展示主窗口。"""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        owns_app = True
    else:
        owns_app = False

    window = MainWindow()
    window.show()

    if owns_app:
        return app.exec_()

    # 若外部已有 QApplication，则仅返回 0，保持兼容
    return 0


if __name__ == "__main__":
    sys.exit(main())