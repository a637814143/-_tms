# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import sys, os, platform, subprocess, math, shutil
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
from src.functions.vectorize import vectorize_feature_dir as vec_dir

try:
    import pandas as pd
except Exception:
    pd = None

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

# ======= 固定输出路径（与你之前一致）=======
_DATA_BASE = r"D:\pythonProject8\data"
PATHS = {
    "split": os.path.join(_DATA_BASE, "split"),
    "csv_info": os.path.join(_DATA_BASE, "CSV", "info"),
    "csv_feature": os.path.join(_DATA_BASE, "CSV", "feature"),
    "models": os.path.join(_DATA_BASE, "models"),
    "results_analysis": os.path.join(_DATA_BASE, "results", "analysis"),
    "results_pred": os.path.join(_DATA_BASE, "results", "modelprediction"),
    "results_abnormal": os.path.join(_DATA_BASE, "results", "abnormal"),
    "results_vector": os.path.join(_DATA_BASE, "results", "vector"),
}
for _p in PATHS.values():
    os.makedirs(_p, exist_ok=True)

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


class VectorWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, feature_dir, out_dir):
        super().__init__()
        self.feature_dir = feature_dir
        self.out_dir = out_dir

    def run(self):
        try:
            result = vec_dir(self.feature_dir, self.out_dir, progress_cb=self.progress.emit)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, input_path, results_dir, models_dir):
        super().__init__()
        self.input_path = input_path
        self.results_dir = results_dir
        self.models_dir = models_dir

    def run(self):
        try:
            res = run_train(self.input_path, self.results_dir, self.models_dir, progress_cb=self.progress.emit)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class AnalysisWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, results_csv, out_dir):
        super().__init__()
        self.results_csv = results_csv
        self.out_dir = out_dir

    def run(self):
        try:
            result = run_analysis(self.results_csv, self.out_dir, progress_cb=self.progress.emit)
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

        # 分页状态
        self._csv_paged_path: Optional[str] = None
        self._csv_total_rows: Optional[int] = None
        self._csv_current_page: int = 1

        # worker
        self.worker: Optional[InfoWorker] = None
        self.vector_worker: Optional[VectorWorker] = None

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
        self.bottom_label = QtWidgets.QLabel("@2025  恶意流量检测系统")
        self.bottom_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        bb.addWidget(self.bottom_label)
        self.left_layout.addWidget(self.bottom_bar)

    def _build_right_panel(self):
        self.right_frame = QtWidgets.QFrame(self.centralwidget)
        self.right_layout = QtWidgets.QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(6, 6, 6, 6)
        self.right_layout.setSpacing(10)

        self.btn_view = QtWidgets.QPushButton("查看流量信息")
        self.btn_fe = QtWidgets.QPushButton("提取特征")
        self.btn_vector = QtWidgets.QPushButton("数据处理（向量化）")
        self.btn_train = QtWidgets.QPushButton("训练模型")
        self.btn_analysis = QtWidgets.QPushButton("运行分析")
        self.btn_predict = QtWidgets.QPushButton("加载模型预测")
        self.btn_export = QtWidgets.QPushButton("导出结果（异常）")
        self.btn_clear = QtWidgets.QPushButton("清空显示")

        for b in [
            self.btn_view,
            self.btn_fe,
            self.btn_vector,
            self.btn_train,
            self.btn_analysis,
            self.btn_predict,
            self.btn_export,
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
        self.btn_vector.clicked.connect(self._on_vectorize_features)
        self.btn_train.clicked.connect(self._on_train_model)
        self.btn_analysis.clicked.connect(self._on_run_analysis)
        self.btn_predict.clicked.connect(self._on_predict)

        self.btn_page_prev.clicked.connect(self._on_page_prev)
        self.btn_page_next.clicked.connect(self._on_page_next)
        self.page_size_spin.valueChanged.connect(self._on_page_size_changed)
        self.btn_show_all.clicked.connect(self._show_full_preview)

        self.output_list.customContextMenuRequested.connect(self._on_output_ctx_menu)
        self.output_list.itemDoubleClicked.connect(self._on_output_double_click)

    # --------- 路径小工具 ----------
    def _project_root(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    def _default_split_dir(self):
        return PATHS["split"]
    def _default_results_dir(self):
        return os.path.join(self._project_root(), "results")
    def _default_models_dir(self):
        return PATHS["models"]
    def _default_csv_info_dir(self):
        return PATHS["csv_info"]
    def _default_csv_feature_dir(self):
        return PATHS["csv_feature"]
    def _analysis_out_dir(self):
        return PATHS["results_analysis"]
    def _prediction_out_dir(self):
        return PATHS["results_pred"]
    def _abnormal_out_dir(self):
        return PATHS["results_abnormal"]
    def _vector_out_dir(self):
        return PATHS["results_vector"]
    def _browse_compat(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择 pcap 文件", "", "pcap (*.pcap *.pcapng);;所有文件 (*)")
        if not p:
            d = QtWidgets.QFileDialog.getExistingDirectory(None, "选择包含多个小包的目录（如 data/split）", self._default_split_dir())
            if d:
                self.file_edit.setText(d); self.display_result(f"已选择目录: {d}", True)
            return
        self.file_edit.setText(p); self.display_result(f"已选择文件: {p}", True)

    def _choose_file(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择 pcap 文件", "", "pcap (*.pcap *.pcapng);;所有文件 (*)")
        if p:
            self.file_edit.setText(p); self.display_result(f"已选择文件: {p}", True)

    def _choose_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(None, "选择包含多个小包的目录（如 data/split）", self._default_split_dir())
        if d:
            self.file_edit.setText(d); self.display_result(f"已选择目录: {d}", True)

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

        self.btn_view.setEnabled(False); self.set_button_progress(self.btn_view, 0)

        self.worker = InfoWorker(
            path=path, workers=workers,
            mode=("all" if mode == "auto" and os.path.isdir(path) else ("file" if mode == "auto" and os.path.isfile(path) else mode)),
            batch_size=batch, start_index=start,
            files=file_list if os.path.isdir(path) else None,
            proto_filter=proto, port_whitelist_text=wl, port_blacklist_text=bl,
            fast=True,
            fast_time_budget=1.0,
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

        if pd is not None and isinstance(df, pd.DataFrame):
            df = self._auto_tag_dataframe(df)

        out_csv = getattr(df, "attrs", {}).get("out_csv", None)
        files_total = getattr(df, "attrs", {}).get("files_total", None)
        errs = getattr(df, "attrs", {}).get("errors", "")
        fast_summary = getattr(df, "attrs", {}).get("fast_summary", False)

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
            if fast_summary:
                truncated_count = 0
                if hasattr(df, "columns") and "truncated" in df.columns:
                    try:
                        truncated_count = int(df["truncated"].sum())
                    except Exception:
                        truncated_count = 0
                note = "（快速模式，仅展示核心统计" + (
                    f"；其中 {truncated_count} 个因时间限制被截断" if truncated_count else ""
                ) + "）"
            else:
                note = "（表格仅显示前 {limit} 行；全部已写入 CSV）".format(
                    limit=PREVIEW_LIMIT_FOR_TABLE
                )
            self.display_result(f"[INFO] 解析完成{note}.\n{head_txt}", append=False)
            rows = len(df) if hasattr(df, "__len__") else 0
            self.bottom_label.setText(f"预览 {min(rows, 20)} 行；共处理文件 {files_total if files_total is not None else '?'} 个  @2025")
        if errs:
            for e in errs.split("\n"):
                if e.strip(): self.display_result(e)

    def _on_worker_error(self, msg):
        self.btn_view.setEnabled(True); self.reset_button_progress(self.btn_view)
        self.worker = None
        QtWidgets.QMessageBox.critical(None, "解析失败", msg)
        self.display_result(f"[错误] 解析失败: {msg}")

    # --------- 导出异常 ----------
    def _on_export_results(self):
        df = self._last_preview_df
        if df is None or (hasattr(df, "empty") and df.empty):
            QtWidgets.QMessageBox.warning(None, "没有数据", "请先查看或加载带预测的数据。"); return
        out_dir = self._abnormal_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        export_df = None
        if "prediction" in df.columns:
            export_df = df[df["prediction"] == -1].copy()
        elif "anomaly_score" in df.columns:
            export_df = df[df["anomaly_score"] > 0].copy()
        else:
            QtWidgets.QMessageBox.information(None, "无异常标记", "没有 prediction 或 anomaly_score 列，无法筛选异常。")
            return
        if export_df.empty:
            QtWidgets.QMessageBox.information(None, "没有异常", "当前数据中未检测到异常行。"); return

        outp = os.path.join(out_dir, f"abnormal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        export_df.to_csv(outp, index=False, encoding="utf-8")
        self._add_output(outp)
        self.display_result(f"[INFO] 已导出异常CSV：{outp}")
        self._open_csv_paged(outp)

    # --------- 提取特征（按顶部路径） ----------
    def _on_extract_features(self):
        path = self.file_edit.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(None, "未选择路径", "请先在顶部选择 pcap 文件或目录。"); return

        out_dir = self._default_csv_feature_dir()
        os.makedirs(out_dir, exist_ok=True)

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
        QtWidgets.QMessageBox.critical(None, "特征提取失败", msg)
        self.display_result(f"[错误] 特征提取失败: {msg}")

    # --------- 向量化处理（基于特征 CSV） ----------
    def _on_vectorize_features(self):
        feature_dir = self._default_csv_feature_dir()
        if not os.path.isdir(feature_dir):
            QtWidgets.QMessageBox.warning(None, "目录不存在", f"特征目录不存在：{feature_dir}")
            return

        csv_files = [n for n in os.listdir(feature_dir) if n.lower().endswith(".csv")]
        if not csv_files:
            QtWidgets.QMessageBox.information(None, "没有特征数据", "请先完成特征提取后再进行向量化处理。")
            return

        out_dir = self._vector_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        self.display_result(f"[INFO] 向量化处理：{feature_dir} -> {out_dir}")
        self.btn_vector.setEnabled(False); self.set_button_progress(self.btn_vector, 1)
        self.vector_worker = VectorWorker(feature_dir, out_dir)
        self.vector_worker.progress.connect(lambda p: self.set_button_progress(self.btn_vector, p))
        self.vector_worker.finished.connect(self._on_vector_finished)
        self.vector_worker.error.connect(self._on_vector_error)
        self.vector_worker.start()

    def _on_vector_finished(self, result):
        self.btn_vector.setEnabled(True)
        self.set_button_progress(self.btn_vector, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_vector))
        self.vector_worker = None

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

        summary = f"[INFO] 向量化完成：{total_rows or 0} 条记录，{total_cols or 0} 个特征。"
        if dataset:
            summary += f" 数据集：{dataset}"
        self.display_result(summary)

    def _on_vector_error(self, msg):
        self.btn_vector.setEnabled(True)
        self.reset_button_progress(self.btn_vector)
        QtWidgets.QMessageBox.critical(None, "向量化失败", msg)
        self.display_result(f"[错误] 向量化失败: {msg}")
        self.vector_worker = None

    # --------- 训练模型（按顶部路径） ----------
    def _on_train_model(self):
        selected_path = self.file_edit.text().strip()
        vector_dir = self._vector_out_dir()

        if selected_path and os.path.exists(selected_path):
            path = selected_path
        else:
            path = vector_dir
            if selected_path:
                self.display_result(f"[WARN] 选择的路径不存在，自动改用向量目录：{vector_dir}")
            else:
                self.display_result(f"[INFO] 未选择路径，默认使用向量目录：{vector_dir}")

        res_dir = self._default_results_dir()
        mdl_dir = self._default_models_dir()
        os.makedirs(res_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)

        self.display_result(f"[INFO] 开始训练，输入: {path}")
        self.btn_train.setEnabled(False); self.set_button_progress(self.btn_train, 1)
        self.train_worker = TrainWorker(path, res_dir, mdl_dir)
        self.train_worker.progress.connect(lambda p: self.set_button_progress(self.btn_train, p))
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.start()

    def _on_train_finished(self, res):
        self.btn_train.setEnabled(True)
        self.set_button_progress(self.btn_train, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_train))
        msg = (f"results:\n- {res.get('results_csv')}\n- {res.get('summary_csv')}\n"
               f"models:\n- {res.get('model_path')}\n- {res.get('scaler_path')}\n"
               f"flows={res.get('flows')} malicious={res.get('malicious')}")
        self.display_result(f"[INFO] 训练完成：\n{msg}")
        for k in ("results_csv", "summary_csv", "model_path", "scaler_path"):
            p = res.get(k)
            if p and os.path.exists(p): self._add_output(p)
        if res.get("results_csv") and os.path.exists(res["results_csv"]):
            self._open_csv_paged(res["results_csv"])

    def _on_train_error(self, msg):
        self.btn_train.setEnabled(True); self.reset_button_progress(self.btn_train)
        QtWidgets.QMessageBox.critical(None, "训练失败", msg)
        self.display_result(f"[错误] 训练失败: {msg}")

    # --------- 运行分析 ----------
    def _on_run_analysis(self):
        out_dir = self._analysis_out_dir()
        os.makedirs(out_dir, exist_ok=True)
        auto = os.path.join(self._default_results_dir(), "iforest_results.csv")
        if os.path.exists(auto):
            csv = auto
        else:
            csv, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择检测结果CSV", "", "CSV Files (*.csv)")
            if not csv:
                self.display_result("请先选择结果CSV"); return

        self.display_result(f"[INFO] 正在分析结果 -> {out_dir}")
        self.btn_analysis.setEnabled(False); self.set_button_progress(self.btn_analysis, 1)
        self.analysis_worker = AnalysisWorker(csv, out_dir)
        self.analysis_worker.progress.connect(lambda p: self.set_button_progress(self.btn_analysis, p))
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.start()

    def _on_analysis_finished(self, result):
        self.btn_analysis.setEnabled(True)
        self.set_button_progress(self.btn_analysis, 100)
        QtCore.QTimer.singleShot(300, lambda: self.reset_button_progress(self.btn_analysis))
        out_dir = None
        plot_paths: List[str] = []
        top20_csv: Optional[str] = None

        if isinstance(result, dict):
            out_dir = result.get("out_dir") or None
            plots = result.get("plots") or []
            if isinstance(plots, (list, tuple)):
                plot_paths = [p for p in plots if isinstance(p, str)]
            elif isinstance(plots, str):
                plot_paths = [plots]
            top20_csv = result.get("top20_csv") if isinstance(result.get("top20_csv"), str) else None
        else:
            out_dir = str(result) if result is not None else None

        if not out_dir:
            out_dir = self._analysis_out_dir()

        self.display_result(f"[INFO] 分析完成，图表保存在 {out_dir}")

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

        fallback_names = [
            "top10_malicious_ratio.png",
            "anomaly_score_distribution.png",
            "top20_packets.csv",
        ]
        for name in fallback_names:
            _mark(os.path.join(out_dir, name))

        summary_csv = os.path.join(self._default_results_dir(), "summary_by_file.csv")
        if os.path.exists(summary_csv):
            _mark(summary_csv)

        if out_dir and os.path.isdir(out_dir):
            _mark(out_dir)

    def _on_analysis_error(self, msg):
        self.btn_analysis.setEnabled(True); self.reset_button_progress(self.btn_analysis)
        QtWidgets.QMessageBox.critical(None, "分析失败", msg)
        self.display_result(f"[错误] 分析失败: {msg}")

    # --------- 模型预测（支持 Pipeline / 模型+scaler / 仅模型） ----------
    def _on_predict(self):
        csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择特征CSV", self._default_csv_feature_dir(), "CSV (*.csv)"
        )
        if not csv_path: return
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "读取失败", f"无法读取 CSV：{e}"); return
        if df.empty:
            QtWidgets.QMessageBox.information(None, "空数据", "该 CSV 没有数据行。"); return

        from joblib import load as joblib_load

        mdl_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择模型（isoforest.joblib 或 Pipeline）",
            self._default_models_dir(), "JOBLIB (*.joblib);;All (*)"
        )
        if not mdl_path: return

        try:
            mdl_obj = joblib_load(mdl_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "加载失败", f"无法加载模型：{e}"); return

        def _is_pipeline(o): return hasattr(o, "named_steps") and hasattr(o, "predict")
        def _is_scaler(o):
            name = o.__class__.__name__.lower()
            return (hasattr(o, "transform") and not hasattr(o, "predict")) or ("scaler" in name)
        def _is_estimator(o): return hasattr(o, "predict") and not _is_pipeline(o)

        is_pipeline = _is_pipeline(mdl_obj)
        scaler = None; estimator = None; pipeline = None

        if is_pipeline:
            pipeline = mdl_obj
        else:
            other_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "选择 scaler.joblib（若无可取消）",
                self._default_models_dir(), "JOBLIB (*.joblib);;All (*)"
            )
            other_obj = None
            if other_path:
                try:
                    other_obj = joblib_load(other_path)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(None, "加载失败", f"无法加载标准化器：{e}"); return

            if _is_scaler(mdl_obj) and (other_obj is not None) and _is_estimator(other_obj):
                scaler, estimator = mdl_obj, other_obj
            elif (other_obj is not None) and _is_scaler(other_obj) and _is_estimator(mdl_obj):
                scaler, estimator = other_obj, mdl_obj
            elif _is_estimator(mdl_obj) and other_obj is None:
                estimator = mdl_obj
            else:
                QtWidgets.QMessageBox.critical(
                    None, "加载失败",
                    "无法判定哪个是模型哪个是scaler：\n"
                    " - 如果你保存的是Pipeline，只需要选一次模型文件即可；\n"
                    " - 如果是分离的 scaler + 模型，请两个都选择，顺序随意。"
                )
                return

        use_df = df.copy()
        if pipeline is None and scaler is not None:
            need_k = int(getattr(scaler, "n_features_in_", 0) or 0)
            if need_k <= 0:
                QtWidgets.QMessageBox.warning(None, "无法确定列数", "StandardScaler 未包含 n_features_in_ 信息。")
                return
            dlg = FeaturePickDialog(list(use_df.columns), need_k)
            if dlg.exec_() != QtWidgets.QDialog.Accepted: return
            cols = dlg.selected_columns()
            for c in cols:
                if c not in use_df.columns: use_df[c] = 0
            X = use_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        else:
            X = use_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).values

        try:
            if pipeline is not None:
                pred = pipeline.predict(X)
                if hasattr(pipeline, "decision_function"):
                    score = -pipeline.decision_function(X)
                elif hasattr(pipeline, "score_samples"):
                    score = -pipeline.score_samples(X)
                else:
                    score = None
            else:
                if scaler is not None:
                    X = scaler.transform(X)
                pred = estimator.predict(X)
                if hasattr(estimator, "decision_function"):
                    score = -estimator.decision_function(X)
                elif hasattr(estimator, "score_samples"):
                    score = -estimator.score_samples(X)
                else:
                    score = None
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "预测失败", f"预测错误：{e}"); return

        out_df = df.copy()
        out_df["prediction"] = pred
        if score is not None:
            out_df["anomaly_score"] = score

        pred_dir = self._prediction_out_dir()
        os.makedirs(pred_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_csv = os.path.join(pred_dir, f"{base}_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        try:
            out_df.to_csv(out_csv, index=False, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "保存失败", f"无法写出预测结果：{e}"); return

        self.display_result(f"[INFO] 预测完成：{out_csv}")
        self._add_output(out_csv)
        self._open_csv_paged(out_csv)

    # --------- 输出列表 ----------
    def _add_output(self, path):
        if not path: return
        it = QtWidgets.QListWidgetItem(os.path.basename(path))
        it.setToolTip(path); it.setData(QtCore.Qt.UserRole, path)
        self.output_list.addItem(it); self.output_list.scrollToBottom()

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

    # --------- 清空 ----------
    def _on_clear(self):
        if getattr(self, "worker", None) and self.worker.isRunning():
            self.worker.requestInterruption()
        if getattr(self, "vector_worker", None) and self.vector_worker.isRunning():
            self.vector_worker.requestInterruption()

        self.results_text.setUpdatesEnabled(False)
        self.table_view.setUpdatesEnabled(False)
        self.output_list.setUpdatesEnabled(False)
        try:
            self.results_text.clear()
            self.table_view.setSortingEnabled(False)
            self.table_view.setModel(None)
            self.display_tabs.setCurrentWidget(self.results_widget)
            self.output_list.clear()
            self.bottom_label.setText("@2025  恶意流量检测系统")
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


def main():
    """启动 Qt 应用并展示主窗口。"""
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    if owns_app:
        sys.exit(app.exec_())
    return window


if __name__ == "__main__":
    main()