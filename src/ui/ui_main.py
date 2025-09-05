## src/ui/ui_main.py
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import os

# 业务函数
from src.functions.info import get_pcap_features as info
from src.functions.feature_extractor import extract_features as get_fe
from src.functions.unsupervised_train import train_unsupervised_on_split as run_train
from src.functions.analyze_results import analyze_results as run_analysis

try:
    import pandas as pd
except Exception:
    pd = None

APP_STYLE = """
QWidget { background-color: #F5F6F8; font-family: "SF Pro Text"; color: #1d1d1f; font-size:13px; }
QTextEdit, QTableView { background-color: white; border:1px solid #E1E1E6; border-radius:8px; }
QLineEdit { background-color:white; border:1px solid #D0D0D6; border-radius:6px; padding:4px; }
QPushButton { background-color:#EAEAED; border:0; border-radius:8px; padding:6px 10px; min-height:28px;}
QPushButton:hover { background-color:#D8D8DC; } QPushButton:pressed { background-color:#CFCFD3; }
QHeaderView::section { background-color:#F1F1F3; padding:4px; border:1px solid #E8E8EA; }
"""

# ----------------- Worker Threads -----------------
class InfoWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            df = info(self.path)
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
            # get_fe 支持 progress_cb
            get_fe(self.pcap_path, self.csv_path, progress_cb=self.progress.emit)
            self.finished.emit(self.csv_path)
        except Exception as e:
            self.error.emit(str(e))

class TrainWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, split_dir, results_dir, models_dir):
        super().__init__()
        self.split_dir = split_dir
        self.results_dir = results_dir
        self.models_dir = models_dir

    def run(self):
        try:
            res = run_train(self.split_dir, self.results_dir, self.models_dir, progress_cb=self.progress.emit)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))

class AnalysisWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, results_csv, out_dir):
        super().__init__()
        self.results_csv = results_csv
        self.out_dir = out_dir

    def run(self):
        try:
            run_analysis(self.results_csv, self.out_dir, progress_cb=self.progress.emit)
            self.finished.emit(self.out_dir)
        except Exception as e:
            self.error.emit(str(e))

# ----------------- Main UI -----------------
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 600)
        MainWindow.setStyleSheet(APP_STYLE)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.h_layout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.h_layout.setContentsMargins(8,8,8,8)
        self.h_layout.setSpacing(10)

        # ---------------- Left ----------------
        self.left_frame = QtWidgets.QFrame(self.centralwidget)
        self.left_layout = QtWidgets.QVBoxLayout(self.left_frame)
        self.left_layout.setContentsMargins(0,0,0,0)
        self.left_layout.setSpacing(6)

        self.file_bar = QtWidgets.QWidget(self.left_frame)
        self.file_bar.setFixedHeight(25)
        fb_layout = QtWidgets.QHBoxLayout(self.file_bar)
        fb_layout.setContentsMargins(6,0,6,0)
        fb_layout.setSpacing(6)

        self.file_label = QtWidgets.QLabel("选择流量文件 (.pcap/.pcapng):", self.file_bar)
        self.file_edit = QtWidgets.QLineEdit(self.file_bar)
        self.file_edit.setPlaceholderText("请选择文件路径或点击 浏览 ...")
        self.file_button = QtWidgets.QPushButton("浏览", self.file_bar)
        fb_layout.addWidget(self.file_label)
        fb_layout.addWidget(self.file_edit)
        fb_layout.addWidget(self.file_button)
        self.left_layout.addWidget(self.file_bar)

        # Tabs
        self.display_tabs = QtWidgets.QTabWidget(self.left_frame)
        self.results_widget = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(self.results_widget)
        self.results_text = QtWidgets.QTextEdit()
        self.results_text.setReadOnly(True)
        r_layout.addWidget(self.results_text)
        self.display_tabs.addTab(self.results_widget,"结果（文本）")

        self.table_widget = QtWidgets.QWidget()
        t_layout = QtWidgets.QVBoxLayout(self.table_widget)
        self.table_view = QtWidgets.QTableView()
        self.table_view.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        t_layout.addWidget(self.table_view)
        self.display_tabs.addTab(self.table_widget,"流量表格")
        self.left_layout.addWidget(self.display_tabs)

        # Bottom
        self.bottom_bar = QtWidgets.QWidget(self.left_frame)
        self.bottom_bar.setFixedHeight(15)
        bb_layout = QtWidgets.QHBoxLayout(self.bottom_bar)
        bb_layout.setContentsMargins(6,0,6,0)
        bb_layout.addStretch()
        self.bottom_label = QtWidgets.QLabel("@2025  恶意流量检测系统")
        self.bottom_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        bb_layout.addWidget(self.bottom_label)
        self.left_layout.addWidget(self.bottom_bar)

        self.h_layout.addWidget(self.left_frame, stretch=3)

        # ---------------- Right ----------------
        self.right_frame = QtWidgets.QFrame(self.centralwidget)
        self.right_layout = QtWidgets.QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(6,6,6,6)
        self.right_layout.setSpacing(10)

        self.btn_view_info = QtWidgets.QPushButton("查看流量信息")
        self.btn_extract_features = QtWidgets.QPushButton("提取特征")
        self.btn_train_model = QtWidgets.QPushButton("训练模型")
        self.btn_run_analysis = QtWidgets.QPushButton("运行分析")
        self.btn_export = QtWidgets.QPushButton("导出结果")
        self.btn_clear = QtWidgets.QPushButton("清空显示")

        self.buttons = [
            self.btn_view_info,
            self.btn_extract_features,
            self.btn_train_model,
            self.btn_run_analysis,
            self.btn_export,
            self.btn_clear
        ]
        for btn in self.buttons:
            self.right_layout.addWidget(btn)
        self.right_layout.addStretch(1)
        self.h_layout.addWidget(self.right_frame, stretch=1)

        MainWindow.setCentralWidget(self.centralwidget)

        # ---------------- Signals ----------------
        self.file_button.clicked.connect(self._browse_file)
        self.btn_view_info.clicked.connect(self._on_view_info)
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_extract_features.clicked.connect(self._on_extract_features)
        self.btn_train_model.clicked.connect(self._on_train_model)
        self.btn_run_analysis.clicked.connect(self._on_run_analysis)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ----------------- Helpers -----------------
    def _project_root(self):
        # /src/ui/ui_main.py -> 项目根
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _default_split_dir(self):
        return os.path.join(self._project_root(), "data", "split")

    def _default_results_dir(self):
        return os.path.join(self._project_root(), "results")

    def _default_models_dir(self):
        return os.path.join(self._project_root(), "models")

    def _browse_file(self):
        path,_ = QtWidgets.QFileDialog.getOpenFileName(None,"选择 pcap 文件","","pcap (*.pcap *.pcapng);;所有文件 (*)")
        if path:
            self.file_edit.setText(path)
            self.display_result(f"已选择文件: {path}", append=True)

    def display_result(self, text:str, append=True):
        if append:
            self.results_text.append(text)
        else:
            self.results_text.setPlainText(text)

    def populate_table_from_df(self, df):
        if pd is None:
            raise RuntimeError("pandas required for populate_table_from_df")
        model = QtGui.QStandardItemModel()
        model.setColumnCount(len(df.columns))
        model.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for row in df.itertuples(index=False):
            items = [QtGui.QStandardItem(str(x)) for x in row]
            model.appendRow(items)
        self.table_view.setModel(model)
        self.table_view.resizeColumnsToContents()
        self.display_tabs.setCurrentWidget(self.table_widget)

    def set_button_progress(self, button, progress):
        button.setStyleSheet(f"""
            QPushButton {{
                border-radius: 8px;
                color:#1d1d1f;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #3399FF stop:{progress/100} #3399FF stop:{progress/100} #EAEAED stop:1 #EAEAED);
            }}
        """)

    # ----------------- Slots -----------------
    def _on_view_info(self):
        path = self.file_edit.text().strip()
        if not path:
            self.display_result("请先选择文件")
            return
        self.display_result(f"[INFO] 正在解析 {path} ...", append=True)
        self.btn_view_info.setEnabled(False)

        self.worker = InfoWorker(path)
        self.worker.progress.connect(lambda p: self.set_button_progress(self.btn_view_info,p))
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_extract_features(self):
        path = self.file_edit.text().strip()
        if not path:
            self.display_result("请先选择文件")
            return

        csv_path, _ = QtWidgets.QFileDialog.getSaveFileName(None, "保存特征CSV", "", "CSV Files (*.csv)")
        if not csv_path:
            self.display_result("[INFO] 已取消保存")
            return

        self.display_result(f"[INFO] 正在提取特征 -> {csv_path} ...")
        self.btn_extract_features.setEnabled(False)

        self.fe_worker = FeatureWorker(path, csv_path)
        self.fe_worker.progress.connect(lambda p: self.set_button_progress(self.btn_extract_features, p))
        self.fe_worker.finished.connect(lambda csv: self._on_fe_finished(csv))
        self.fe_worker.error.connect(lambda e: self._on_fe_error(e))
        self.fe_worker.start()

    def _on_fe_finished(self, csv_path):
        self.btn_extract_features.setEnabled(True)
        self.set_button_progress(self.btn_extract_features,0)
        self.btn_extract_features.setStyleSheet("")
        self.display_result(f"[INFO] 特征提取完成，CSV已保存: {csv_path}")

    def _on_fe_error(self, msg):
        self.btn_extract_features.setEnabled(True)
        self.set_button_progress(self.btn_extract_features,0)
        self.btn_extract_features.setStyleSheet("")
        self.display_result(f"[错误] 特征提取失败: {msg}")

    def _on_train_model(self):
        """
        优先自动使用 项目根/data/split 作为训练目录：
        - 若存在且包含 .pcap/.pcapng -> 直接开始训练
        - 否则弹出“选择文件夹”对话框（只允许选文件夹）
        """
        project_root = self._project_root()
        default_split = self._default_split_dir()

        def dir_has_pcap(d):
            try:
                return any(name.lower().endswith((".pcap", ".pcapng")) for name in os.listdir(d))
            except Exception:
                return False

        use_dir = None
        if os.path.isdir(default_split) and dir_has_pcap(default_split):
            use_dir = default_split
        else:
            # 回退到对话框，默认打开项目根或上次选择的文件所在目录
            guess_dir = project_root
            last_file = self.file_edit.text().strip()
            if last_file:
                guess_dir = os.path.dirname(last_file)

            options = QtWidgets.QFileDialog.Options()
            options |= QtWidgets.QFileDialog.ShowDirsOnly
            chosen = QtWidgets.QFileDialog.getExistingDirectory(
                None,
                "选择切割后PCAP目录（里面应有 .pcap/.pcapng 文件）",
                guess_dir,
                options
            )
            if not chosen:
                self.display_result("已取消训练。")
                return
            if not dir_has_pcap(chosen):
                QtWidgets.QMessageBox.warning(
                    None,
                    "目录里没有PCAP",
                    "该目录下没有 .pcap / .pcapng 文件。\n\n请把切割后的包放进这个目录，或重新选择包含小包的目录。"
                )
                return
            use_dir = chosen

        # 输出目录统一放到项目根
        results_dir = self._default_results_dir()
        models_dir  = self._default_models_dir()
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir,  exist_ok=True)

        self.display_result(f"[INFO] 开始训练，目录: {use_dir}")
        self.btn_train_model.setEnabled(False)

        self.train_worker = TrainWorker(use_dir, results_dir, models_dir)
        self.train_worker.progress.connect(lambda p: self.set_button_progress(self.btn_train_model, p))
        self.train_worker.finished.connect(self._on_train_finished)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.start()

    def _on_train_finished(self, res):
        self.btn_train_model.setEnabled(True)
        self.set_button_progress(self.btn_train_model,0)
        self.btn_train_model.setStyleSheet("")
        # 简要展示关键结果
        msg = (f"results:\n- {res.get('results_csv')}\n- {res.get('summary_csv')}\n"
               f"models:\n- {res.get('model_path')}\n- {res.get('scaler_path')}\n"
               f"packets={res.get('packets')} malicious={res.get('malicious')}")
        self.display_result(f"[INFO] 训练完成：\n{msg}")

    def _on_train_error(self, msg):
        self.btn_train_model.setEnabled(True)
        self.set_button_progress(self.btn_train_model,0)
        self.btn_train_model.setStyleSheet("")
        self.display_result(f"[错误] 训练失败: {msg}")

    def _on_run_analysis(self):
        """
        优先自动寻找 项目根/results/iforest_results.csv
        - 若存在：直接分析并输出到 项目根/results/analysis
        - 否则：让用户选择CSV
        """
        results_dir = self._default_results_dir()
        results_csv_auto = os.path.join(results_dir, "iforest_results.csv")
        out_dir = os.path.join(results_dir, "analysis")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(results_csv_auto):
            csv_path = results_csv_auto
        else:
            csv_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "选择检测结果CSV", "", "CSV Files (*.csv)")
            if not csv_path:
                self.display_result("请先选择结果CSV")
                return

        self.display_result(f"[INFO] 正在分析结果 -> {out_dir}")
        self.btn_run_analysis.setEnabled(False)

        self.analysis_worker = AnalysisWorker(csv_path, out_dir)
        self.analysis_worker.progress.connect(lambda p: self.set_button_progress(self.btn_run_analysis, p))
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.start()

    def _on_analysis_finished(self, out_dir):
        self.btn_run_analysis.setEnabled(True)
        self.set_button_progress(self.btn_run_analysis,0)
        self.btn_run_analysis.setStyleSheet("")
        self.display_result(f"[INFO] 分析完成，图表保存在 {out_dir}")

    def _on_analysis_error(self, msg):
        self.btn_run_analysis.setEnabled(True)
        self.set_button_progress(self.btn_run_analysis,0)
        self.btn_run_analysis.setStyleSheet("")
        self.display_result(f"[错误] 分析失败: {msg}")

    def _on_worker_finished(self, df):
        self.btn_view_info.setEnabled(True)
        self.set_button_progress(self.btn_view_info,0)
        self.btn_view_info.setStyleSheet("")
        if df.empty:
            self.display_result("解析完成，但未找到流量数据。")
        else:
            self.display_result(df.head().to_string(), append=False)
            self.populate_table_from_df(df)
            self.bottom_label.setText(f"共 {len(df)} 条流，文件: {os.path.basename(self.file_edit.text())} @2025")

    def _on_worker_error(self,msg):
        self.btn_view_info.setEnabled(True)
        self.set_button_progress(self.btn_view_info,0)
        self.btn_view_info.setStyleSheet("")
        self.display_result(f"[错误] 解析失败: {msg}")

    def _on_clear(self):
        self.results_text.clear()
        self.table_view.setModel(None)
        self.display_tabs.setCurrentWidget(self.results_widget)
        self.bottom_label.setText("@2025  恶意流量检测系统")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow","恶意流量检测系统 — 主功能页面"))

# ----------------- Quick run -----------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
