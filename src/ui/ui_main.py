# -*- coding: utf-8 -*-
"""PyQt5 GUI integrating the PCAP extraction → vectorization → training → detection workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

from PyQt5 import QtCore, QtWidgets

from PCAP import (
    DetectionResult,
    TrainingSummary,
    extract_sources_to_jsonl,
    train_hist_gradient_boosting,
    vectorize_jsonl_files,
    detect_pcap_with_model,
)
from src.configuration import project_root


class ExtractWorker(QtCore.QThread):
    """Background worker that streams PCAP extraction into a JSONL file."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    message = QtCore.pyqtSignal(str)

    def __init__(self, source: Path, output: Path, workers: Optional[int] = None) -> None:
        super().__init__()
        self.source = source
        self.output = output
        self.workers = workers

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            summary = extract_sources_to_jsonl(
                self.source,
                self.output,
                max_workers=self.workers,
                progress_callback=self.progress.emit,
                text_callback=self.message.emit,
                show_progress=False,
            )
        except Exception as exc:  # pragma: no cover - surfacing to UI
            self.error.emit(str(exc))
            return
        self.finished.emit(summary)


class VectorizeWorker(QtCore.QThread):
    """Worker that converts JSONL extraction output into the mandated CSV layout."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, inputs: Sequence[Path], output: Path, label: Optional[int] = None) -> None:
        super().__init__()
        self.inputs = list(inputs)
        self.output = output
        self.label = label

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            summary = vectorize_jsonl_files(
                self.inputs,
                self.output,
                label_override=self.label,
                show_progress=False,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(summary)


class TrainWorker(QtCore.QThread):
    """Worker wrapping ``train_hist_gradient_boosting`` for GUI usage."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)

    def __init__(self, dataset: Path, model: Path, iterations: Optional[int]) -> None:
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.iterations = iterations

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            self.progress.emit(5)
            kwargs = {"max_iter": self.iterations} if self.iterations else {}
            summary = train_hist_gradient_boosting(self.dataset, self.model, **kwargs)
            self.progress.emit(100)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(summary)


class DetectWorker(QtCore.QThread):
    """Worker executing inference on PCAP files using a trained model."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, model: Path, pcap: Path) -> None:
        super().__init__()
        self.model = model
        self.pcap = pcap

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            result = detect_pcap_with_model(self.model, self.pcap)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class MainWindow(QtWidgets.QMainWindow):
    """Main application window orchestrating the PCAP workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PCAP 工作流工具")
        self.resize(960, 720)

        self._base_dir = project_root() / "artifacts"
        self._jsonl_dir = self._base_dir / "jsonl"
        self._csv_dir = self._base_dir / "datasets"
        self._model_dir = self._base_dir / "models"
        for folder in (self._base_dir, self._jsonl_dir, self._csv_dir, self._model_dir):
            folder.mkdir(parents=True, exist_ok=True)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(self._build_extract_group())
        layout.addWidget(self._build_vectorize_group())
        layout.addWidget(self._build_train_group())
        layout.addWidget(self._build_detect_group())
        layout.addWidget(self._build_log_group())

        self.setCentralWidget(central)

        self._extract_worker: Optional[ExtractWorker] = None
        self._vectorize_worker: Optional[VectorizeWorker] = None
        self._train_worker: Optional[TrainWorker] = None
        self._detect_worker: Optional[DetectWorker] = None
        self._jsonl_inputs: List[Path] = []

    # ------------------------------------------------------------------
    # UI builders
    def _build_extract_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("1️⃣ 提取特征 (PCAP → JSONL)")
        layout = QtWidgets.QGridLayout(group)

        self.pcap_path_edit = QtWidgets.QLineEdit()
        self.pcap_path_edit.setPlaceholderText("选择 PCAP 文件或目录…")
        layout.addWidget(self.pcap_path_edit, 0, 0, 1, 3)

        choose_file = QtWidgets.QPushButton("选择文件")
        choose_file.clicked.connect(self._choose_pcap_file)
        layout.addWidget(choose_file, 0, 3)

        choose_dir = QtWidgets.QPushButton("选择目录")
        choose_dir.clicked.connect(self._choose_pcap_directory)
        layout.addWidget(choose_dir, 0, 4)

        self.jsonl_output_edit = QtWidgets.QLineEdit(str(self._jsonl_dir / "features.jsonl"))
        layout.addWidget(self.jsonl_output_edit, 1, 0, 1, 3)

        browse_jsonl = QtWidgets.QPushButton("输出路径…")
        browse_jsonl.clicked.connect(self._choose_jsonl_output)
        layout.addWidget(browse_jsonl, 1, 3)

        self.extract_progress = QtWidgets.QProgressBar()
        self.extract_progress.setRange(0, 100)
        self.extract_progress.setValue(0)
        layout.addWidget(self.extract_progress, 1, 4)

        self.extract_btn = QtWidgets.QPushButton("开始提取")
        self.extract_btn.clicked.connect(self._start_extraction)
        layout.addWidget(self.extract_btn, 2, 4)

        self.extract_status = QtWidgets.QLabel()
        self.extract_status.setWordWrap(True)
        layout.addWidget(self.extract_status, 2, 0, 1, 4)

        return group

    def _build_vectorize_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("2️⃣ 向量化 (JSONL → CSV)")
        layout = QtWidgets.QGridLayout(group)

        self.jsonl_inputs_edit = QtWidgets.QLineEdit()
        self.jsonl_inputs_edit.setPlaceholderText("选择一个或多个 JSONL 文件…")
        layout.addWidget(self.jsonl_inputs_edit, 0, 0, 1, 3)

        choose_jsonl = QtWidgets.QPushButton("选择 JSONL…")
        choose_jsonl.clicked.connect(self._choose_jsonl_inputs)
        layout.addWidget(choose_jsonl, 0, 3)

        clear_jsonl = QtWidgets.QPushButton("清除")
        clear_jsonl.clicked.connect(self._clear_jsonl_inputs)
        layout.addWidget(clear_jsonl, 0, 4)

        self.csv_output_edit = QtWidgets.QLineEdit(str(self._csv_dir / "dataset.csv"))
        layout.addWidget(self.csv_output_edit, 1, 0, 1, 3)

        browse_csv = QtWidgets.QPushButton("输出路径…")
        browse_csv.clicked.connect(self._choose_csv_output)
        layout.addWidget(browse_csv, 1, 3)

        self.vectorize_progress = QtWidgets.QProgressBar()
        self.vectorize_progress.setRange(0, 100)
        self.vectorize_progress.setValue(0)
        layout.addWidget(self.vectorize_progress, 1, 4)

        self.vectorize_btn = QtWidgets.QPushButton("开始向量化")
        self.vectorize_btn.clicked.connect(self._start_vectorization)
        layout.addWidget(self.vectorize_btn, 2, 4)

        self.vectorize_status = QtWidgets.QLabel()
        self.vectorize_status.setWordWrap(True)
        layout.addWidget(self.vectorize_status, 2, 0, 1, 4)

        return group

    def _build_train_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("3️⃣ 模型训练 (CSV → model.txt)")
        layout = QtWidgets.QGridLayout(group)

        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setPlaceholderText("选择特征 CSV 数据集…")
        layout.addWidget(self.dataset_edit, 0, 0, 1, 3)

        dataset_btn = QtWidgets.QPushButton("选择 CSV…")
        dataset_btn.clicked.connect(self._choose_dataset)
        layout.addWidget(dataset_btn, 0, 3)

        self.model_output_edit = QtWidgets.QLineEdit(str(self._model_dir / "model.txt"))
        layout.addWidget(self.model_output_edit, 1, 0, 1, 3)

        browse_model = QtWidgets.QPushButton("输出路径…")
        browse_model.clicked.connect(self._choose_model_output)
        layout.addWidget(browse_model, 1, 3)

        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(50, 2000)
        self.iterations_spin.setValue(300)
        layout.addWidget(QtWidgets.QLabel("Boosting 轮数"), 2, 0)
        layout.addWidget(self.iterations_spin, 2, 1)

        self.train_progress = QtWidgets.QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress, 2, 2, 1, 2)

        self.train_btn = QtWidgets.QPushButton("开始训练")
        self.train_btn.clicked.connect(self._start_training)
        layout.addWidget(self.train_btn, 2, 4)

        self.train_status = QtWidgets.QLabel()
        self.train_status.setWordWrap(True)
        layout.addWidget(self.train_status, 3, 0, 1, 5)

        return group

    def _build_detect_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("4️⃣ 模型检测 (PCAP → 预测结果)")
        layout = QtWidgets.QGridLayout(group)

        self.model_input_edit = QtWidgets.QLineEdit()
        self.model_input_edit.setPlaceholderText("选择训练好的模型…")
        layout.addWidget(self.model_input_edit, 0, 0, 1, 3)

        choose_model = QtWidgets.QPushButton("选择模型…")
        choose_model.clicked.connect(self._choose_model_input)
        layout.addWidget(choose_model, 0, 3)

        self.detect_pcap_edit = QtWidgets.QLineEdit()
        self.detect_pcap_edit.setPlaceholderText("选择要检测的 PCAP/PCAPNG…")
        layout.addWidget(self.detect_pcap_edit, 1, 0, 1, 3)

        choose_detect_pcap = QtWidgets.QPushButton("选择文件…")
        choose_detect_pcap.clicked.connect(self._choose_detect_pcap)
        layout.addWidget(choose_detect_pcap, 1, 3)

        self.detect_btn = QtWidgets.QPushButton("开始检测")
        self.detect_btn.clicked.connect(self._start_detection)
        layout.addWidget(self.detect_btn, 0, 4, 2, 1)

        self.detect_status = QtWidgets.QLabel()
        self.detect_status.setWordWrap(True)
        layout.addWidget(self.detect_status, 2, 0, 1, 5)

        return group

    def _build_log_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("运行日志")
        layout = QtWidgets.QVBoxLayout(group)
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return group

    # ------------------------------------------------------------------
    # Helpers
    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")

    def _update_jsonl_inputs(self) -> None:
        if self._jsonl_inputs:
            display = "; ".join(str(path) for path in self._jsonl_inputs)
        else:
            display = ""
        self.jsonl_inputs_edit.setText(display)

    def _default_timestamped_path(self, directory: Path, stem: str, suffix: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return directory / f"{stem}_{ts}{suffix}"

    # ------------------------------------------------------------------
    # Extraction handlers
    def _choose_pcap_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 PCAP/PCAPNG 文件",
            str(self._base_dir),
            "PCAP 文件 (*.pcap *.pcapng);;所有文件 (*)",
        )
        if path:
            self.pcap_path_edit.setText(path)
            jsonl_path = self._default_timestamped_path(self._jsonl_dir, Path(path).stem, ".jsonl")
            self.jsonl_output_edit.setText(str(jsonl_path))

    def _choose_pcap_directory(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 PCAP 目录", str(self._base_dir))
        if path:
            self.pcap_path_edit.setText(path)
            jsonl_path = self._default_timestamped_path(self._jsonl_dir, Path(path).name or "pcaps", ".jsonl")
            self.jsonl_output_edit.setText(str(jsonl_path))

    def _choose_jsonl_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存 JSONL",
            self.jsonl_output_edit.text() or str(self._jsonl_dir / "features.jsonl"),
            "JSON Lines (*.jsonl);;所有文件 (*)",
        )
        if path:
            self.jsonl_output_edit.setText(path)

    def _start_extraction(self) -> None:
        source_text = self.pcap_path_edit.text().strip()
        if not source_text:
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请先选择 PCAP 文件或目录。")
            return
        output_path = Path(self.jsonl_output_edit.text().strip() or self._default_timestamped_path(self._jsonl_dir, "features", ".jsonl"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        source_path = Path(source_text)
        if not source_path.exists():
            QtWidgets.QMessageBox.critical(self, "路径不存在", f"未找到路径：{source_path}")
            return

        self.extract_btn.setEnabled(False)
        self.extract_progress.setValue(0)
        self.extract_status.setText("正在提取特征…")
        self._append_log(f"开始提取：{source_path} → {output_path}")

        self._extract_worker = ExtractWorker(source_path, output_path)
        self._extract_worker.progress.connect(self.extract_progress.setValue)
        self._extract_worker.message.connect(lambda msg: self.extract_status.setText(msg))
        self._extract_worker.finished.connect(self._on_extraction_finished)
        self._extract_worker.error.connect(self._on_extraction_error)
        self._extract_worker.start()

    def _on_extraction_finished(self, summary) -> None:
        self.extract_btn.setEnabled(True)
        self.extract_progress.setValue(100)
        self.extract_status.setText("提取完成。")
        self._extract_worker = None

        path = Path(getattr(summary, "path", self.jsonl_output_edit.text()))
        self._append_log(
            "提取完成：{} 条记录，成功 {}，输出 {}".format(
                getattr(summary, "record_count", "?"),
                getattr(summary, "success_count", "?"),
                path,
            )
        )
        if path.exists():
            self._jsonl_inputs.append(path)
            self._update_jsonl_inputs()

    def _on_extraction_error(self, message: str) -> None:
        self.extract_btn.setEnabled(True)
        self.extract_progress.setValue(0)
        self.extract_status.setText("提取失败。")
        self._append_log(f"[错误] 特征提取失败：{message}")
        QtWidgets.QMessageBox.critical(self, "特征提取失败", message)
        self._extract_worker = None

    # ------------------------------------------------------------------
    # Vectorization handlers
    def _choose_jsonl_inputs(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "选择 JSONL 文件",
            self.jsonl_inputs_edit.text() or str(self._jsonl_dir),
            "JSON Lines (*.jsonl);;所有文件 (*)",
        )
        if paths:
            self._jsonl_inputs = [Path(p) for p in paths]
            self._update_jsonl_inputs()

    def _clear_jsonl_inputs(self) -> None:
        self._jsonl_inputs = []
        self._update_jsonl_inputs()

    def _choose_csv_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存 CSV",
            self.csv_output_edit.text() or str(self._csv_dir / "dataset.csv"),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if path:
            self.csv_output_edit.setText(path)

    def _start_vectorization(self) -> None:
        if not self._jsonl_inputs:
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请至少选择一个 JSONL 文件。")
            return
        output_path = Path(self.csv_output_edit.text().strip() or self._csv_dir / "dataset.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.vectorize_btn.setEnabled(False)
        self.vectorize_progress.setValue(10)
        self.vectorize_status.setText("正在写入 CSV…")
        self._append_log(f"开始向量化：{len(self._jsonl_inputs)} 个 JSONL → {output_path}")

        self._vectorize_worker = VectorizeWorker(self._jsonl_inputs, output_path)
        self._vectorize_worker.finished.connect(self._on_vectorize_finished)
        self._vectorize_worker.error.connect(self._on_vectorize_error)
        self._vectorize_worker.start()

    def _on_vectorize_finished(self, summary) -> None:
        self.vectorize_btn.setEnabled(True)
        self.vectorize_progress.setValue(100)
        self.vectorize_status.setText("向量化完成。")
        self._vectorize_worker = None

        path = Path(getattr(summary, "path", self.csv_output_edit.text()))
        self._append_log(
            "向量化完成：{} 条流量，{} 列，输出 {}".format(
                getattr(summary, "flow_count", "?"),
                getattr(summary, "column_count", "?"),
                path,
            )
        )
        self.dataset_edit.setText(str(path))

    def _on_vectorize_error(self, message: str) -> None:
        self.vectorize_btn.setEnabled(True)
        self.vectorize_progress.setValue(0)
        self.vectorize_status.setText("向量化失败。")
        self._append_log(f"[错误] 向量化失败：{message}")
        QtWidgets.QMessageBox.critical(self, "向量化失败", message)
        self._vectorize_worker = None

    # ------------------------------------------------------------------
    # Training handlers
    def _choose_dataset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 CSV 数据集",
            self.dataset_edit.text() or str(self._csv_dir),
            "CSV 文件 (*.csv);;所有文件 (*)",
        )
        if path:
            self.dataset_edit.setText(path)

    def _choose_model_output(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存模型",
            self.model_output_edit.text() or str(self._model_dir / "model.txt"),
            "模型文件 (*.txt);;所有文件 (*)",
        )
        if path:
            self.model_output_edit.setText(path)

    def _start_training(self) -> None:
        dataset = self.dataset_edit.text().strip()
        if not dataset:
            QtWidgets.QMessageBox.warning(self, "缺少数据集", "请先选择 CSV 数据集。")
            return
        model_path = Path(self.model_output_edit.text().strip() or self._model_dir / "model.txt")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_path = Path(dataset)
        if not dataset_path.exists():
            QtWidgets.QMessageBox.critical(self, "文件不存在", f"未找到数据集：{dataset_path}")
            return

        iterations = self.iterations_spin.value()

        self.train_btn.setEnabled(False)
        self.train_progress.setValue(5)
        self.train_status.setText("正在训练模型…")
        self._append_log(f"开始训练：{dataset_path} → {model_path} (max_iter={iterations})")

        self._train_worker = TrainWorker(dataset_path, model_path, iterations)
        self._train_worker.progress.connect(self.train_progress.setValue)
        self._train_worker.finished.connect(self._on_training_finished)
        self._train_worker.error.connect(self._on_training_error)
        self._train_worker.start()

    def _on_training_finished(self, summary: TrainingSummary) -> None:
        self.train_btn.setEnabled(True)
        self.train_progress.setValue(100)
        self.train_status.setText("训练完成。")
        self._train_worker = None

        self.model_input_edit.setText(str(summary.model_path))
        classes = getattr(summary, "classes", [])
        mapping = getattr(summary, "label_mapping", None)
        dropped = getattr(summary, "dropped_flows", 0)

        lines = [
            f"训练完成：{summary.flow_count} 条流量，{len(summary.feature_names)} 个特征。",
            f"类别：{classes}",
        ]
        if mapping:
            lines.append(f"标签映射：{mapping}")
        if dropped:
            lines.append(f"忽略未标注样本：{dropped}")

        for line in lines:
            self._append_log(line)

    def _on_training_error(self, message: str) -> None:
        self.train_btn.setEnabled(True)
        self.train_progress.setValue(0)
        self.train_status.setText("训练失败。")
        self._append_log(f"[错误] 模型训练失败：{message}")
        QtWidgets.QMessageBox.critical(self, "训练失败", message)
        self._train_worker = None

    # ------------------------------------------------------------------
    # Detection handlers
    def _choose_model_input(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            self.model_input_edit.text() or str(self._model_dir),
            "模型文件 (*.txt);;所有文件 (*)",
        )
        if path:
            self.model_input_edit.setText(path)

    def _choose_detect_pcap(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择待检测的 PCAP/PCAPNG",
            self.detect_pcap_edit.text() or str(self._base_dir),
            "PCAP 文件 (*.pcap *.pcapng);;所有文件 (*)",
        )
        if path:
            self.detect_pcap_edit.setText(path)

    def _start_detection(self) -> None:
        model = self.model_input_edit.text().strip()
        pcap = self.detect_pcap_edit.text().strip()
        if not model or not pcap:
            QtWidgets.QMessageBox.warning(self, "缺少输入", "请同时选择模型和 PCAP 文件。")
            return

        model_path = Path(model)
        pcap_path = Path(pcap)
        if not model_path.exists():
            QtWidgets.QMessageBox.critical(self, "模型不存在", f"未找到模型文件：{model_path}")
            return
        if not pcap_path.exists():
            QtWidgets.QMessageBox.critical(self, "文件不存在", f"未找到 PCAP 文件：{pcap_path}")
            return

        self.detect_btn.setEnabled(False)
        self.detect_status.setText("正在检测…")
        self._append_log(f"开始检测：{pcap_path} 使用模型 {model_path}")

        self._detect_worker = DetectWorker(model_path, pcap_path)
        self._detect_worker.finished.connect(self._on_detection_finished)
        self._detect_worker.error.connect(self._on_detection_error)
        self._detect_worker.start()

    def _on_detection_finished(self, result: DetectionResult) -> None:
        self.detect_btn.setEnabled(True)
        self.detect_status.setText("检测完成。")
        self._detect_worker = None

        if not result.success:
            message = result.error or "未知错误"
            self._append_log(f"[错误] 检测失败：{message}")
            QtWidgets.QMessageBox.critical(self, "检测失败", message)
            return

        suspicious = sum(1 for value in result.predictions if value != 0)
        self._append_log(
            "检测完成：共 {} 条流量，疑似恶意 {} 条，最高分 {:.3f}".format(
                result.flow_count,
                suspicious,
                max(result.scores) if result.scores else 0.0,
            )
        )
        if result.prediction_labels:
            preview = ", ".join(result.prediction_labels[:5])
            self._append_log(f"预测标签示例：{preview}")

    def _on_detection_error(self, message: str) -> None:
        self.detect_btn.setEnabled(True)
        self.detect_status.setText("检测失败。")
        self._append_log(f"[错误] 检测失败：{message}")
        QtWidgets.QMessageBox.critical(self, "检测失败", message)
        self._detect_worker = None


def run() -> None:
    """Convenience entry point for launching the GUI."""

    import sys

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


__all__ = ["MainWindow", "run"]
