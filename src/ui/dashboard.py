# -*- coding: utf-8 -*-
"""结果展示相关的小部件。"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


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
        self.timeline_label.setStyleSheet(
            "QLabel { border:1px solid #E6E9EF; border-radius:6px; background:#FFFFFF; }"
        )
        self.timeline_label.setScaledContents(True)
        layout.addWidget(self.timeline_label)

        self._timeline_path: Optional[str] = None

    def update_metrics(self, analysis: Optional[Dict], metadata: Optional[Dict]) -> None:
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


__all__ = ["ResultsDashboard"]
