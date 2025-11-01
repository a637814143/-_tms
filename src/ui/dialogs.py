# -*- coding: utf-8 -*-
"""主界面使用的各类对话框。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from PyQt5 import QtCore, QtWidgets

from src.configuration import load_config, project_root
from src.functions.annotations import upsert_annotation


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
        mid.addStretch(1)
        mid.addWidget(btn_add)
        mid.addWidget(btn_remove)
        mid.addSpacing(10)
        mid.addWidget(btn_up)
        mid.addWidget(btn_down)
        mid.addStretch(1)

        self.list_sel = QtWidgets.QListWidget()
        self.list_sel.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        body.addWidget(self.list_all, 5)
        body.addLayout(mid, 1)
        body.addWidget(self.list_sel, 5)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        lay.addWidget(btns)

        self.need_k = need_k
        btn_add.clicked.connect(self._add)
        btn_remove.clicked.connect(self._remove)
        btn_up.clicked.connect(lambda: self._move(-1))
        btn_down.clicked.connect(lambda: self._move(1))
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def _add(self) -> None:
        for it in self.list_all.selectedItems():
            self.list_sel.addItem(it.text())

    def _remove(self) -> None:
        for it in self.list_sel.selectedItems():
            self.list_sel.takeItem(self.list_sel.row(it))

    def _move(self, delta: int) -> None:
        rows = [self.list_sel.row(it) for it in self.list_sel.selectedItems()]
        if not rows:
            return
        row = rows[0]
        new_row = row + delta
        if 0 <= new_row < self.list_sel.count():
            item = self.list_sel.takeItem(row)
            self.list_sel.insertItem(new_row, item)
            self.list_sel.setCurrentRow(new_row)

    def selected_columns(self) -> List[str]:
        return [self.list_sel.item(i).text() for i in range(self.list_sel.count())]

    def accept(self) -> None:
        sel = self.selected_columns()
        if len(sel) != self.need_k:
            QtWidgets.QMessageBox.warning(
                self, "列数不一致", f"当前选择 {len(sel)} 列，需要 {self.need_k} 列。"
            )
            return
        super().accept()


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
        self.btn_mark_normal = btns.addButton(
            "标注为正常", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.btn_mark_anomaly = btns.addButton(
            "标注为异常", QtWidgets.QDialogButtonBox.ActionRole
        )
        btns.addButton(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(btns)

        btns.rejected.connect(self.reject)
        self.btn_mark_normal.clicked.connect(lambda: self._store_label(0.0))
        self.btn_mark_anomaly.clicked.connect(lambda: self._store_label(1.0))

    def _store_label(self, value: float) -> None:
        try:
            upsert_annotation(
                self._record, label=value, notes=self.notes_edit.text().strip() or None
            )
            QtWidgets.QMessageBox.information(self, "标注成功", "已保存人工标注。")
            self.annotation_saved.emit(value)
            self.accept()
        except Exception as exc:  # pragma: no cover - 界面展示
            QtWidgets.QMessageBox.critical(self, "保存失败", f"无法写入标注：{exc}")


class ConfigEditorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑全局配置 (YAML)")
        self.resize(720, 540)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.path = Path(project_root() / "config" / "default.yaml")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        layout.addWidget(QtWidgets.QLabel(f"配置文件：{self.path}"))
        self.editor = QtWidgets.QPlainTextEdit()
        layout.addWidget(self.editor, 1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
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


__all__ = [
    "FeaturePickDialog",
    "AnomalyDetailDialog",
    "ConfigEditorDialog",
]
