# -*- coding: utf-8 -*-
"""界面用到的模型与委托，减轻 ``ui_main`` 的体积。"""

from __future__ import annotations

from PyQt5 import QtCore

try:  # pragma: no cover - pandas 在部分环境可能不可用
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - fallback
    pd = None  # type: ignore


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


__all__ = ["PandasFrameModel"]
