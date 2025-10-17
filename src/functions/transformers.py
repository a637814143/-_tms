"""Simple preprocessing utilities to align flow features across train/infer stages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin


def _maybe_float(value: object) -> object:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class FeatureAligner(BaseEstimator, TransformerMixin):
    """Ensure a deterministic column order and fill missing columns."""

    def __init__(
        self,
        feature_order: Iterable[str],
        fill_value: float = 0.0,
        column_fill_values: Optional[Dict[str, object]] = None,
    ) -> None:
        self.feature_order = list(feature_order)
        self.fill_value = float(fill_value)
        self.column_fill_values: Dict[str, object] = {
            str(k): v for k, v in (column_fill_values or {}).items()
        }

    def fit(self, X, y: Optional[object] = None):  # noqa: D401 - sklearn signature
        return self

    def transform(self, X) -> pd.DataFrame:  # noqa: D401 - sklearn signature
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureAligner 仅支持 pandas DataFrame 输入。")
        Xc = X.copy()
        for col in self.feature_order:
            if col not in Xc.columns:
                fill = self.column_fill_values.get(col, self.fill_value)
                Xc[col] = fill
        return Xc[self.feature_order].copy()

    # metadata helpers -------------------------------------------------
    def to_metadata(self) -> Dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "fill_value": float(self.fill_value),
            "column_fill_values": self.column_fill_values,
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "FeatureAligner":
        feature_order = payload.get("feature_order") or []
        fill_value = payload.get("fill_value", 0.0)
        column_fill_values = payload.get("column_fill_values") or {}
        return cls(
            feature_order=feature_order,
            fill_value=float(fill_value),
            column_fill_values=dict(column_fill_values),
        )


class SimpleImputerByDtype(BaseEstimator, TransformerMixin):
    """Fill numeric columns with median and categorical with mode/empty string."""

    def __init__(
        self,
        preset_numeric: Optional[Dict[str, object]] = None,
        preset_categorical: Optional[Dict[str, object]] = None,
    ) -> None:
        self._preset_numeric = {
            str(k): _maybe_float(v) for k, v in (preset_numeric or {}).items()
        }
        self._preset_categorical = {
            str(k): v for k, v in (preset_categorical or {}).items()
        }
        self.num_median_: Dict[str, float] = {
            k: float(v)
            for k, v in self._preset_numeric.items()
            if isinstance(v, (int, float))
        }
        self.cat_mode_: Dict[str, Any] = dict(self._preset_categorical)

    def fit(self, X: pd.DataFrame, y: Optional[object] = None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimpleImputerByDtype 仅支持 pandas DataFrame 输入。")
        for col in X.columns:
            series = X[col]
            if is_numeric_dtype(series):
                if col in self._preset_numeric:
                    value = self._preset_numeric[col]
                    try:
                        self.num_median_[col] = float(value)
                        continue
                    except (TypeError, ValueError):
                        pass
                numeric = pd.to_numeric(series, errors="coerce")
                self.num_median_[col] = float(numeric.median()) if not numeric.dropna().empty else 0.0
            else:
                if col in self._preset_categorical:
                    self.cat_mode_[col] = self._preset_categorical[col]
                    continue
                mode_series = series.mode(dropna=True)
                if not mode_series.empty:
                    self.cat_mode_[col] = mode_series.iloc[0]
                else:
                    self.cat_mode_[col] = ""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimpleImputerByDtype 仅支持 pandas DataFrame 输入。")
        Xc = X.copy()
        for col in Xc.columns:
            if col in self.num_median_:
                numeric = pd.to_numeric(Xc[col], errors="coerce")
                Xc[col] = numeric.fillna(self.num_median_[col])
            elif col in self.cat_mode_:
                Xc[col] = Xc[col].fillna(self.cat_mode_[col])
        return Xc

    def to_metadata(self) -> Dict[str, object]:
        return {
            "num_median": {k: float(v) for k, v in self.num_median_.items()},
            "cat_mode": {k: ("" if v is None else v) for k, v in self.cat_mode_.items()},
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "SimpleImputerByDtype":
        num_meta = {k: _maybe_float(v) for k, v in (payload.get("num_median") or {}).items()}
        cat_meta = dict(payload.get("cat_mode") or {})
        inst = cls(preset_numeric=num_meta, preset_categorical=cat_meta)
        inst.num_median_ = {k: float(v) for k, v in num_meta.items() if isinstance(v, (int, float))}
        inst.cat_mode_ = dict(cat_meta)
        return inst


class PreprocessPipeline(BaseEstimator, TransformerMixin):
    """Compose FeatureAligner and SimpleImputer for reuse between train/infer."""

    def __init__(
        self,
        feature_order: Iterable[str],
        fill_value: float = 0.0,
        *,
        fill_values: Optional[Dict[str, object]] = None,
        categorical_maps: Optional[Dict[str, Dict[str, object]]] = None,
        aligner_in_pipeline: bool = False,
    ) -> None:
        self.feature_order = list(feature_order)
        self.fill_value = float(fill_value)
        self.fill_values = {
            str(k): _maybe_float(v) for k, v in (fill_values or {}).items()
        }
        self.categorical_maps = {
            str(k): dict(v) for k, v in (categorical_maps or {}).items()
        }
        categorical_fill = {
            key: -1 for key in self.categorical_maps.keys()
        }
        column_fill_values = dict(categorical_fill)
        column_fill_values.update(self.fill_values)

        self.column_fill_values = column_fill_values
        self.aligner_in_pipeline = bool(aligner_in_pipeline)
        self.aligner = FeatureAligner(
            self.feature_order,
            self.fill_value,
            column_fill_values=self.column_fill_values,
        )
        self.imputer = SimpleImputerByDtype(preset_numeric=self.fill_values)

    def fit(self, X, y: Optional[object] = None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocessPipeline 仅支持 pandas DataFrame 输入。")
        aligned = X if self.aligner_in_pipeline else self.aligner.transform(X)
        self.imputer.fit(aligned)
        return self

    def transform(self, X):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocessPipeline 仅支持 pandas DataFrame 输入。")
        aligned = X if self.aligner_in_pipeline else self.aligner.transform(X)
        return self.imputer.transform(aligned)

    # metadata helpers -------------------------------------------------
    def to_metadata(self) -> Dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "fill_value": float(self.fill_value),
            "fill_values": self.fill_values,
            "categorical_maps": self.categorical_maps,
            "column_fill_values": self.column_fill_values,
            "aligner_in_pipeline": self.aligner_in_pipeline,
            "input_columns": list(self.feature_order),
            "imputer": self.imputer.to_metadata(),
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "PreprocessPipeline":
        feature_order = payload.get("feature_order") or payload.get("input_columns") or []
        fill_value = payload.get("fill_value", 0.0)
        fill_values = payload.get("fill_values") or {}
        categorical_maps = payload.get("categorical_maps") or {}
        aligner_in_pipeline = bool(payload.get("aligner_in_pipeline", False))
        inst = cls(
            feature_order=feature_order,
            fill_value=float(fill_value),
            fill_values=fill_values,
            categorical_maps=categorical_maps,
            aligner_in_pipeline=aligner_in_pipeline,
        )
        imputer_meta = payload.get("imputer") or {}
        inst.imputer = SimpleImputerByDtype.from_metadata(imputer_meta)
        # 覆盖 aligner 的列填充值，确保与训练阶段保持一致
        column_fill_values = payload.get("column_fill_values") or {}
        if column_fill_values:
            inst.column_fill_values = dict(column_fill_values)
            inst.aligner.column_fill_values = dict(column_fill_values)
        return inst