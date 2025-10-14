"""Simple preprocessing utilities to align flow features across train/infer stages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureAligner(BaseEstimator, TransformerMixin):
    """Ensure a deterministic column order and fill missing columns."""

    def __init__(self, feature_order: Iterable[str], fill_value: float = 0.0) -> None:
        self.feature_order = list(feature_order)
        self.fill_value = float(fill_value)

    def fit(self, X, y: Optional[object] = None):  # noqa: D401 - sklearn signature
        return self

    def transform(self, X) -> pd.DataFrame:  # noqa: D401 - sklearn signature
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureAligner 仅支持 pandas DataFrame 输入。")
        Xc = X.copy()
        for col in self.feature_order:
            if col not in Xc.columns:
                Xc[col] = self.fill_value
        return Xc[self.feature_order].copy()

    # metadata helpers -------------------------------------------------
    def to_metadata(self) -> Dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "fill_value": float(self.fill_value),
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "FeatureAligner":
        feature_order = payload.get("feature_order") or []
        fill_value = payload.get("fill_value", 0.0)
        return cls(feature_order=feature_order, fill_value=float(fill_value))


class SimpleImputerByDtype(BaseEstimator, TransformerMixin):
    """Fill numeric columns with median and categorical with mode/empty string."""

    def __init__(self) -> None:
        self.num_median_: Dict[str, float] = {}
        self.cat_mode_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[object] = None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SimpleImputerByDtype 仅支持 pandas DataFrame 输入。")
        self.num_median_.clear()
        self.cat_mode_.clear()
        for col in X.columns:
            series = X[col]
            if is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                self.num_median_[col] = float(numeric.median()) if not numeric.dropna().empty else 0.0
            else:
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
        inst = cls()
        inst.num_median_ = {k: float(v) for k, v in (payload.get("num_median") or {}).items()}
        inst.cat_mode_ = dict(payload.get("cat_mode") or {})
        return inst


class PreprocessPipeline(BaseEstimator, TransformerMixin):
    """Compose FeatureAligner and SimpleImputer for reuse between train/infer."""

    def __init__(self, feature_order: Iterable[str], fill_value: float = 0.0) -> None:
        self.aligner = FeatureAligner(feature_order, fill_value)
        self.imputer = SimpleImputerByDtype()

    def fit(self, X, y: Optional[object] = None):  # noqa: D401
        aligned = self.aligner.transform(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.aligner.feature_order))
        self.imputer.fit(aligned)
        return self

    def transform(self, X):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PreprocessPipeline 仅支持 pandas DataFrame 输入。")
        aligned = self.aligner.transform(X)
        return self.imputer.transform(aligned)

    # metadata helpers -------------------------------------------------
    def to_metadata(self) -> Dict[str, object]:
        return {
            "feature_order": list(self.aligner.feature_order),
            "fill_value": float(self.aligner.fill_value),
            "imputer": self.imputer.to_metadata(),
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "PreprocessPipeline":
        feature_order = payload.get("feature_order") or []
        fill_value = payload.get("fill_value", 0.0)
        inst = cls(feature_order=feature_order, fill_value=float(fill_value))
        imputer_meta = payload.get("imputer") or {}
        inst.imputer = SimpleImputerByDtype.from_metadata(imputer_meta)
        return inst
