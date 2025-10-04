"""Reusable feature preprocessing transformers for flow anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_TOKEN = "<MISSING>"


def _ensure_dataframe(X: object, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, pd.Series):
        return X.to_frame().T.copy()
    if isinstance(X, np.ndarray):
        if columns is None:
            raise ValueError("NumPy 数组缺少列名，无法恢复原始特征顺序。")
        if X.ndim != 2:
            raise ValueError("输入数组必须是二维的")
        return pd.DataFrame(X, columns=list(columns))
    raise TypeError(f"不支持的输入类型: {type(X)!r}")


@dataclass
class _CategoricalSchema:
    source: str
    labels: List[str]
    missing_token: str = MISSING_TOKEN

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "labels": list(self.labels),
            "missing_token": self.missing_token,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "_CategoricalSchema":
        source = str(payload.get("source", ""))
        labels_raw = payload.get("labels") or []
        labels = [str(item) for item in labels_raw]
        missing = str(payload.get("missing_token", MISSING_TOKEN))
        return cls(source=source, labels=labels, missing_token=missing)


class FlowFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """统一的特征对齐与编码器，用于训练与推理阶段保持一致。"""

    def __init__(
        self,
        *,
        feature_columns: Optional[List[str]] = None,
        fill_values: Optional[Dict[str, float]] = None,
        categorical_maps: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.initial_fill_values = dict(fill_values) if fill_values else None
        self.initial_categorical_maps = (
            {key: dict(value) for key, value in categorical_maps.items()}
            if categorical_maps
            else None
        )

    def fit(self, X: object, y: Optional[object] = None):  # noqa: D401 - scikit-learn API
        df = _ensure_dataframe(X, self.feature_columns)
        if self.feature_columns is None:
            self.feature_columns_ = [str(col) for col in df.columns]
        else:
            self.feature_columns_ = [str(col) for col in self.feature_columns]
        missing = [col for col in self.feature_columns_ if col not in df.columns]
        if missing:
            raise ValueError(f"训练数据缺少特征列: {', '.join(missing)}")

        self.fill_values_: Dict[str, float] = {}
        if self.initial_fill_values:
            self.fill_values_.update({str(k): float(v) for k, v in self.initial_fill_values.items()})

        self.categorical_maps_: Dict[str, _CategoricalSchema] = {}
        if self.initial_categorical_maps:
            for key, payload in self.initial_categorical_maps.items():
                self.categorical_maps_[str(key)] = _CategoricalSchema.from_dict(payload)

        transformed_blocks: List[pd.Series] = []
        for column in self.feature_columns_:
            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                numeric = series.fillna(False).astype(bool).astype("int8")
                self.fill_values_.setdefault(column, 0.0)
                transformed_blocks.append(numeric.astype("float32"))
            elif pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                if column not in self.fill_values_:
                    non_na = numeric.dropna()
                    median = float(non_na.median()) if not non_na.empty else 0.0
                    if np.isnan(median) or np.isinf(median):
                        median = float(non_na.iloc[0]) if not non_na.empty else 0.0
                    self.fill_values_[column] = median
                transformed_blocks.append(numeric.astype("float32"))
            else:
                normalized = series.fillna(MISSING_TOKEN).astype(str).str.strip()
                normalized = normalized.replace("", MISSING_TOKEN)
                schema = self.categorical_maps_.get(column)
                if schema is None:
                    labels: List[str] = []
                    mapping: Dict[str, int] = {}
                    for value in normalized.drop_duplicates():
                        value_str = str(value)
                        if value_str not in mapping:
                            mapping[value_str] = len(labels)
                            labels.append(value_str)
                    schema = _CategoricalSchema(source=column, labels=labels)
                    self.categorical_maps_[column] = schema
                transformed_blocks.append(normalized.astype("string"))
                self.fill_values_.setdefault(column, -1.0)

        self.feature_names_in_ = np.asarray(self.feature_columns_, dtype=object)
        self.output_feature_names_ = list(self.feature_columns_)
        self._fitted = True
        return self

    def transform(self, X: object) -> np.ndarray:  # noqa: D401 - scikit-learn API
        if not getattr(self, "_fitted", False):
            raise RuntimeError("FlowFeaturePreprocessor 尚未 fit，无法 transform。")
        df = _ensure_dataframe(X, self.feature_columns_)
        row_count = len(df)
        transformed_cols: List[np.ndarray] = []

        for column in self.feature_columns_:
            if column not in df.columns:
                raise ValueError(f"输入数据缺少特征列: {column}")
            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                numeric = series.fillna(False).astype(bool).astype("int8")
                transformed_cols.append(numeric.to_numpy(dtype="float32", copy=False))
            elif pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                fill_value = self.fill_values_.get(column, 0.0)
                filled = numeric.fillna(fill_value).astype("float32")
                transformed_cols.append(filled.to_numpy(dtype="float32", copy=False))
            else:
                normalized = series.fillna(MISSING_TOKEN).astype(str).str.strip()
                normalized = normalized.replace("", MISSING_TOKEN)
                schema = self.categorical_maps_.get(column)
                if schema is None:
                    fill_value = int(self.fill_values_.get(column, -1.0))
                    transformed_cols.append(np.full(row_count, fill_value, dtype="float32"))
                    continue
                mapping = {label: idx for idx, label in enumerate(schema.labels)}
                codes = normalized.map(mapping).fillna(-1).astype("int32")
                transformed_cols.append(codes.to_numpy(dtype="float32", copy=False))
        if not transformed_cols:
            return np.empty((row_count, 0), dtype="float32")
        return np.column_stack(transformed_cols).astype("float32", copy=False)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("FlowFeaturePreprocessor 尚未 fit，无法获取列名。")
        return np.asarray(self.output_feature_names_, dtype=object)

    # 序列化辅助
    def to_metadata(self) -> Dict[str, object]:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("FlowFeaturePreprocessor 尚未 fit，无法导出 schema。")
        categorical = {
            col: schema.to_dict() for col, schema in self.categorical_maps_.items()
        }
        return {
            "input_columns": list(self.feature_columns_),
            "feature_columns": list(self.output_feature_names_),
            "fill_values": {k: float(v) for k, v in self.fill_values_.items()},
            "categorical_maps": categorical,
        }

    @classmethod
    def from_metadata(cls, payload: Dict[str, object]) -> "FlowFeaturePreprocessor":
        feature_columns = payload.get("input_columns") or payload.get("feature_columns")
        fill_values = payload.get("fill_values") or {}
        categorical_raw = payload.get("categorical_maps") or {}
        categorical_maps = {
            key: value for key, value in categorical_raw.items() if isinstance(value, dict)
        }
        return cls(
            feature_columns=list(feature_columns) if feature_columns else None,
            fill_values=fill_values,
            categorical_maps=categorical_maps,
        )