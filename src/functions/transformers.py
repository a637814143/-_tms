"""Simple preprocessing utilities to align flow features across train/infer stages."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
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


class FeatureWeighter(BaseEstimator, TransformerMixin):
    """Apply per-feature weights to attenuate low-importance columns."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = {str(k): float(v) for k, v in (weights or {}).items()}
        self._columns_: List[str] = []

    def set_weights(self, weights: Dict[str, float]) -> None:
        self.weights = {str(k): float(v) for k, v in weights.items()}

    def fit(self, X, y: Optional[object] = None):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureWeighter 仅支持 pandas DataFrame 输入。")
        self._columns_ = list(X.columns)
        return self

    def transform(self, X):  # noqa: D401
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureWeighter 仅支持 pandas DataFrame 输入。")
        if not self.weights:
            return X.copy()
        Xc = X.copy()
        for col, weight in self.weights.items():
            if col in Xc.columns:
                try:
                    Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0) * float(weight)
                except Exception:
                    Xc[col] = Xc[col]
        return Xc

    def to_metadata(self) -> Dict[str, float]:
        return dict(self.weights)


class DeepFeatureExtractor(BaseEstimator, TransformerMixin):
    """Lightweight autoencoder-style projector with reconstruction error output."""

    def __init__(
        self,
        latent_dim: int = 32,
        *,
        max_epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-2,
        max_samples: int = 20000,
        random_state: int = 42,
    ) -> None:
        self.latent_dim = int(max(2, latent_dim))
        self.max_epochs = int(max(1, max_epochs))
        self.batch_size = int(max(8, batch_size))
        self.learning_rate = float(max(1e-5, learning_rate))
        self.max_samples = int(max(500, max_samples))
        self.random_state = int(random_state)

        self.input_dim_: Optional[int] = None
        self.latent_dim_: Optional[int] = None
        self.training_loss_: Optional[float] = None
        self.encoder_weights_: Optional[np.ndarray] = None
        self.encoder_bias_: Optional[np.ndarray] = None
        self.decoder_weights_: Optional[np.ndarray] = None
        self.decoder_bias_: Optional[np.ndarray] = None
        self.latent_slice_: Optional[slice] = None
        self.error_index_: Optional[int] = None

    def _init_parameters(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.random_state)
        latent = max(1, min(self.latent_dim, max(2, input_dim // 2)))
        latent = min(latent, input_dim)
        limit = 1.0 / max(1.0, np.sqrt(input_dim))
        self.encoder_weights_ = rng.uniform(-limit, limit, size=(input_dim, latent))
        self.encoder_bias_ = np.zeros(latent, dtype=float)
        self.decoder_weights_ = rng.uniform(-limit, limit, size=(latent, input_dim))
        self.decoder_bias_ = np.zeros(input_dim, dtype=float)
        self.input_dim_ = input_dim
        self.latent_dim_ = latent
        self.latent_slice_ = slice(input_dim, input_dim + latent)
        self.error_index_ = input_dim + latent

    def _encode(self, X: np.ndarray) -> np.ndarray:
        hidden = X @ self.encoder_weights_ + self.encoder_bias_
        return np.tanh(hidden)

    def _decode(self, H: np.ndarray) -> np.ndarray:
        return H @ self.decoder_weights_ + self.decoder_bias_

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        latent = self._encode(X)
        recon = self._decode(latent)
        return latent, recon

    def fit(self, X, y: Optional[object] = None):  # noqa: D401
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("DeepFeatureExtractor 需要二维输入矩阵。")

        n_samples, n_features = arr.shape
        self._init_parameters(n_features)

        rng = np.random.default_rng(self.random_state)
        if n_samples > self.max_samples:
            idx = rng.choice(n_samples, size=self.max_samples, replace=False)
            train_arr = arr[idx]
        else:
            train_arr = arr

        best_loss = np.inf
        patience = 3
        stalled = 0

        for _epoch in range(self.max_epochs):
            idx = rng.permutation(len(train_arr))
            shuffled = train_arr[idx]
            epoch_loss = 0.0
            steps = 0
            for start in range(0, len(shuffled), self.batch_size):
                end = start + self.batch_size
                batch = shuffled[start:end]
                if batch.size == 0:
                    continue
                steps += 1
                latent, recon = self._forward(batch)
                error = recon - batch
                loss = float(np.mean((error) ** 2))
                epoch_loss += loss

                grad_recon = error / max(1, batch.shape[0])
                grad_W2 = latent.T @ grad_recon
                grad_b2 = grad_recon.sum(axis=0)
                grad_hidden = grad_recon @ self.decoder_weights_.T
                grad_hidden *= (1.0 - latent**2)
                grad_W1 = batch.T @ grad_hidden
                grad_b1 = grad_hidden.sum(axis=0)

                self.decoder_weights_ -= self.learning_rate * grad_W2
                self.decoder_bias_ -= self.learning_rate * grad_b2
                self.encoder_weights_ -= self.learning_rate * grad_W1
                self.encoder_bias_ -= self.learning_rate * grad_b1

            epoch_loss /= max(1, steps)
            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                stalled = 0
            else:
                stalled += 1
                if stalled >= patience:
                    break

        self.training_loss_ = float(best_loss if np.isfinite(best_loss) else 0.0)
        return self

    def transform(self, X):  # noqa: D401
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("DeepFeatureExtractor 需要二维输入矩阵。")
        if self.encoder_weights_ is None or self.decoder_weights_ is None:
            raise RuntimeError("DeepFeatureExtractor 尚未拟合。")

        latent, recon = self._forward(arr)
        error = np.mean((recon - arr) ** 2, axis=1)
        error = error.reshape(-1, 1)
        output = np.hstack([arr, latent, error])
        self.last_reconstruction_error_ = error.ravel()
        return output

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"f{i}" for i in range(self.input_dim_ or 0)]
        base = list(input_features)
        latent_dim = self.latent_dim_ or 0
        base.extend([f"deep_latent_{i}" for i in range(latent_dim)])
        base.append("deep_recon_error")
        return np.array(base, dtype=object)
