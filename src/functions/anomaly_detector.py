"""自适应异常检测集成器。

该模块提供 ``EnsembleAnomalyDetector``，通过多模型投票和分位数阈值
自动校准，实现比单一 IsolationForest 更稳定的检测表现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def _normalize_scores(arr: np.ndarray) -> np.ndarray:
    """将不同算法的得分缩放到统一尺度。"""

    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=float)

    safe = arr[finite_mask]
    mean = float(np.mean(safe))
    std = float(np.std(safe))
    if std <= 1e-9:
        std = max(1e-9, float(np.max(np.abs(safe)) or 1.0))

    normed = np.zeros_like(arr, dtype=float)
    normed[finite_mask] = (safe - mean) / std
    normed[~finite_mask] = 0.0
    return normed


@dataclass
class DetectorInfo:
    name: str
    estimator: OutlierMixin
    weight: float


class EnsembleAnomalyDetector(BaseEstimator, OutlierMixin):
    """多模型投票的异常检测器。

    - IsolationForest 捕获全局稀疏异常
    - LocalOutlierFactor(novelty=True) 捕获局部密度异常
    - OneClassSVM 对复杂边界更敏感

    通过归一化后的得分求平均，依据 ``contamination`` 自动选取阈值。
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_estimators: int = 200,
        n_neighbors: int = 35,
        svm_gamma: str = "scale",
        random_state: int = 42,
    ) -> None:
        self.contamination = float(max(1e-4, min(0.49, contamination)))
        self.n_estimators = int(max(32, n_estimators))
        self.n_neighbors = int(max(5, n_neighbors))
        self.svm_gamma = svm_gamma
        self.random_state = random_state

        self.detectors_: Dict[str, DetectorInfo] = {}
        self.threshold_: Optional[float] = None
        self.offset_: Optional[float] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.fit_decision_scores_: Optional[np.ndarray] = None
        self.fit_raw_scores_: Dict[str, np.ndarray] = {}
        self.fit_votes_: Dict[str, np.ndarray] = {}

    # sklearn API -----------------------------------------------------
    def fit(self, X: np.ndarray, y=None):  # noqa: D401  (sklearn 兼容签名)
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须为二维数组")

        n_samples = X.shape[0]
        if n_samples < 10:
            raise ValueError("样本量过少，至少需要 10 条记录")

        self.feature_names_in_ = None
        self.fit_raw_scores_.clear()
        self.fit_votes_.clear()

        estimators: Iterable[DetectorInfo] = [
            DetectorInfo(
                "iforest",
                IsolationForest(
                    n_estimators=self.n_estimators,
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1,
                    warm_start=True,
                ),
                weight=1.0,
            ),
            DetectorInfo(
                "lof",
                LocalOutlierFactor(
                    n_neighbors=min(self.n_neighbors, max(5, n_samples - 1)),
                    contamination=self.contamination,
                    novelty=True,
                    metric="minkowski",
                ),
                weight=0.9,
            ),
            DetectorInfo(
                "ocsvm",
                OneClassSVM(
                    kernel="rbf",
                    gamma=self.svm_gamma,
                    nu=self.contamination,
                ),
                weight=0.8,
            ),
        ]

        decision_stack = []
        weight_stack = []

        # 对于样本非常多的情况，OneClassSVM 会较慢，采样训练提升速度
        max_svm_samples = 50000
        if n_samples > max_svm_samples:
            idx = np.random.default_rng(self.random_state).choice(
                n_samples, size=max_svm_samples, replace=False
            )
            svm_subset = X[idx]
        else:
            svm_subset = X

        for info in estimators:
            estimator = info.estimator
            if isinstance(estimator, OneClassSVM) and svm_subset is not X:
                estimator.fit(svm_subset)
            else:
                estimator.fit(X)

            if hasattr(estimator, "decision_function"):
                dec = estimator.decision_function(X)
            elif hasattr(estimator, "score_samples"):
                dec = estimator.score_samples(X)
            else:
                # 退化情况：仅返回 predict 结果
                dec = estimator.predict(X)

            decision_stack.append(_normalize_scores(dec))
            weight_stack.append(float(info.weight))
            self.fit_raw_scores_[info.name] = np.asarray(dec, dtype=float)
            if hasattr(estimator, "predict"):
                try:
                    pred = estimator.predict(X)
                except Exception:
                    pred = np.where(np.asarray(dec, dtype=float) <= 0, -1, 1)
            else:
                pred = np.where(np.asarray(dec, dtype=float) <= 0, -1, 1)
            self.fit_votes_[info.name] = np.asarray(pred, dtype=int)
            self.detectors_[info.name] = info

        stacked = np.vstack(decision_stack)
        weights = np.asarray(weight_stack, dtype=float)
        weights = weights / weights.sum()
        combined = np.average(stacked, axis=0, weights=weights)

        self.fit_decision_scores_ = combined.astype(float)
        self.threshold_ = float(np.quantile(combined, self.contamination))
        self.offset_ = float(np.mean(combined))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self._compute_decision_scores(X)
        if self.threshold_ is None:
            raise RuntimeError("模型尚未拟合")
        return scores - self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = self._compute_decision_scores(X)
        return scores.astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X)
        if self.threshold_ is None:
            raise RuntimeError("模型尚未拟合")
        return np.where(scores <= self.threshold_, -1, 1)

    # 内部方法 ---------------------------------------------------------
    def _compute_decision_scores(self, X: np.ndarray) -> np.ndarray:
        if not self.detectors_:
            raise RuntimeError("模型尚未拟合")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须为二维数组")

        decision_stack = []
        weights = []
        for name, info in self.detectors_.items():
            estimator = info.estimator
            if hasattr(estimator, "decision_function"):
                dec = estimator.decision_function(X)
            elif hasattr(estimator, "score_samples"):
                dec = estimator.score_samples(X)
            else:
                dec = estimator.predict(X)
            if name in self.fit_raw_scores_:
                # 使用训练阶段的统计量进行标准化
                ref = _normalize_scores(self.fit_raw_scores_[name])
                ref_mean = float(np.mean(ref))
                ref_std = float(np.std(ref) or 1.0)
            else:
                ref_mean = 0.0
                ref_std = 1.0
            normalized = (np.asarray(dec, dtype=float) - ref_mean) / ref_std
            decision_stack.append(normalized)
            weights.append(float(info.weight))

        stacked = np.vstack(decision_stack)
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        combined = np.average(stacked, axis=0, weights=weights_arr)
        return combined.astype(float)


__all__ = ["EnsembleAnomalyDetector"]

