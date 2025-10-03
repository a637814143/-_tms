"""自适应异常检测集成器。

该模块提供 ``EnsembleAnomalyDetector``，通过多模型投票和分位数阈值
自动校准，实现比单一 IsolationForest 更稳定的检测表现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

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
        self.vote_threshold_: Optional[float] = None
        self.offset_: Optional[float] = None
        self.feature_names_in_: Optional[np.ndarray] = None
        self.fit_decision_scores_: Optional[np.ndarray] = None
        self.fit_vote_ratios_: Optional[np.ndarray] = None
        self.fit_raw_scores_: Dict[str, np.ndarray] = {}
        self.fit_votes_: Dict[str, np.ndarray] = {}
        self.last_combined_scores_: Optional[np.ndarray] = None
        self.last_vote_ratio_: Optional[np.ndarray] = None

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
        self.last_combined_scores_ = None
        self.last_vote_ratio_ = None

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
        vote_stack = []
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

            normalized = _normalize_scores(dec)
            decision_stack.append(normalized)
            weight_stack.append(float(info.weight))
            raw_scores = np.asarray(dec, dtype=float)
            self.fit_raw_scores_[info.name] = raw_scores
            if hasattr(estimator, "predict"):
                try:
                    pred = estimator.predict(X)
                except Exception:
                    pred = np.where(raw_scores <= 0, -1, 1)
            else:
                pred = np.where(raw_scores <= 0, -1, 1)
            votes = np.where(np.asarray(pred, dtype=int) == -1, 1.0, 0.0)
            vote_stack.append(votes)
            self.fit_votes_[info.name] = np.asarray(pred, dtype=int)
            self.detectors_[info.name] = info

        stacked = np.vstack(decision_stack)
        weights = np.asarray(weight_stack, dtype=float)
        weights = weights / weights.sum()
        combined = np.average(stacked, axis=0, weights=weights)

        vote_matrix = np.vstack(vote_stack) if vote_stack else np.zeros((0, n_samples))
        if vote_matrix.size:
            vote_ratio = np.clip(vote_matrix.mean(axis=0), 0.0, 1.0)
        else:
            vote_ratio = np.zeros(n_samples, dtype=float)

        self.fit_decision_scores_ = combined.astype(float)
        self.fit_vote_ratios_ = vote_ratio.astype(float)
        self.last_combined_scores_ = self.fit_decision_scores_.copy()
        self.last_vote_ratio_ = self.fit_vote_ratios_.copy()

        self.threshold_ = float(np.quantile(combined, self.contamination))
        self.offset_ = float(np.mean(combined))

        suspect_count = max(1, int(np.ceil(self.contamination * n_samples)))
        ranked_idx = np.argsort(combined)[:suspect_count]
        if ranked_idx.size and vote_ratio.size:
            candidates = vote_ratio[ranked_idx]
            self.vote_threshold_ = float(np.clip(np.quantile(candidates, 0.25), 0.1, 1.0))
        else:
            self.vote_threshold_ = float(np.clip(np.mean(vote_ratio) if vote_ratio.size else 0.5, 0.1, 1.0))

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores, _ = self._compute_decision_details(X)
        if self.threshold_ is None:
            raise RuntimeError("模型尚未拟合")
        return scores - self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores, _ = self._compute_decision_details(X)
        return scores.astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores, vote_ratio = self._compute_decision_details(X)
        if self.threshold_ is None:
            raise RuntimeError("模型尚未拟合")
        threshold = float(self.threshold_)
        vote_threshold = float(self.vote_threshold_ if self.vote_threshold_ is not None else 0.5)
        anomalies = (scores <= threshold) & (vote_ratio >= vote_threshold)
        if not np.any(anomalies) and len(scores):
            top_k = max(1, int(np.ceil(self.contamination * len(scores))))
            idx = np.argsort(scores)[:top_k]
            anomalies = np.zeros(len(scores), dtype=bool)
            anomalies[idx] = True
        return np.where(anomalies, -1, 1)

    # 内部方法 ---------------------------------------------------------
    def _compute_decision_details(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.detectors_:
            raise RuntimeError("模型尚未拟合")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须为二维数组")

        decision_stack = []
        vote_stack = []
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
                ref = _normalize_scores(self.fit_raw_scores_[name])
                ref_mean = float(np.mean(ref))
                ref_std = float(np.std(ref) or 1.0)
            else:
                ref_mean = 0.0
                ref_std = 1.0
            arr = np.asarray(dec, dtype=float)
            normalized = (arr - ref_mean) / ref_std
            decision_stack.append(normalized)
            weights.append(float(info.weight))
            if hasattr(estimator, "predict"):
                try:
                    pred = estimator.predict(X)
                except Exception:
                    pred = np.where(arr <= 0, -1, 1)
            else:
                pred = np.where(arr <= 0, -1, 1)
            vote_stack.append(np.where(np.asarray(pred, dtype=int) == -1, 1.0, 0.0))

        stacked = np.vstack(decision_stack)
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
        combined = np.average(stacked, axis=0, weights=weights_arr)

        if vote_stack:
            vote_ratio = np.clip(np.mean(np.vstack(vote_stack), axis=0), 0.0, 1.0)
        else:
            vote_ratio = np.zeros(combined.shape[0], dtype=float)

        self.last_combined_scores_ = combined.astype(float)
        self.last_vote_ratio_ = vote_ratio.astype(float)

        return self.last_combined_scores_, self.last_vote_ratio_


__all__ = ["EnsembleAnomalyDetector"]

