"""Training and inference helpers for tree-based PCAP flow classifiers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .feature_extractor import extract_pcap_features
from .vectorizer import (
    VectorizationResult,
    load_vectorized_dataset,
    numeric_feature_names,
    vectorize_flows,
)
from .csv_utils import read_csv_flexible
from .logging_utils import get_logger

try:  # pandas 在预测阶段可选
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - 预测模块可在无 pandas 环境运行
    pd = None  # type: ignore


logger = get_logger(__name__)

try:  # 规则引擎不是硬依赖
    from .risk_rules import (  # type: ignore
        DEFAULT_TRIGGER_THRESHOLD,
        DEFAULT_MODEL_WEIGHT,
        DEFAULT_RULE_WEIGHT,
        DEFAULT_FUSION_THRESHOLD,
        RULE_TRIGGER_THRESHOLD,
        fuse_model_rule_votes,
        get_fusion_settings,
        get_rule_settings,
        score_rules as apply_risk_rules,
    )
except Exception:  # pragma: no cover - 缺少依赖时退化
    apply_risk_rules = None  # type: ignore
    DEFAULT_TRIGGER_THRESHOLD = 40.0  # type: ignore
    DEFAULT_MODEL_WEIGHT = 0.3  # type: ignore
    DEFAULT_RULE_WEIGHT = 0.7  # type: ignore
    DEFAULT_FUSION_THRESHOLD = 0.2  # type: ignore
    RULE_TRIGGER_THRESHOLD = 25.0  # type: ignore

    def get_rule_settings(profile: Optional[str] = None) -> Dict[str, object]:  # type: ignore
        return {
            "params": {},
            "trigger_threshold": float(DEFAULT_TRIGGER_THRESHOLD),
            "model_weight": float(DEFAULT_MODEL_WEIGHT),
            "rule_weight": float(DEFAULT_RULE_WEIGHT),
            "fusion_threshold": float(DEFAULT_FUSION_THRESHOLD),
            "profile": profile,
        }

    def get_fusion_settings(profile: Optional[str] = None) -> Dict[str, float]:  # type: ignore
        return {
            "model_weight": float(DEFAULT_MODEL_WEIGHT),
            "rule_weight": float(DEFAULT_RULE_WEIGHT),
            "fusion_threshold": float(DEFAULT_FUSION_THRESHOLD),
            "profile": profile or "baseline",
        }

    def fuse_model_rule_votes(
        model_scores: Iterable[object],
        rule_scores: Optional[Iterable[object]],
        *,
        profile: Optional[str] = None,
        model_weight: Optional[float] = None,
        rule_weight: Optional[float] = None,
        threshold: Optional[float] = None,
        trigger_threshold: Optional[float] = None,
        model_confidence: Optional[Iterable[object]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model_arr = np.asarray(model_scores, dtype=np.float64).reshape(-1)
        if model_arr.size == 0:
            empty = np.zeros(0, dtype=np.float64)
            empty_bool = empty.astype(bool)
            return empty, empty_bool, empty_bool

        model_arr = np.clip(model_arr, 0.0, 1.0)

        if rule_scores is None:
            rule_arr = np.zeros_like(model_arr)
            normalized_rules = rule_arr
            active_rule_weight = float(rule_weight or 0.0)
        else:
            rule_arr = np.asarray(rule_scores, dtype=np.float64).reshape(-1)
            if rule_arr.size != model_arr.size:
                if rule_arr.size < model_arr.size:
                    padded = np.zeros_like(model_arr)
                    padded[: rule_arr.size] = rule_arr
                    rule_arr = padded
                else:
                    rule_arr = rule_arr[: model_arr.size]
            normalized_rules = np.clip(rule_arr / 100.0, 0.0, 1.0)
            active_rule_weight = float(rule_weight or 0.0)

        trig_threshold = float(trigger_threshold or DEFAULT_TRIGGER_THRESHOLD)
        rules_triggered = (rule_arr >= trig_threshold).astype(bool)

        if model_confidence is not None:
            conf_arr = np.asarray(model_confidence, dtype=np.float64).reshape(-1)
            if conf_arr.size == 0:
                conf_arr = np.zeros_like(model_arr)
            elif conf_arr.size == 1:
                conf_arr = np.full_like(model_arr, float(conf_arr[0]))
            elif conf_arr.size != model_arr.size:
                limit = min(conf_arr.size, model_arr.size)
                padded = np.empty_like(model_arr)
                padded[:limit] = conf_arr[:limit]
                if limit < model_arr.size:
                    padded[limit:] = conf_arr[limit - 1]
                conf_arr = padded
            conf_arr = np.clip(conf_arr, 0.0, 1.0)
        else:
            conf_arr = None

        model_w = float(model_weight or DEFAULT_MODEL_WEIGHT)
        rule_w = float(active_rule_weight)
        total = model_w + rule_w
        if not np.isfinite(total) or total <= 0.0:
            total = 1.0

        profile_key = str(profile or "baseline").strip().lower()
        if profile_key.startswith("agg"):
            high_pair = (0.5, 0.5)
            low_pair = (0.35, 0.65)
        else:
            high_pair = (0.8, 0.2)
            low_pair = (0.6, 0.4)

        if conf_arr is not None:
            high_mask = conf_arr >= 0.8
            model_weights = np.where(high_mask, total * high_pair[0], total * low_pair[0])
            rule_weights = np.where(high_mask, total * high_pair[1], total * low_pair[1])
            fused_scores = model_weights * model_arr + rule_weights * normalized_rules
        else:
            fused_scores = model_w * model_arr + rule_w * normalized_rules
        fused_flags = (fused_scores >= float(threshold or DEFAULT_FUSION_THRESHOLD)).astype(bool)

        return fused_scores, fused_flags, rules_triggered

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "MODEL_SCHEMA_VERSION",
    "META_COLUMNS",
    "TrainingSummary",
    "DetectionResult",
    "write_metadata",
    "ModelTrainer",
    "ModelPredictor",
    "EnsembleVotingClassifier",
    "train_hist_gradient_boosting",
    "train_supervised_on_split",
    "detect_pcap_with_model",
    "summarize_prediction_labels",
    "compute_risk_components",
]


_PARAM_CACHE: Dict[type, Set[str]] = {}


def _normalise_label_name(name: str) -> str:
    """Normalise label column names for tolerant matching."""

    return name.strip().lstrip("\ufeff").lower()


def _find_label_column(columns: Sequence[str], desired: str) -> Optional[str]:
    """Locate the label column, ignoring case, whitespace and BOM markers."""

    if desired in columns:
        return desired

    target = _normalise_label_name(desired)
    normalised = {_normalise_label_name(col): col for col in columns}
    return normalised.get(target)


class EnsembleVotingClassifier:
    """A lightweight voting classifier for incrementally trained ensembles."""

    def __init__(
        self,
        estimators: Sequence[Any],
        *,
        voting: str = "soft",
        weights: Optional[Sequence[Optional[float]]] = None,
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        weight_metric: Optional[str] = None,
    ) -> None:
        if not estimators:
            raise ValueError("estimators sequence cannot be empty")

        self.voting = voting
        self.estimators_: List[Any] = [est for est in estimators if est is not None]
        if not self.estimators_:
            raise ValueError("at least one valid estimator is required")

        seen: List[Any] = []
        for estimator in self.estimators_:
            classes = getattr(estimator, "classes_", None)
            if classes is None:
                continue
            for cls in classes:
                if not any(cls == existing for existing in seen):
                    seen.append(cls)

        if not seen:
            raise ValueError("estimators must expose non-empty classes_ for ensemble")

        self.classes_ = np.asarray(seen, dtype=object)
        self._class_index = {cls: idx for idx, cls in enumerate(self.classes_)}

        self.estimator_weights_ = self._prepare_weights(weights)
        self.estimator_metadata_ = self._prepare_metadata(metadata)
        self.weight_metric_ = self._normalize_weight_metric(weight_metric)

    @staticmethod
    def _normalize_weight_metric(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        token = str(value).strip().lower()
        if not token:
            return None
        if token in {"auc", "roc_auc"}:
            return "auc"
        if token in {"f1", "f1_score"}:
            return "f1"
        if token in {"samples", "sample_count", "rows"}:
            return "samples"
        return token

    def _prepare_weights(
        self, weights: Optional[Sequence[Optional[float]]]
    ) -> List[float]:
        sanitized: List[float] = []
        for idx, _ in enumerate(self.estimators_):
            candidate = weights[idx] if weights is not None and idx < len(weights) else None
            sanitized.append(self._sanitize_weight(candidate))
        if not sanitized:
            raise ValueError("ensemble requires at least one estimator weight")
        return sanitized

    def _prepare_metadata(
        self, metadata: Optional[Sequence[Optional[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for idx, _ in enumerate(self.estimators_):
            entry = metadata[idx] if metadata is not None and idx < len(metadata) else None
            if isinstance(entry, dict):
                prepared.append(dict(entry))
            else:
                prepared.append({})
        return prepared

    @staticmethod
    def _sanitize_weight(value: Optional[float]) -> float:
        if value is None:
            return 1.0
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(weight) or weight <= 0.0:
            return 1.0
        return weight

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.estimators_)

    def _empty_proba(self, rows: int) -> np.ndarray:
        return np.zeros((rows, self.classes_.size), dtype=np.float64)

    def _align_proba(self, estimator: Any, proba: np.ndarray) -> np.ndarray:
        aligned = self._empty_proba(len(proba))
        est_classes = getattr(estimator, "classes_", None)
        if est_classes is None:
            return aligned

        for src_idx, cls in enumerate(est_classes):
            dst_idx = self._class_index.get(cls)
            if dst_idx is None:
                continue
            aligned[:, dst_idx] = proba[:, src_idx]
        return aligned

    def _predict_to_proba(self, estimator: Any, X: np.ndarray) -> np.ndarray:
        predictions = estimator.predict(X)
        aligned = self._empty_proba(len(predictions))
        for row, label in enumerate(predictions):
            idx = self._class_index.get(label)
            if idx is None and isinstance(label, (np.generic, np.ndarray)):
                try:
                    idx = self._class_index.get(label.item())
                except Exception:  # pragma: no cover - defensive
                    idx = None
            if idx is not None:
                aligned[row, idx] = 1.0
        return aligned

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        accumulator: Optional[np.ndarray] = None
        total_weight = 0.0

        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
                proba = np.asarray(proba, dtype=np.float64)
                if proba.ndim == 1:
                    proba = np.column_stack([1.0 - proba, proba])
                aligned = self._align_proba(estimator, proba)
            else:
                aligned = self._predict_to_proba(estimator, X)

            if accumulator is None:
                accumulator = np.zeros_like(aligned)
            accumulator += aligned * weight
            total_weight += weight

        if accumulator is None or total_weight <= 0.0:
            raise RuntimeError("ensemble does not contain usable estimators")

        averaged = accumulator / float(total_weight)
        row_sums = averaged.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                averaged,
                row_sums,
                out=np.zeros_like(averaged),
                where=row_sums > 0,
            )
        if np.any(row_sums <= 0):
            zero_mask = row_sums[:, 0] <= 0
            normalized[zero_mask] = 1.0 / float(self.classes_.size)
        return normalized

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]


DEFAULT_MAX_ENSEMBLE_MEMBERS = 10


def _normalized_weight_metric(value: Optional[str]) -> str:
    normalized = EnsembleVotingClassifier._normalize_weight_metric(value)
    if normalized in {None, ""}:
        return "samples"
    return str(normalized)


def _compute_member_weight(
    weight_metric: str,
    sample_count: int,
    metrics: Optional[Dict[str, float]] = None,
) -> float:
    metrics = metrics or {}
    metric_value: Optional[float] = None

    def _pick(keys: Sequence[str]) -> Optional[float]:
        for key in keys:
            value = metrics.get(key)
            if value is None:
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(candidate) and candidate > 0.0:
                return candidate
        return None

    metric_lower = weight_metric.strip().lower()
    if metric_lower == "auc":
        metric_value = _pick(["val_auc", "validation_auc", "train_auc", "train_auc_macro"])
    elif metric_lower == "f1":
        metric_value = _pick([
            "val_f1", "validation_f1", "val_f1_weighted", "train_f1_weighted", "train_f1", "train_f1_macro",
        ])

    if metric_lower == "samples" or metric_value is None:
        metric_value = float(sample_count or 1)

    if not np.isfinite(metric_value) or metric_value <= 0.0:
        return float(sample_count or 1)
    return float(metric_value)


def _resolve_rule_fusion_settings() -> Dict[str, object]:
    try:
        rule_settings = get_rule_settings()
    except Exception:
        rule_settings = {}

    profile: Optional[str] = None
    rule_threshold_value: Optional[float] = None
    fusion_threshold_value: Optional[float] = None
    fusion_model_weight_value: Optional[float] = None
    fusion_rule_weight_value: Optional[float] = None

    if isinstance(rule_settings, dict):
        profile_raw = rule_settings.get("profile")
        if isinstance(profile_raw, str) and profile_raw.strip():
            profile = profile_raw.strip()

        try:
            rule_threshold_value = float(rule_settings.get("trigger_threshold"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            rule_threshold_value = float(DEFAULT_TRIGGER_THRESHOLD)

        try:
            fusion_threshold_value = float(rule_settings.get("fusion_threshold"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            fusion_threshold_value = float(DEFAULT_FUSION_THRESHOLD)

        try:
            fusion_model_weight_value = float(rule_settings.get("model_weight"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            fusion_model_weight_value = float(DEFAULT_MODEL_WEIGHT)

        try:
            fusion_rule_weight_value = float(rule_settings.get("rule_weight"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            fusion_rule_weight_value = float(DEFAULT_RULE_WEIGHT)

    total_weight = float((fusion_model_weight_value or 0.0) + (fusion_rule_weight_value or 0.0))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        normalized_model = 1.0
        normalized_rules = 0.0
    else:
        normalized_model = float(fusion_model_weight_value or 0.0) / total_weight
        normalized_rules = float(fusion_rule_weight_value or 0.0) / total_weight

    return {
        "profile": profile,
        "rule_threshold": rule_threshold_value,
        "fusion_threshold": fusion_threshold_value,
        "model_weight": fusion_model_weight_value,
        "rule_weight": fusion_rule_weight_value,
        "weights": {
            "model": float(normalized_model),
            "rules": float(normalized_rules),
        },
    }


def _normalize_member_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    estimator = item.get("estimator")
    if estimator is None:
        return None

    normalized: Dict[str, Any] = {"estimator": estimator}
    normalized["weight"] = EnsembleVotingClassifier._sanitize_weight(item.get("weight"))

    if "trained_at" in item and item.get("trained_at") is not None:
        normalized["trained_at"] = str(item.get("trained_at"))
    if "training_samples" in item and item.get("training_samples") is not None:
        try:
            normalized["training_samples"] = int(item.get("training_samples"))
        except (TypeError, ValueError):
            pass

    metrics = item.get("metrics")
    if isinstance(metrics, dict) and metrics:
        metric_map: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                metric_map[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        if metric_map:
            normalized["metrics"] = metric_map

    for extra_key in ("notes", "comment"):
        if extra_key in item:
            normalized[extra_key] = item[extra_key]

    return normalized


def _load_existing_ensemble(
    model_path: Union[str, Path]
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    path = Path(model_path)
    if not path.exists():
        return [], None

    try:
        artifact = load(path)
    except Exception:
        return [], None

    members: List[Dict[str, Any]] = []
    weight_metric: Optional[str] = None

    if isinstance(artifact, dict):
        ensemble_blob = artifact.get("ensemble")
        if isinstance(ensemble_blob, dict):
            weight_metric = ensemble_blob.get("weight_metric")
            raw_members = ensemble_blob.get("members")
            if isinstance(raw_members, list):
                for item in raw_members:
                    if isinstance(item, dict):
                        normalized = _normalize_member_record(item)
                        if normalized is not None:
                            members.append(normalized)
        else:
            model_obj = artifact.get("model")
            if isinstance(model_obj, EnsembleVotingClassifier):
                weight_metric = getattr(model_obj, "weight_metric_", None)
                for est, weight, meta in zip(
                    model_obj.estimators_,
                    getattr(model_obj, "estimator_weights_", []),
                    getattr(model_obj, "estimator_metadata_", []),
                ):
                    payload: Dict[str, Any] = {"estimator": est, "weight": weight}
                    if isinstance(meta, dict):
                        payload.update(meta)
                    normalized = _normalize_member_record(payload)
                    if normalized is not None:
                        members.append(normalized)
            elif model_obj is not None:
                normalized = _normalize_member_record({"estimator": model_obj, "weight": 1.0})
                if normalized is not None:
                    members.append(normalized)

    return members, weight_metric


def _valid_estimator_param_names(model_class: type) -> Set[str]:
    """Return the constructor parameter names supported by an estimator."""

    cached = _PARAM_CACHE.get(model_class)
    if cached is not None:
        return cached

    try:
        instance = model_class()
    except Exception:
        names: Set[str] = set()
    else:
        try:
            names = set(instance.get_params(deep=False).keys())
        except Exception:
            names = set()

    _PARAM_CACHE[model_class] = names
    return names


def _filter_estimator_params(
    model_class: type, **kwargs: Union[int, float, None]
) -> Dict[str, Union[int, float, None]]:
    """Drop unsupported or ``None`` values from estimator kwargs."""

    valid_names = _valid_estimator_param_names(model_class)
    if not valid_names:
        return {key: value for key, value in kwargs.items() if value is not None}

    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in valid_names
    }


@dataclass
class TrainingSummary:
    """Metadata describing a completed training run."""

    model_path: Path
    classes: List[str]
    feature_names: List[str]
    flow_count: int
    label_mapping: Optional[Dict[int, str]] = None
    dropped_flows: int = 0
    decision_threshold: Optional[float] = None
    positive_label: Optional[str] = None
    positive_class: Optional[Union[int, str]] = None
    metrics: Optional[Dict[str, float]] = None
    class_weights: Optional[Dict[str, float]] = None
    validation_samples: Optional[int] = None


@dataclass
class DetectionResult:
    """Container summarising model predictions for a PCAP file."""

    path: Path
    success: bool
    flow_count: int
    feature_names: List[str]
    predictions: List[int]
    scores: List[float]
    flows: List[Dict[str, object]]
    error: Optional[str] = None
    prediction_labels: Optional[List[str]] = None
    normalized_labels: Optional[List[str]] = None
    model_flags: Optional[List[bool]] = None
    model_statuses: Optional[List[Optional[str]]] = None
    rule_scores: Optional[List[float]] = None
    rule_flags: Optional[List[bool]] = None
    rule_reasons: Optional[List[str]] = None
    fusion_scores: Optional[List[float]] = None
    fusion_flags: Optional[List[bool]] = None
    fusion_statuses: Optional[List[str]] = None
    fusion_threshold: Optional[float] = None
    fusion_weights: Optional[Dict[str, float]] = None
    rule_profile: Optional[str] = None


DEFAULT_MODEL_PARAMS: Dict[str, Union[int, float, None]] = {
    "learning_rate": 0.1,
    "max_depth": None,
    "max_iter": 300,
    "l2_regularization": 0.0,
    "random_state": 1337,
}


def write_metadata(
    model_path: Union[str, Path],
    feature_names: Sequence[str],
    label_mapping: Optional[Dict[int, str]] = None,
    model_params: Optional[Dict[str, Union[int, float, str, None]]] = None,
) -> Path:
    """Persist model metadata required by the UI layer."""

    metadata: Dict[str, object] = {
        "feature_names": list(feature_names),
        "label_mapping": label_mapping or {},
        "model_params": model_params or {},
        "schema_version": MODEL_SCHEMA_VERSION,
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = Path(model_path).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata_path


MODEL_SCHEMA_VERSION = "2025.10"

# Columns that should be preserved when presenting prediction results in the UI.
META_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Timestamp",
}

# Keywords used to interpret prediction labels as "正常" or "异常".
_ANOMALY_KEYWORDS = {
    "1",
    "-1",
    "attack",
    "anomaly",
    "anomalous",
    "malicious",
    "恶意",
    "异常",
    "是",
    "yes",
    "true",
}

_NORMAL_KEYWORDS = {
    "0",
    "benign",
    "normal",
    "legit",
    "合法",
    "正常",
    "否",
    "no",
    "false",
}

_POSITIVE_KEYWORDS = {
    "异常",
    "恶意",
    "malicious",
    "anomaly",
    "attack",
    "1",
    "-1",
}


@dataclass
class _PositiveClassInfo:
    index: Optional[int]
    value: Optional[Union[int, str]]
    label: Optional[str]


def _resolve_positive_class(
    model: HistGradientBoostingClassifier,
    label_mapping: Optional[Dict[int, str]] = None,
) -> _PositiveClassInfo:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return _PositiveClassInfo(None, None, None)

    class_list = list(classes)
    pos_idx: Optional[int] = None

    def _label_for(cls_value: Union[int, str]) -> str:
        if label_mapping:
            try:
                mapped = label_mapping.get(int(cls_value))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                mapped = label_mapping.get(str(cls_value))  # type: ignore[arg-type]
            if mapped is not None:
                return str(mapped).strip()
        return str(cls_value).strip()

    for idx, cls_value in enumerate(class_list):
        lowered = _label_for(cls_value).lower()
        if lowered in _POSITIVE_KEYWORDS:
            pos_idx = idx
            break

    if pos_idx is None:
        try:
            if 1 in class_list:
                pos_idx = class_list.index(1)
            elif -1 in class_list:
                pos_idx = class_list.index(-1)
        except ValueError:
            pos_idx = None

    if pos_idx is None and class_list:
        pos_idx = len(class_list) - 1

    if pos_idx is None:
        return _PositiveClassInfo(None, None, None)

    positive_value = class_list[pos_idx]
    positive_label = _label_for(positive_value)
    return _PositiveClassInfo(pos_idx, positive_value, positive_label)


def _predict_positive_scores(
    model: Any,
    X: np.ndarray,
    label_mapping: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, _PositiveClassInfo]:
    info = _resolve_positive_class(model, label_mapping)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if (
            info.index is not None
            and np.ndim(proba) == 2
            and proba.shape[1] >= 2
        ):
            scores = proba[:, info.index]
        elif np.ndim(proba) == 2:
            scores = proba.max(axis=1)
        else:
            scores = proba
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if np.ndim(decision) == 1:
            scores = 1.0 / (1.0 + np.exp(-decision))
        else:
            if info.index is not None and decision.shape[1] > info.index:
                scores = decision[:, info.index]
            else:
                scores = decision.max(axis=1)
    else:
        predictions = model.predict(X)
        scores = predictions

    return np.asarray(scores, dtype=np.float64), info


def train_hist_gradient_boosting(
    dataset_path: Union[str, Path],
    model_path: Union[str, Path],
    **kwargs: Union[int, float, None],
) -> TrainingSummary:
    """Train a tree-based classifier similar to the EMBER pipeline."""

    max_members_raw = kwargs.pop("max_ensemble_members", None)
    ensemble_weight_metric_raw = kwargs.pop("ensemble_weight_metric", None)
    reset_ensemble_flag = kwargs.pop("reset_ensemble", None)

    X, y, feature_names, label_mapping, stats = load_vectorized_dataset(
        dataset_path, show_progress=True, return_stats=True
    )
    if y is None or y.size == 0:
        if stats.total_rows and not stats.labeled_rows:
            raise ValueError(
                "Dataset includes a label column but no non-empty labels; cannot train a classifier."
            )
        raise ValueError("Dataset does not contain labels; cannot train a classifier.")

    filtered_params = _filter_estimator_params(HistGradientBoostingClassifier, **kwargs)

    params = DEFAULT_MODEL_PARAMS.copy()
    params.update(filtered_params)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X, y)

    sample_count = int(X.shape[0])

    metrics_map: Dict[str, float] = {}
    try:
        train_predictions = clf.predict(X)
    except Exception:
        train_predictions = None

    if train_predictions is not None:
        try:
            metrics_map["train_f1_weighted"] = float(
                f1_score(y, train_predictions, average="weighted")
            )
        except Exception:
            pass

    try:
        if hasattr(clf, "predict_proba"):
            proba_train = clf.predict_proba(X)
        else:
            proba_train = None
    except Exception:
        proba_train = None

    if proba_train is not None and np.ndim(proba_train) == 2 and proba_train.size:
        unique_labels = np.unique(y)
        try:
            if proba_train.shape[1] == 2 or unique_labels.size == 2:
                pos_info = _resolve_positive_class(clf, label_mapping)
                pos_index = pos_info.index if pos_info.index is not None else proba_train.shape[1] - 1
                pos_index = max(0, min(proba_train.shape[1] - 1, int(pos_index)))
                metrics_map["train_auc"] = float(
                    roc_auc_score(y, proba_train[:, pos_index])
                )
            elif proba_train.shape[1] > 2 and unique_labels.size > 2:
                metrics_map["train_auc_macro"] = float(
                    roc_auc_score(y, proba_train, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    existing_members, existing_metric = _load_existing_ensemble(model_path)

    try:
        max_members = int(max_members_raw) if max_members_raw is not None else DEFAULT_MAX_ENSEMBLE_MEMBERS
    except (TypeError, ValueError):
        max_members = DEFAULT_MAX_ENSEMBLE_MEMBERS
    max_members = max(1, max_members)

    requested_metric = None
    if ensemble_weight_metric_raw is not None:
        requested_metric = str(ensemble_weight_metric_raw)
    elif existing_metric is not None:
        requested_metric = str(existing_metric)
    weight_metric = _normalized_weight_metric(requested_metric)

    reset_ensemble = bool(reset_ensemble_flag)
    retained_members = list(existing_members)
    total_dropped = 0

    if reset_ensemble and retained_members:
        total_dropped += len(retained_members)
        retained_members = []

    if max_members <= 1:
        total_dropped += len(retained_members)
        retained_members = []
    else:
        allowed_previous = max_members - 1
        if len(retained_members) > allowed_previous:
            drop_count = len(retained_members) - allowed_previous
            total_dropped += drop_count
            retained_members = retained_members[-allowed_previous:]

    trained_at = datetime.now().isoformat(timespec="seconds")
    member_weight = _compute_member_weight(weight_metric, sample_count, metrics_map)

    new_member: Dict[str, Any] = {
        "estimator": clf,
        "weight": member_weight,
        "trained_at": trained_at,
        "training_samples": sample_count,
    }
    if metrics_map:
        new_member["metrics"] = metrics_map
    if reset_ensemble:
        new_member["reset"] = True

    ensemble_members = retained_members + [new_member]

    estimators = [item["estimator"] for item in ensemble_members]
    weights = [EnsembleVotingClassifier._sanitize_weight(item.get("weight")) for item in ensemble_members]
    estimator_metadata = []
    for item in ensemble_members:
        meta = {key: value for key, value in item.items() if key not in {"estimator", "weight"}}
        estimator_metadata.append(meta)

    ensemble_model = EnsembleVotingClassifier(
        estimators,
        voting="soft",
        weights=weights,
        metadata=estimator_metadata,
        weight_metric=weight_metric,
    )

    ensemble_info = {
        "members": ensemble_members,
        "max_members": max_members,
        "weight_metric": weight_metric,
        "updated_at": trained_at,
        "reset": bool(reset_ensemble),
    }

    artifact = {
        "model": ensemble_model,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "ensemble": ensemble_info,
    }
    dump(artifact, model_path)

    try:
        model_params = clf.get_params(deep=False)
    except Exception:  # pragma: no cover - defensive
        model_params = params
    write_metadata(model_path, feature_names, label_mapping, model_params)

    ensemble_count = len(ensemble_members)
    retained_count = max(ensemble_count - 1, 0)
    ensemble_metrics: Optional[Dict[str, float]] = {
        "ensemble_members": float(ensemble_count),
        "ensemble_added": 1.0,
        "ensemble_retained": float(retained_count),
        "ensemble_dropped": float(total_dropped),
    }

    fusion_settings = _resolve_rule_fusion_settings()

    _write_model_metadata(
        model_path,
        feature_names=feature_names,
        label_mapping=label_mapping,
        decision_threshold=None,
        positive_label=None,
        positive_class=None,
        metrics=ensemble_metrics,
        class_weights=None,
        ensemble_members=ensemble_count,
        ensemble_added=1,
        ensemble_retained=retained_count,
        ensemble_dropped=total_dropped,
        ensemble_weight_metric=weight_metric,
        ensemble_max_members=max_members,
        rule_profile=fusion_settings.get("profile"),
        rule_threshold=fusion_settings.get("rule_threshold"),
        fusion_threshold=fusion_settings.get("fusion_threshold"),
        fusion_model_weight=fusion_settings.get("model_weight"),
        fusion_rule_weight=fusion_settings.get("rule_weight"),
        fusion_weights=fusion_settings.get("weights"),
    )

    if label_mapping:
        classes_display = [label_mapping.get(int(cls), str(cls)) for cls in ensemble_model.classes_]
    else:
        classes_display = [str(cls) for cls in ensemble_model.classes_]

    return TrainingSummary(
        model_path=Path(model_path),
        classes=classes_display,
        feature_names=feature_names,
        flow_count=int(X.shape[0]),
        label_mapping=label_mapping,
        dropped_flows=int(stats.dropped_rows),
        metrics=ensemble_metrics,
    )


def _write_model_metadata(
    model_path: Union[str, Path],
    *,
    feature_names: Iterable[Union[str, bytes]],
    label_mapping: Optional[Dict[int, str]] = None,
    decision_threshold: Optional[float] = None,
    positive_label: Optional[str] = None,
    positive_class: Optional[Union[int, str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    class_weights: Optional[Dict[str, float]] = None,
    ensemble_members: Optional[int] = None,
    ensemble_added: Optional[int] = None,
    ensemble_retained: Optional[int] = None,
    ensemble_dropped: Optional[int] = None,
    ensemble_weight_metric: Optional[str] = None,
    ensemble_max_members: Optional[int] = None,
    rule_profile: Optional[str] = None,
    rule_threshold: Optional[float] = None,
    fusion_threshold: Optional[float] = None,
    fusion_model_weight: Optional[float] = None,
    fusion_rule_weight: Optional[float] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Path:
    """Persist metadata alongside the trained model for UI alignment."""

    path = Path(model_path)
    metadata_path = path.with_name("metadata.json")

    feature_list = [str(name) for name in feature_names]
    if label_mapping:
        labels = {str(key): str(value) for key, value in label_mapping.items()}
    else:
        labels = {}

    metadata = {
        "feature_names": feature_list,
        "feature_order": feature_list,
        "feature_names_in": feature_list,
        "label_mapping": labels,
        "model_path": str(path.name),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "score_column": "malicious_score",
    }

    if decision_threshold is not None:
        metadata["threshold"] = float(decision_threshold)
        metadata["decision_threshold"] = float(decision_threshold)
    if positive_label:
        metadata["positive_label"] = str(positive_label)
    if positive_class is not None:
        metadata["positive_class"] = (
            int(positive_class)
            if isinstance(positive_class, (int, np.integer))
            else str(positive_class)
        )
    if metrics:
        metadata["model_metrics"] = {key: float(value) for key, value in metrics.items()}
    if class_weights:
        metadata["class_weights"] = class_weights
    if ensemble_members is not None:
        metadata["ensemble_members"] = int(ensemble_members)
    if ensemble_added is not None:
        metadata["ensemble_added"] = int(ensemble_added)
    if ensemble_retained is not None:
        metadata["ensemble_retained"] = int(ensemble_retained)
    if ensemble_dropped is not None:
        metadata["ensemble_dropped"] = int(ensemble_dropped)
    if ensemble_weight_metric:
        metadata["ensemble_weight_metric"] = str(ensemble_weight_metric)
    if ensemble_max_members is not None:
        metadata["ensemble_max_members"] = int(ensemble_max_members)
    if rule_profile:
        metadata["rule_profile"] = str(rule_profile)
    if rule_threshold is not None:
        metadata["rule_threshold"] = float(rule_threshold)
    if fusion_threshold is not None:
        metadata["fusion_threshold"] = float(fusion_threshold)
    if fusion_model_weight is not None:
        metadata["fusion_model_weight"] = float(fusion_model_weight)
    if fusion_rule_weight is not None:
        metadata["fusion_rule_weight"] = float(fusion_rule_weight)

    weights_payload: Optional[Dict[str, float]]
    if isinstance(fusion_weights, dict) and fusion_weights:
        weights_payload = {
            "model": float(fusion_weights.get("model", 1.0)),
            "rules": float(fusion_weights.get("rules", 0.0)),
        }
    else:
        total = float((fusion_model_weight or 0.0) + (fusion_rule_weight or 0.0))
        if not np.isfinite(total) or total <= 0.0:
            weights_payload = {"model": 1.0, "rules": 0.0}
        else:
            weights_payload = {
                "model": float(fusion_model_weight or 0.0) / total,
                "rules": float(fusion_rule_weight or 0.0) / total,
            }

    metadata["fusion_weights"] = weights_payload

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata_path


def _load_model_artifact(
    model_path: Union[str, Path]
) -> Tuple[Any, List[str], Optional[Dict[int, str]]]:
    artifact = load(model_path)
    model = artifact.get("model")
    feature_names = artifact.get("feature_names")
    label_mapping = artifact.get("label_mapping")

    metadata_path = Path(model_path).with_suffix(".json")
    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:  # pragma: no cover - metadata is optional
            metadata = {}

    if feature_names is None:
        feature_names = metadata.get("feature_names") if metadata else None
    if feature_names is None:
        raise ValueError("Model artifact is missing required feature names")

    if label_mapping is None and metadata:
        raw_mapping = metadata.get("label_mapping")
        if isinstance(raw_mapping, dict):
            parsed_mapping: Dict[int, str] = {}
            for key, value in raw_mapping.items():
                try:
                    parsed_mapping[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
            if parsed_mapping:
                label_mapping = parsed_mapping

    if model is None:
        raise ValueError("Model artifact is missing the estimator instance")

    return model, [str(name) for name in feature_names], label_mapping if label_mapping else None


def _resolve_dataset_path(input_path: Union[str, Path]) -> Path:
    """Resolve the dataset CSV path from a user-provided input."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到训练数据: {input_path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError("目前仅支持以 CSV 形式提供的训练数据。")
        return path

    candidates: List[Path] = []
    preferred_tokens = ("vectorized", "feature", "train")
    for child in path.glob("*.csv"):
        lowered = child.name.lower()
        if any(token in lowered for token in preferred_tokens):
            candidates.append(child)
    if not candidates:
        candidates = list(path.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError("指定目录中未找到任何 CSV 训练数据。")

    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_detection_result(
    model: Any,
    feature_names: List[str],
    label_mapping: Optional[Dict[int, str]],
    flows: List[Dict[str, object]],
    path: Union[str, Path],
    *,
    vectorized: Optional[VectorizationResult] = None,
    predictions: Optional[Sequence[object]] = None,
    scores: Optional[Sequence[float]] = None,
) -> DetectionResult:
    if vectorized is None:
        vectorized = vectorize_flows(
            flows, feature_names=feature_names, include_labels=False
        )

    if vectorized.flow_count == 0:
        return DetectionResult(
            path=Path(path),
            success=True,
            flow_count=0,
            feature_names=feature_names,
            predictions=[],
            scores=[],
            flows=[],
            prediction_labels=[],
        )

    X = vectorized.matrix

    computed_scores: np.ndarray
    if scores is None:
        computed_scores, _ = _predict_positive_scores(model, X, label_mapping)
    else:
        computed_scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    predicted_values: np.ndarray
    if predictions is None:
        predicted_values = np.asarray(model.predict(X)).reshape(-1)
    else:
        predicted_values = np.asarray(predictions).reshape(-1)

    if predicted_values.shape[0] != vectorized.flow_count:
        raise ValueError("prediction count does not match flow count")

    if computed_scores.shape[0] != vectorized.flow_count:
        raise ValueError("score count does not match flow count")

    prediction_labels: List[str] = []
    if label_mapping:
        for value in predicted_values:
            prediction_labels.append(label_mapping.get(int(value), str(value)))
    else:
        prediction_labels = [str(value) for value in predicted_values]

    normalized_labels, _, _, _ = summarize_prediction_labels(
        predicted_values, label_mapping
    )

    model_statuses: List[Optional[str]] = []
    model_flags = np.zeros(vectorized.flow_count, dtype=bool)
    for idx in range(vectorized.flow_count):
        base_status = normalized_labels[idx] if idx < len(normalized_labels) else None
        interpreted = base_status if base_status in {"异常", "正常"} else _interpret_label_value(
            predicted_values[idx]
        )
        if interpreted == "异常":
            model_flags[idx] = True
            model_statuses.append("异常")
        elif interpreted == "正常":
            model_flags[idx] = False
            model_statuses.append("正常")
        else:
            fallback_flag = bool(computed_scores[idx] >= 0.5)
            model_flags[idx] = fallback_flag
            model_statuses.append("异常" if fallback_flag else "正常")

    def _safe_float(value: object, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    try:
        rule_config = get_rule_settings()
    except Exception:
        rule_config = {}

    rule_params_config = rule_config.get("params") if isinstance(rule_config, dict) else None
    rule_params_dict = (
        dict(rule_params_config) if isinstance(rule_params_config, dict) else {}
    )
    rule_profile = rule_config.get("profile") if isinstance(rule_config, dict) else None
    if not isinstance(rule_profile, str) or not rule_profile.strip():
        rule_profile = None
    else:
        rule_profile = rule_profile.strip()

    try:
        fusion_defaults = get_fusion_settings(profile=rule_profile)
    except Exception:
        fusion_defaults = {
            "model_weight": DEFAULT_MODEL_WEIGHT,
            "rule_weight": DEFAULT_RULE_WEIGHT,
            "fusion_threshold": DEFAULT_FUSION_THRESHOLD,
            "profile": rule_profile or "baseline",
        }

    if rule_profile is None:
        profile_candidate = fusion_defaults.get("profile")
        if isinstance(profile_candidate, str) and profile_candidate.strip():
            rule_profile = profile_candidate.strip()

    default_rule_threshold = _safe_float(
        rule_config.get("trigger_threshold") if isinstance(rule_config, dict) else None,
        DEFAULT_TRIGGER_THRESHOLD,
    )
    effective_rule_threshold = min(
        float(default_rule_threshold),
        _safe_float(RULE_TRIGGER_THRESHOLD, default_rule_threshold),
    )
    fusion_model_weight_base = _safe_float(
        fusion_defaults.get("model_weight") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_MODEL_WEIGHT,
    )
    fusion_rule_weight_base = _safe_float(
        fusion_defaults.get("rule_weight") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_RULE_WEIGHT,
    )
    fusion_threshold_value = _safe_float(
        fusion_defaults.get("fusion_threshold") if isinstance(fusion_defaults, dict) else None,
        DEFAULT_FUSION_THRESHOLD,
    )

    rule_scores_array: Optional[np.ndarray] = None
    rule_reasons: Optional[List[str]] = None
    rule_flags: Optional[np.ndarray] = None
    if pd is not None and apply_risk_rules is not None and flows:
        try:
            frame = pd.DataFrame(flows)
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            try:
                score_series, reason_series = apply_risk_rules(
                    frame,
                    params=rule_params_dict,
                    profile=rule_profile,
                )
            except Exception:
                score_series = None
                reason_series = None
            if score_series is not None:
                rule_scores_array = np.asarray(score_series, dtype=np.float64).reshape(-1)
                if reason_series is not None:
                    rule_reasons = [str(value) if value is not None else "" for value in reason_series.tolist()]
                rule_flags = rule_scores_array >= float(effective_rule_threshold)

    if rule_scores_array is not None and getattr(rule_scores_array, "size", 0) == 0:
        rule_scores_array = None

    model_score_input = np.clip(computed_scores, 0.0, 1.0)
    model_confidence = np.maximum(model_score_input, 1.0 - model_score_input)

    fusion_scores, fusion_flags, rules_triggered = fuse_model_rule_votes(
        model_score_input,
        rule_scores_array,
        profile=rule_profile,
        model_weight=float(fusion_model_weight_base),
        rule_weight=float(fusion_rule_weight_base),
        threshold=float(fusion_threshold_value),
        trigger_threshold=float(effective_rule_threshold),
        model_confidence=model_confidence,
    )
    fusion_scores = np.asarray(fusion_scores, dtype=np.float64).reshape(-1)
    fusion_flags = np.asarray(fusion_flags, dtype=bool).reshape(-1)
    rules_triggered = np.asarray(rules_triggered, dtype=bool).reshape(-1)

    rule_flags = rules_triggered

    final_statuses: List[str] = []
    for idx in range(vectorized.flow_count):
        is_anomaly = bool(fusion_flags[idx]) if idx < fusion_flags.size else bool(model_flags[idx])
        final_statuses.append("异常" if is_anomaly else "正常")

    annotated_flows: List[Dict[str, object]] = []
    for index, (flow, label, label_name, score) in enumerate(
        zip(flows, predicted_values, prediction_labels, computed_scores)
    ):
        annotated = dict(flow)
        try:
            annotated["prediction"] = int(label)
        except Exception:
            annotated["prediction"] = label
        annotated["prediction_label"] = label_name
        annotated["malicious_score"] = float(score)
        if index < len(model_statuses) and model_statuses[index] is not None:
            annotated["model_status"] = model_statuses[index]
        if index < len(final_statuses):
            annotated["prediction_status"] = int(bool(fusion_flags[index]))
            annotated["fusion_status"] = final_statuses[index]
        if fusion_scores.size > index:
            annotated["fusion_score"] = float(fusion_scores[index])
            annotated["fusion_decision"] = int(bool(fusion_flags[index]))
        annotated["model_flag"] = bool(model_flags[index]) if index < model_flags.shape[0] else False
        if rule_scores_array is not None and index < rule_scores_array.shape[0]:
            annotated["rules_score"] = float(rule_scores_array[index])
            triggered = (
                bool(rule_flags[index])
                if rule_flags is not None and index < rule_flags.shape[0]
                else bool(rule_scores_array[index] >= float(effective_rule_threshold))
            )
            if triggered:
                annotated["rules_triggered"] = True
            if rule_reasons is not None and index < len(rule_reasons):
                reason_value = rule_reasons[index]
                if reason_value:
                    annotated["rules_reasons"] = reason_value
        annotated_flows.append(annotated)

    predictions_list = [int(value) for value in predicted_values]
    scores_list = [float(value) for value in computed_scores]
    fusion_score_list = [float(value) for value in fusion_scores.tolist()]
    fusion_flag_list = [bool(flag) for flag in fusion_flags.tolist()]
    model_flag_list = [bool(flag) for flag in model_flags.tolist()]

    if rule_scores_array is not None:
        rule_score_list = [float(value) for value in rule_scores_array.tolist()]
    else:
        rule_score_list = None

    if rule_flags is not None:
        rule_flag_list = [bool(flag) for flag in rule_flags.tolist()]
    elif rule_score_list is not None:
        rule_flag_list = [score >= float(effective_rule_threshold) for score in rule_score_list]
    else:
        rule_flag_list = None

    if rule_reasons is not None:
        rule_reason_list = [reason for reason in rule_reasons]
    else:
        rule_reason_list = None

    total_weight = float(fusion_model_weight_base + fusion_rule_weight_base)
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        fusion_weight_model = 1.0
        fusion_weight_rules = 0.0
    else:
        fusion_weight_model = float(fusion_model_weight_base) / total_weight
        fusion_weight_rules = float(fusion_rule_weight_base) / total_weight

    return DetectionResult(
        path=Path(path),
        success=True,
        flow_count=len(flows),
        feature_names=feature_names,
        predictions=predictions_list,
        scores=scores_list,
        flows=annotated_flows,
        prediction_labels=prediction_labels,
        normalized_labels=list(normalized_labels),
        model_flags=model_flag_list,
        model_statuses=[status for status in model_statuses],
        rule_scores=rule_score_list,
        rule_flags=rule_flag_list,
        rule_reasons=rule_reason_list,
        fusion_scores=fusion_score_list,
        fusion_flags=fusion_flag_list,
        fusion_statuses=final_statuses,
        fusion_threshold=float(fusion_threshold_value),
        fusion_weights={
            "model": fusion_weight_model,
            "rules": fusion_weight_rules,
        },
        rule_profile=rule_profile,
    )


class ModelTrainer:
    """High-level interface combining training and metadata generation."""

    def __init__(self, *, model_class=HistGradientBoostingClassifier):
        self.model_class = model_class
        self._valid_param_names = _valid_estimator_param_names(model_class)

    @staticmethod
    def _filter_params_static(
        valid_names: Set[str], **kwargs: Union[int, float, None]
    ) -> Dict[str, Union[int, float, None]]:
        if not valid_names:
            return {key: value for key, value in kwargs.items() if value is not None}
        return {
            key: value
            for key, value in kwargs.items()
            if value is not None and key in valid_names
        }

    def _filter_params(self, **kwargs: Union[int, float, None]) -> Dict[str, Union[int, float, None]]:
        return self._filter_params_static(self._valid_param_names, **kwargs)

    def train(
        self,
        dataset_path: Union[str, Path],
        model_path: Union[str, Path],
        **kwargs: Union[int, float, None],
    ) -> TrainingSummary:
        ensemble_options: Dict[str, Union[int, float, None]] = {}
        for key in ("max_ensemble_members", "ensemble_weight_metric", "reset_ensemble"):
            if key in kwargs:
                ensemble_options[key] = kwargs.pop(key)

        model_kwargs = self._filter_params(**kwargs)
        return train_hist_gradient_boosting(
            dataset_path,
            model_path,
            **ensemble_options,
            **model_kwargs,
        )

    def train_from_split(
        self,
        input_path: Union[str, Path],
        results_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        **kwargs: Union[int, float, None],
    ) -> Dict[str, object]:
        return train_supervised_on_split(
            split_dir=input_path,
            results_dir=results_dir,
            models_dir=models_dir,
            **kwargs,
        )


class ModelPredictor:
    """Convenience wrapper to load a trained model and perform inference."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.model, self.feature_names, self.label_mapping = _load_model_artifact(
            self.model_path
        )

    def predict_matrix(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and anomaly scores for a batch of samples."""

        model = self.model
        predictions = np.asarray(model.predict(matrix)).reshape(-1)

        scores, _ = _predict_positive_scores(model, matrix, self.label_mapping)

        return predictions, np.asarray(scores, dtype=np.float64)

    def predict_flows(self, flows: Iterable[Dict[str, object]], *, path: Union[str, Path] = "memory") -> DetectionResult:
        flow_list = [dict(flow) for flow in flows]
        vectorized = vectorize_flows(
            flow_list, feature_names=self.feature_names, include_labels=False
        )

        if vectorized.flow_count > 0:
            predictions, scores = self.predict_matrix(vectorized.matrix)
        else:
            predictions, scores = None, None

        return _build_detection_result(
            self.model,
            self.feature_names,
            self.label_mapping,
            flow_list,
            path,
            vectorized=vectorized,
            predictions=predictions,
            scores=scores,
        )

    def detect_pcap(self, pcap_path: Union[str, Path]) -> DetectionResult:
        return detect_pcap_with_model(self.model_path, pcap_path)


def detect_pcap_with_model(
    model_path: Union[str, Path],
    pcap_path: Union[str, Path],
) -> DetectionResult:
    """Vectorize a PCAP file and run inference using a trained model."""

    model, feature_names, label_mapping = _load_model_artifact(model_path)
    result = extract_pcap_features(pcap_path)

    if not result.get("success", False):
        return DetectionResult(
            path=Path(pcap_path),
            success=False,
            flow_count=0,
            feature_names=feature_names,
            predictions=[],
            scores=[],
            flows=[],
            error=result.get("error"),
            prediction_labels=None,
        )

    flows = [dict(flow) for flow in result.get("flows", [])]
    return _build_detection_result(model, feature_names, label_mapping, flows, pcap_path)


def train_supervised_on_split(
    split_dir: Union[str, Path],
    results_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    *,
    model_tag: str = "latest",
    label_col: str = "LabelBinary",  # ★ 默认就用 LabelBinary
    positive_labels: Sequence[str] = ("MALICIOUS", "ATTACK"),
    **kwargs: Union[int, float, None],
) -> Dict[str, object]:
    """Train a supervised classifier on labelled split CSVs.

    The return structure matches the legacy unsupervised entry point so the UI/CLI
    can stay unchanged while switching the underlying model family.
    """

    if pd is None:
        raise RuntimeError("训练流程需要 pandas 依赖，请先安装相关依赖。")

    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"未找到训练数据目录: {split_dir}")

    csv_files: List[Path] = []
    if split_path.is_file():
        if split_path.suffix.lower() != ".csv":
            raise RuntimeError("训练数据必须为 CSV 文件或包含 CSV 的目录。")
        csv_files.append(split_path)
    else:
        for root, _, files in os.walk(split_path):
            for name in files:
                if name.lower().endswith(".csv"):
                    csv_files.append(Path(root) / name)

    if not csv_files:
        raise RuntimeError(f"未在 {split_dir} 找到任何 CSV 用于训练")

    frames: List[pd.DataFrame] = []
    for path in sorted(csv_files):
        try:
            frames.append(read_csv_flexible(path))
        except Exception as exc:  # pragma: no cover - 容错读取
            logger.warning("跳过无法读取的文件 %s: %s", path, exc)
            continue

    if not frames:
        raise RuntimeError("无法从训练目录读取任何有效的特征 CSV。")

    full_df = pd.concat(frames, ignore_index=True)
    if full_df.empty:
        raise RuntimeError("训练数据为空，无法进行有监督建模。")

    matched_label_col = _find_label_column(full_df.columns, label_col)
    if matched_label_col is None:
        available = ", ".join(str(col) for col in full_df.columns)
        raise ValueError(f"数据中不存在标签列 {label_col} (现有列: {available})")

    if matched_label_col != label_col:
        logger.info("检测到标签列 %s ，将替代期望列 %s", matched_label_col, label_col)

    # ★ 新增：检查 & 转成 0/1 数值
    print("调试标签列:", matched_label_col)
    print(full_df[matched_label_col].head())
    print(full_df[matched_label_col].dtype)
    print(full_df[matched_label_col].value_counts(dropna=False))

    full_df[matched_label_col] = full_df[matched_label_col].astype(float).astype(int)

    raw_labels = full_df[matched_label_col]
    if raw_labels.dtype == "O":
        positive_set = {str(value).upper() for value in positive_labels}
        y = raw_labels.astype(str).str.upper().isin(positive_set).astype(int)
    else:
        y = raw_labels.astype(int)

    feature_columns = list(numeric_feature_names())
    feature_df = full_df.reindex(columns=feature_columns, fill_value=0.0)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = feature_df.to_numpy(dtype=np.float64, copy=False)
    y_arr = y.to_numpy(dtype=np.int64, copy=False)

    test_size = float(kwargs.pop("test_size", 0.2) or 0.2)
    random_state = kwargs.pop("random_state", 42)

    filtered_params = _filter_estimator_params(HistGradientBoostingClassifier, **kwargs)
    params = DEFAULT_MODEL_PARAMS.copy()
    params.update(filtered_params)

    stratify_labels = y_arr if len(np.unique(y_arr)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print("y_test 分布:", np.bincount(y_test))
    print("y_pred 分布:", np.bincount(y_pred))
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

    logger.info(
        "模型评估指标 accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    clf.fit(X, y_arr)

    label_mapping = {0: "BENIGN", 1: "MALICIOUS"}

    models_root = Path(models_dir) if models_dir else split_path
    models_root.mkdir(parents=True, exist_ok=True)
    model_path = models_root / "model.joblib"

    pipeline_payload = {
        "model": clf,
        "feature_names": feature_columns,
        "label_mapping": label_mapping,
    }
    dump(pipeline_payload, model_path)

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    stamp_token = timestamp.strftime("%Y%m%d_%H%M%S")

    metadata: Dict[str, object] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "timestamp": timestamp_str,
        "pipeline_latest": model_path.name,
        "pipeline_path": model_path.name,
        "feature_order": feature_columns,
        "feature_names_in": feature_columns,
        "feature_columns": feature_columns,
        "label_mapping": label_mapping,
        "model_tag": model_tag,
        "model_type": "supervised_hist_gradient_boosting",
        "label_column": label_col,
        "positive_labels": list(positive_labels),
        "positive_class": 1,
        "positive_label": "MALICIOUS",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "ensemble_members": 1,
    }

    metadata["model_metrics"] = metrics

    metadata_path = models_root / f"iforest_metadata_{stamp_token}.json"
    latest_metadata_path = models_root / "latest_iforest_metadata.json"
    model_metadata_path = model_path.with_suffix(".json")

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    with latest_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    with model_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    return {
        "model_path": str(model_path),
        "pipeline_path": str(model_path),
        "pipeline_latest": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata_latest": str(latest_metadata_path),
        "model_metadata_path": str(model_metadata_path),
        "model_joblib": str(model_path),
        "results_csv": None,
        "summary_csv": None,
        "scaler_path": None,
        "flows": int(X.shape[0]),
        "malicious": int(np.count_nonzero(y_arr)),
        "feature_columns": feature_columns,
        "classes": ["0", "1"],
        "label_mapping": label_mapping,
        "dropped_flows": 0,
        "timestamp": timestamp_str,
        "schema_version": MODEL_SCHEMA_VERSION,
        "summary": None,
        "metadata": metadata,
        "decision_threshold": None,
        "model_metrics": metrics,
        "positive_label": "MALICIOUS",
        "positive_class": 1,
    }


def _interpret_label_name(label: str) -> Optional[str]:
    token = label.strip().lower()
    if token in _ANOMALY_KEYWORDS:
        return "异常"
    if token in _NORMAL_KEYWORDS:
        return "正常"
    return None


def _interpret_label_value(value: object) -> Optional[str]:
    try:
        int_value = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if int_value in (1, -1):
        return "异常"
    if int_value == 0:
        return "正常"
    return None


def summarize_prediction_labels(
    predictions: Iterable[object],
    label_mapping: Optional[Dict[int, str]] = None,
) -> Tuple[List[str], int, int, Optional[str]]:
    """Normalise prediction labels and derive anomaly statistics."""

    normalized_labels: List[str] = []
    anomaly_count = 0
    normal_count = 0

    for value in predictions:
        mapped: Optional[str] = None
        if label_mapping is not None:
            try:
                mapped = label_mapping.get(int(value))
            except (TypeError, ValueError):
                mapped = None

        label_name = mapped or str(value)

        status = _interpret_label_name(label_name)
        if status is None:
            status = _interpret_label_value(value)

        if status == "异常":
            anomaly_count += 1
            normalized_labels.append("异常")
        elif status == "正常":
            normal_count += 1
            normalized_labels.append("正常")
        else:
            normalized_labels.append(label_name)
            inferred = _interpret_label_value(value)
            if inferred == "异常":
                anomaly_count += 1
            elif inferred == "正常":
                normal_count += 1

    status_text: Optional[str]
    if anomaly_count > 0:
        status_text = "异常"
    elif normalized_labels:
        status_text = "正常"
    else:
        status_text = None

    return normalized_labels, anomaly_count, normal_count, status_text


def compute_risk_components(
    scores: Iterable[float],
    vote_ratio: Iterable[float],
    threshold: float,
    vote_threshold: float,
    score_std: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine anomaly scores and vote ratios into a composite risk estimate."""

    scores_arr = np.asarray(list(scores), dtype=np.float64)
    votes_arr = np.asarray(list(vote_ratio), dtype=np.float64)

    if scores_arr.shape != votes_arr.shape:
        raise ValueError("Scores and vote ratios must have the same shape")

    scale = float(score_std) if score_std not in (None, 0) else 1.0

    score_component = (scores_arr - float(threshold)) / scale
    score_component = np.clip(score_component, 0.0, None)

    vote_component = votes_arr - float(vote_threshold)
    vote_component = np.clip(vote_component, 0.0, None)

    risk_score = 0.6 * score_component + 0.4 * vote_component
    return risk_score, score_component, vote_component