"""Training and inference helpers for tree-based PCAP flow classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
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

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "MODEL_SCHEMA_VERSION",
    "META_COLUMNS",
    "TrainingSummary",
    "DetectionResult",
    "write_metadata",
    "ModelTrainer",
    "ModelPredictor",
    "train_hist_gradient_boosting",
    "train_unsupervised_on_split",
    "detect_pcap_with_model",
    "summarize_prediction_labels",
    "compute_risk_components",
]


_PARAM_CACHE: Dict[type, Set[str]] = {}


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
    model: HistGradientBoostingClassifier,
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

    stratify: Optional[np.ndarray]
    try:
        unique = np.unique(y)
    except TypeError:
        unique = np.unique(y.astype(str))  # type: ignore[attr-defined]
    use_stratify = len(unique) > 1
    if use_stratify:
        stratify = y
    else:
        stratify = None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=int(params.get("random_state", 1337) or 1337),
            stratify=stratify,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = X, np.empty((0, X.shape[1])), y, np.empty(0)

    class_weights: Dict[str, float] = {}
    class_weight_values: Optional[Dict[Union[int, str], float]] = None
    try:
        unique_full, counts_full = np.unique(y, return_counts=True)
    except TypeError:
        unique_full = np.unique(y.astype(str))  # type: ignore[attr-defined]
        counts_full = np.ones_like(unique_full, dtype=float)

    if len(unique_full) > 1:
        class_weight_values = {
            cls: 1.0 / float(cnt) for cls, cnt in zip(unique_full, counts_full)
        }
        class_weights = {str(cls): float(weight) for cls, weight in class_weight_values.items()}

    sample_weight_train = None
    if class_weight_values is not None:
        sample_weight_train = np.asarray([class_weight_values[val] for val in y_train], dtype=float)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X_train, y_train, sample_weight=sample_weight_train)

    metrics_payload: Dict[str, float] = {}
    decision_threshold: Optional[float] = None
    positive_label: Optional[str] = None
    positive_class: Optional[Union[int, str]] = None

    if X_val.size and y_val.size:
        scores_val, pos_info = _predict_positive_scores(clf, X_val, label_mapping)
        positive_label = pos_info.label
        positive_class = pos_info.value
        if pos_info.value is not None:
            y_val_binary = (y_val == pos_info.value).astype(int)
        else:
            y_val_binary = (y_val == clf.classes_[-1]).astype(int)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_val_binary,
            scores_val,
        )
        beta = 0.5
        if pr_thresholds.size:
            precision_vals = precision_curve[:-1]
            recall_vals = recall_curve[:-1]
            numerator = (1 + beta**2) * precision_vals * recall_vals
            denominator = (beta**2 * precision_vals) + recall_vals
            with np.errstate(divide="ignore", invalid="ignore"):
                f_beta_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            best_idx = int(np.argmax(f_beta_scores))
            decision_threshold = float(pr_thresholds[best_idx])
            metrics_payload["f0_5"] = float(f_beta_scores[best_idx])
        else:
            decision_threshold = 0.5
            metrics_payload["f0_5"] = 0.0

        pred_binary = (scores_val >= float(decision_threshold)).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_binary,
            pred_binary,
            average="binary",
            zero_division=0,
        )
        metrics_payload.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

        try:
            roc_auc = roc_auc_score(y_val_binary, scores_val)
            metrics_payload["roc_auc"] = float(roc_auc)
        except ValueError:
            pass

        try:
            pr_auc = average_precision_score(y_val_binary, scores_val)
            metrics_payload["pr_auc"] = float(pr_auc)
        except ValueError:
            pass

        try:
            tn, fp, fn, tp = confusion_matrix(y_val_binary, pred_binary).ravel()
            metrics_payload.update(
                {
                    "tp": float(tp),
                    "fp": float(fp),
                    "tn": float(tn),
                    "fn": float(fn),
                }
            )
        except ValueError:
            pass

        metrics_payload["validation_samples"] = int(len(y_val_binary))
        metrics_payload["positive_samples"] = int(y_val_binary.sum())
    else:
        decision_threshold = 0.5
        if label_mapping:
            positive_label = next(iter(label_mapping.values()), None)

    if positive_class is None or positive_label is None:
        info = _resolve_positive_class(clf, label_mapping)
        if positive_class is None:
            positive_class = info.value
        if positive_label is None:
            positive_label = info.label

    if class_weight_values is not None:
        full_weights = np.asarray([class_weight_values[val] for val in y], dtype=float)
        clf.fit(X, y, sample_weight=full_weights)
    else:
        clf.fit(X, y)

    artifact = {
        "model": clf,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
    }
    dump(artifact, model_path)
    try:
        model_params = clf.get_params(deep=False)
    except Exception:  # pragma: no cover - defensive
        model_params = params
    write_metadata(model_path, feature_names, label_mapping, model_params)

    _write_model_metadata(
        model_path,
        feature_names=feature_names,
        label_mapping=label_mapping,
        decision_threshold=decision_threshold,
        positive_label=positive_label,
        positive_class=positive_class,
        metrics=metrics_payload,
        class_weights=class_weights,
    )

    if label_mapping:
        classes_display = [label_mapping.get(int(cls), str(cls)) for cls in clf.classes_]
    else:
        classes_display = [str(cls) for cls in clf.classes_]

    return TrainingSummary(
        model_path=Path(model_path),
        classes=classes_display,
        feature_names=feature_names,
        flow_count=X.shape[0],
        label_mapping=label_mapping,
        dropped_flows=int(stats.dropped_rows),
        decision_threshold=decision_threshold,
        positive_label=positive_label,
        positive_class=positive_class,
        metrics=metrics_payload or None,
        class_weights=class_weights or None,
        validation_samples=int(len(y_val)) if X_val.size else None,
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

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return metadata_path


def _load_model_artifact(
    model_path: Union[str, Path]
) -> Tuple[HistGradientBoostingClassifier, List[str], Optional[Dict[int, str]]]:
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
    model: HistGradientBoostingClassifier,
    feature_names: List[str],
    label_mapping: Optional[Dict[int, str]],
    flows: List[Dict[str, object]],
    path: Union[str, Path],
) -> DetectionResult:
    vectorized: VectorizationResult = vectorize_flows(
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
    scores, pos_info = _predict_positive_scores(model, X, label_mapping)

    predictions = model.predict(X)
    prediction_labels: List[str] = []
    if label_mapping:
        for value in predictions:
            prediction_labels.append(label_mapping.get(int(value), str(value)))
    else:
        prediction_labels = [str(value) for value in predictions]

    status_labels, _, _, _ = summarize_prediction_labels(predictions, label_mapping)

    annotated_flows: List[Dict[str, object]] = []
    for flow, label, label_name, score, status in zip(
        flows, predictions, prediction_labels, scores, status_labels
    ):
        annotated = dict(flow)
        annotated["prediction"] = int(label)
        annotated["prediction_label"] = label_name
        annotated["malicious_score"] = float(score)
        if status in {"异常", "正常"}:
            annotated["prediction_status"] = status
        annotated_flows.append(annotated)

    return DetectionResult(
        path=Path(path),
        success=True,
        flow_count=len(flows),
        feature_names=feature_names,
        predictions=[int(value) for value in predictions],
        scores=[float(value) for value in scores],
        flows=annotated_flows,
        prediction_labels=prediction_labels,
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
        model_kwargs = self._filter_params(**kwargs)
        return train_hist_gradient_boosting(dataset_path, model_path, **model_kwargs)

    def train_from_split(
        self,
        input_path: Union[str, Path],
        results_dir: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        **kwargs: Union[int, float, None],
    ) -> Dict[str, object]:
        dataset_path = _resolve_dataset_path(input_path)
        models_root = Path(models_dir) if models_dir else dataset_path.parent
        models_root.mkdir(parents=True, exist_ok=True)

        model_path = models_root / "model.joblib"

        summary = self.train(dataset_path, model_path, **kwargs)

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        stamp_token = timestamp.strftime("%Y%m%d_%H%M%S")

        numeric_names = numeric_feature_names()

        model_metadata_path = model_path.with_name("metadata.json")

        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "timestamp": timestamp_str,
            "pipeline_latest": model_path.name,
            "pipeline_path": model_path.name,
            "feature_order": list(numeric_names),
            "feature_names_in": list(numeric_names),
            "feature_columns": summary.feature_names,
            "label_mapping": summary.label_mapping or {},
            "contamination": kwargs.get("contamination"),
            "training_anomaly_ratio": kwargs.get("training_anomaly_ratio"),
            "model_metadata_path": str(model_metadata_path),
        }

        metadata["score_column"] = "malicious_score"
        if summary.decision_threshold is not None:
            metadata["threshold"] = float(summary.decision_threshold)
            metadata["decision_threshold"] = float(summary.decision_threshold)
        if summary.positive_label:
            metadata["positive_label"] = summary.positive_label
        if summary.positive_class is not None:
            if isinstance(summary.positive_class, (np.integer, int)):
                metadata["positive_class"] = int(summary.positive_class)
            else:
                metadata["positive_class"] = str(summary.positive_class)
        if summary.metrics:
            metadata["model_metrics"] = summary.metrics
        if summary.class_weights:
            metadata["class_weights"] = summary.class_weights
        if summary.validation_samples is not None:
            metadata["validation_samples"] = int(summary.validation_samples)

        metadata_path = models_root / f"iforest_metadata_{stamp_token}.json"
        latest_metadata_path = models_root / "latest_iforest_metadata.json"

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        with latest_metadata_path.open("w", encoding="utf-8") as handle:
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
            "flows": summary.flow_count,
            "malicious": 0,
            "feature_columns": summary.feature_names,
            "classes": summary.classes,
            "label_mapping": summary.label_mapping,
            "dropped_flows": summary.dropped_flows,
            "timestamp": timestamp_str,
            "schema_version": MODEL_SCHEMA_VERSION,
            "summary": summary,
            "metadata": metadata,
            "decision_threshold": summary.decision_threshold,
            "model_metrics": summary.metrics,
            "positive_label": summary.positive_label,
            "positive_class": (
                int(summary.positive_class)
                if isinstance(summary.positive_class, (np.integer, int))
                else summary.positive_class
            ),
        }


class ModelPredictor:
    """Convenience wrapper to load a trained model and perform inference."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        self.model, self.feature_names, self.label_mapping = _load_model_artifact(
            self.model_path
        )

    def predict_matrix(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = self.model
        predictions = model.predict(matrix)

        scores, _ = _predict_positive_scores(model, matrix, self.label_mapping)

        return predictions, np.asarray(scores, dtype=np.float64)

    def predict_flows(self, flows: Iterable[Dict[str, object]], *, path: Union[str, Path] = "memory") -> DetectionResult:
        flow_list = [dict(flow) for flow in flows]
        return _build_detection_result(
            self.model,
            self.feature_names,
            self.label_mapping,
            flow_list,
            path,
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


def train_unsupervised_on_split(
    input_path: Union[str, Path],
    results_dir: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    **kwargs: Union[int, float, None],
) -> Dict[str, object]:
    """Backwards-compatible entry point preserved for the UI/CLI."""

    trainer = ModelTrainer()
    return trainer.train_from_split(
        input_path,
        results_dir=results_dir,
        models_dir=models_dir,
        **kwargs,
    )


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
            elif inferred == "正":
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