"""Training and inference helpers for tree-based PCAP flow classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier

from .static_features import extract_pcap_features
from .vectorizer import VectorizationResult, load_vectorized_dataset, vectorize_flows

__all__ = [
    "DEFAULT_MODEL_PARAMS",
    "TrainingSummary",
    "DetectionResult",
    "train_hist_gradient_boosting",
    "detect_pcap_with_model",
    "summarize_prediction_labels",
    "compute_risk_components",
]


@dataclass
class TrainingSummary:
    """Metadata describing a completed training run."""

    model_path: Path
    classes: List[str]
    feature_names: List[str]
    flow_count: int
    label_mapping: Optional[Dict[int, str]] = None
    dropped_flows: int = 0


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

    params = DEFAULT_MODEL_PARAMS.copy()
    params.update(kwargs)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X, y)

    artifact = {
        "model": clf,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
    }
    dump(artifact, model_path)

    _write_model_metadata(
        model_path,
        feature_names=feature_names,
        label_mapping=label_mapping,
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
    )


def _write_model_metadata(
    model_path: Union[str, Path],
    *,
    feature_names: Iterable[Union[str, bytes]],
    label_mapping: Optional[Dict[int, str]] = None,
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
    }

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
    if model is None or feature_names is None:
        raise ValueError("Model artifact is missing required data")
    return model, list(feature_names), label_mapping if label_mapping else None


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
    vectorized: VectorizationResult = vectorize_flows(
        flows, feature_names=feature_names, include_labels=False
    )

    if vectorized.flow_count == 0:
        return DetectionResult(
            path=Path(pcap_path),
            success=True,
            flow_count=0,
            feature_names=feature_names,
            predictions=[],
            scores=[],
            flows=[],
            prediction_labels=[],
        )

    X = vectorized.matrix
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        scores = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if decision.ndim == 1:
            scores = 1.0 / (1.0 + np.exp(-decision))
        else:
            scores = decision.max(axis=1)
    else:
        scores = model.predict(X)

    predictions = model.predict(X)
    prediction_labels: List[str] = []
    if label_mapping:
        for value in predictions:
            prediction_labels.append(label_mapping.get(int(value), str(value)))
    else:
        prediction_labels = [str(value) for value in predictions]

    annotated_flows: List[Dict[str, object]] = []
    for flow, label, label_name, score in zip(flows, predictions, prediction_labels, scores):
        annotated = dict(flow)
        annotated["prediction"] = int(label)
        annotated["prediction_label"] = label_name
        annotated["malicious_score"] = float(score)
        annotated_flows.append(annotated)

    return DetectionResult(
        path=Path(pcap_path),
        success=True,
        flow_count=len(flows),
        feature_names=feature_names,
        predictions=[int(value) for value in predictions],
        scores=[float(value) for value in scores],
        flows=annotated_flows,
        prediction_labels=prediction_labels,
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