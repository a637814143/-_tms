"""Training and inference helpers for tree-based PCAP flow classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier

from .static_features import extract_pcap_features
from .vectorizer import VectorizationResult, load_vectorized_dataset, vectorize_flows


# Columns containing metadata generated during preprocessing that should be
# retained alongside the feature matrix.
META_COLUMNS = {
    "__source_file__",
    "__source_path__",
    "pcap_file",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
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


def train_unsupervised_on_split(
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