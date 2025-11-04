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
    "ModelTrainer",
    "ModelPredictor",
    "train_hist_gradient_boosting",
    "train_unsupervised_on_split",
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

    @staticmethod
    def _filter_params(**kwargs: Union[int, float, None]) -> Dict[str, Union[int, float, None]]:
        return {key: value for key, value in kwargs.items() if value is not None}

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

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(matrix)
            scores = proba.max(axis=1)
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(matrix)
            if np.ndim(decision) == 1:
                scores = 1.0 / (1.0 + np.exp(-decision))
            else:
                scores = decision.max(axis=1)
        else:
            scores = predictions

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