"""Lightweight fallbacks for simple JSON-based anomaly models."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

DEFAULT_SCORE_COLUMN = "malicious_score"
DEFAULT_THRESHOLD = 0.5
_POSITIVE_LABELS = {"1", "-1", "异常", "abnormal", "malicious", "anomaly", "threat"}
_NEGATIVE_LABELS = {"0", "正常", "benign", "normal", "clean"}


@dataclass(frozen=True)
class SimpleModel:
    """Minimal representation of the exported JSON metadata."""

    data: Dict[str, Any]

    @property
    def score_column(self) -> str:
        column = self.data.get("score_column") or self.data.get("score_col")
        if isinstance(column, str) and column.strip():
            return column.strip()
        return DEFAULT_SCORE_COLUMN

    @property
    def threshold(self) -> float:
        for key in ("decision_threshold", "threshold", "score_threshold"):
            value = self.data.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return DEFAULT_THRESHOLD

    @property
    def positive_label(self) -> Optional[str]:
        value = self.data.get("positive_label") or self.data.get("positive_class")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @property
    def label_mapping(self) -> Dict[str, str]:
        mapping_raw = self.data.get("label_mapping")
        if isinstance(mapping_raw, dict):
            return {str(key): str(val) for key, val in mapping_raw.items()}
        return {}


def _normalize_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    return candidate


def load_simple_model(path: Union[str, Path]) -> SimpleModel:
    """Load the lightweight JSON pipeline metadata."""

    path_obj = _normalize_path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"未找到轻量模型文件: {path_obj}")
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("简化模型文件格式错误，应为 JSON 对象。")
    return SimpleModel(payload)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _normalize_label(value: Any, *, score: float, model: SimpleModel) -> str:
    if value is not None and value != "":
        text = str(value).strip()
        if text:
            mapped = model.label_mapping.get(text)
            if mapped:
                text = mapped
            lower = text.lower()
            positive_hint = model.positive_label
            if positive_hint and text.lower() == positive_hint.lower():
                return "异常"
            if lower in _POSITIVE_LABELS:
                return "异常"
            if lower in _NEGATIVE_LABELS:
                return "正常"
            try:
                as_int = int(text)
            except (TypeError, ValueError):
                pass
            else:
                mapped = model.label_mapping.get(str(as_int))
                if mapped:
                    return mapped
                if as_int in (1, -1):
                    return "异常"
                if as_int == 0:
                    return "正常"
            if text in {"异常", "正常"}:
                return text
            if positive_hint and str(text).lower() == positive_hint.lower():
                return "异常"
    return "异常" if score >= model.threshold else "正常"


def _resolve_output_path(source_csv: Union[str, Path], output_path: Optional[Union[str, Path]]) -> Path:
    if output_path:
        return _normalize_path(output_path)
    source_obj = _normalize_path(source_csv)
    return source_obj.with_name(f"{source_obj.stem}_predictions.csv")


def simple_predict(
    model: SimpleModel,
    feature_csv: Union[str, Path],
    *,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate predictions by thresholding the configured score column."""

    csv_path = _normalize_path(feature_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到特征 CSV: {csv_path}")

    out_path = _resolve_output_path(csv_path, output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    score_column = model.score_column
    if score_column not in fieldnames:
        fieldnames.append(score_column)

    extra_columns = ["prediction", "prediction_status", "is_malicious"]
    for column in extra_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    anomaly_count = 0
    normal_count = 0

    for row in rows:
        score = _safe_float(row.get(score_column))
        label_source = None
        for key in ("prediction", "predicted_label", "label", "status"):
            if row.get(key) not in (None, ""):
                label_source = row.get(key)
                break
        normalized = _normalize_label(label_source, score=score, model=model)
        if normalized == "异常":
            anomaly_count += 1
            row["is_malicious"] = "1"
            row["prediction_status"] = "1"
        else:
            normal_count += 1
            row["is_malicious"] = "0"
            row["prediction_status"] = "0"
        row["prediction"] = normalized
        row[score_column] = f"{score:.6f}"

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_path": str(out_path),
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "threshold": model.threshold,
        "score_column": score_column,
    }
    return str(out_path), summary


def train_unsupervised_on_split(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Placeholder that guides the user towards the full training pipeline."""

    raise RuntimeError(
        "当前环境未包含完整的训练依赖，请使用 src.functions.modeling.train_unsupervised_on_split。"
    )


__all__ = [
    "SimpleModel",
    "load_simple_model",
    "simple_predict",
    "train_unsupervised_on_split",
]