"""Lightweight fallback implementations for training and inference.

This module is used when optional heavy dependencies such as ``numpy`` or
``pandas`` are not available in the execution environment.  It implements a
very small anomaly detection workflow based purely on the Python standard
library so that the CLI ``train``/``predict`` commands continue to function
for smoke tests.

The detector is intentionally simple: features are z-scored using population
statistics and the anomaly score is defined as the negative L1 distance from
the centre.  Records in the contamination quantile are marked as anomalies.
Metadata is persisted so that ``predict`` can reproduce the same behaviour.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from src.functions.logging_utils import get_logger, log_training_run

logger = get_logger(__name__)


SIMPLE_MAX_FEATURES = 4_096
SIMPLE_VARIANCE_SAMPLE_ROWS = 512


@dataclass
class _SimpleModel:
    columns: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    threshold: float
    contamination: float

    @property
    def vote_threshold(self) -> float:
        # Binary votes in this simple detector -> threshold at 0.5
        return 0.5

    def score_row(self, row: Sequence[float]) -> float:
        # Negative sum of absolute z-scores so that smaller values denote
        # stronger anomalies (similar to IsolationForest behaviour).
        total = 0.0
        for name, value in zip(self.columns, row):
            mean = self.means.get(name, 0.0)
            std = self.stds.get(name, 1.0) or 1.0
            total += abs((value - mean) / std)
        return -total


def _read_csv_structure(path: str) -> Tuple[List[str], List[int]]:
    """Return the sanitised column names and indices used for extraction."""

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError("训练数据为空。") from exc

    columns: List[str] = []
    indices: List[int] = []
    for idx, raw in enumerate(header):
        name = raw.strip()
        if not name:
            continue
        columns.append(name)
        indices.append(idx)

    if not columns:
        raise RuntimeError("未发现可用于训练的特征列。")

    return columns, indices


def _iter_numeric_rows(
    path: str,
    *,
    source_indices: Sequence[int],
    limit: Optional[int] = None,
) -> Iterator[List[float]]:
    """Yield numeric rows using the provided source column indices."""

    if not source_indices:
        raise RuntimeError("未发现可用于训练的特征列。")

    max_required = max(source_indices)
    emitted = 0
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for line_no, row in enumerate(reader, start=2):
            if limit is not None and emitted >= limit:
                break
            if len(row) <= max_required:
                logger.debug("Skipping row %d with insufficient columns", line_no)
                continue
            numeric: List[float] = []
            try:
                for idx in source_indices:
                    value = float(row[idx])
                    if not math.isfinite(value):
                        value = 0.0
                    numeric.append(value)
            except ValueError:
                logger.debug("Skipping non-numeric row %d", line_no)
                continue
            emitted += 1
            yield numeric


def _select_feature_indices(
    path: str,
    columns: Sequence[str],
    base_indices: Sequence[int],
) -> Tuple[List[int], Optional[Dict[str, object]]]:
    total_columns = len(columns)
    if total_columns <= SIMPLE_MAX_FEATURES:
        return list(range(total_columns)), None

    sample_count = 0
    coverage = [0] * total_columns
    nonzero = [0] * total_columns
    means = [0.0] * total_columns
    m2 = [0.0] * total_columns

    for row in _iter_numeric_rows(
        path,
        source_indices=base_indices,
        limit=SIMPLE_VARIANCE_SAMPLE_ROWS,
    ):
        sample_count += 1
        for idx, value in enumerate(row):
            coverage[idx] += 1
            if abs(value) > 1e-12:
                nonzero[idx] += 1
            delta = value - means[idx]
            means[idx] += delta / coverage[idx]
            delta2 = value - means[idx]
            m2[idx] += delta * delta2

    if sample_count == 0:
        raise RuntimeError("训练数据为空，无法继续。")

    scores = []
    for idx in range(total_columns):
        variance = m2[idx] / max(coverage[idx], 1)
        variance = max(variance, 0.0)
        coverage_ratio = coverage[idx] / sample_count
        nonzero_ratio = nonzero[idx] / sample_count
        score = variance * (0.2 + 0.5 * coverage_ratio + 0.3 * nonzero_ratio)
        scores.append((idx, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    keep_positions = {idx for idx, _ in scores[:SIMPLE_MAX_FEATURES]}
    ordered_positions = [idx for idx in range(total_columns) if idx in keep_positions]
    if not ordered_positions:
        ordered_positions = list(range(min(SIMPLE_MAX_FEATURES, total_columns)))

    logger.warning(
        "检测到特征列数为 %d，超出回退模型的安全范围，将自动保留前 %d 列以降低内存消耗。",
        total_columns,
        len(ordered_positions),
    )

    info = {
        "original_features": total_columns,
        "kept_features": len(ordered_positions),
        "dropped_features": total_columns - len(ordered_positions),
    }
    return ordered_positions, info


def _population_mean(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return 0.0
    return sum(seq) / len(seq)


def _population_std(values: Iterable[float], mean: Optional[float] = None) -> float:
    seq = list(values)
    if not seq:
        return 1.0
    if mean is None:
        mean = _population_mean(seq)
    var = sum((val - mean) ** 2 for val in seq) / max(len(seq), 1)
    return math.sqrt(var) if var > 0 else 1.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    q = min(max(q, 0.0), 1.0)
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _dump_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _stream_export_results(
    output_path: str,
    columns: Sequence[str],
    row_iter: Iterator[List[float]],
    scores: Sequence[float],
    votes: Sequence[float],
    risk: Sequence[float],
    predictions: Sequence[int],
) -> None:
    header = list(columns) + [
        "anomaly_score",
        "vote_ratio",
        "risk_score",
        "prediction",
        "is_malicious",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, score, vote, rk, pred in zip(row_iter, scores, votes, risk, predictions):
            writer.writerow(
                list(row)
                + [
                    f"{score:.6f}",
                    f"{vote:.6f}",
                    f"{rk:.6f}",
                    pred,
                    1 if pred == -1 else 0,
                ]
            )


def compute_risk_components(
    scores: Sequence[float],
    vote_ratio: Sequence[float],
    threshold: float,
    vote_threshold: float,
    score_std: float,
) -> Tuple[List[float], List[float], List[float]]:
    score_std = float(score_std or 1.0)
    vote_threshold = float(vote_threshold or 0.5)
    result_risk: List[float] = []
    result_score: List[float] = []
    result_vote: List[float] = []
    for score, vote in zip(scores, vote_ratio):
        # Logistic shaping so that scores far below the threshold approach 1.
        z = (score - threshold) / score_std
        score_component = 1.0 / (1.0 + math.exp(z))
        vote_component = 1.0 if vote >= vote_threshold else vote / max(vote_threshold, 1e-6)
        risk = 0.6 * score_component + 0.4 * vote_component
        result_risk.append(max(0.0, min(1.0, risk)))
        result_score.append(max(0.0, min(1.0, score_component)))
        result_vote.append(max(0.0, min(1.0, vote_component)))
    return result_risk, result_score, result_vote


def _export_results(
    output_path: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[float]],
    scores: Sequence[float],
    votes: Sequence[float],
    risk: Sequence[float],
    predictions: Sequence[int],
) -> None:
    _stream_export_results(output_path, columns, iter(rows), scores, votes, risk, predictions)


def _resolve_dataset_path(split_dir: str) -> str:
    if os.path.isfile(split_dir):
        return split_dir
    if os.path.isdir(split_dir):
        for candidate in sorted(os.listdir(split_dir)):
            if candidate.lower().endswith(".csv"):
                return os.path.join(split_dir, candidate)
    raise FileNotFoundError(f"未找到可用的训练数据: {split_dir}")


def train_unsupervised_on_split(
    split_dir: str,
    results_dir: str,
    models_dir: str,
    contamination: float = 0.05,
    base_estimators: int = 50,  # unused, kept for signature compatibility
    progress_cb=None,
    workers: int = 4,
    rbf_components: Optional[int] = None,
    rbf_gamma: Optional[float] = None,
    fusion_alpha: float = 0.5,
    enable_supervised_fusion: bool = True,
    feature_selection_ratio: Optional[float] = None,
    pipeline_components: Optional[Dict[str, bool]] = None,
    memory_budget_bytes: Optional[int] = None,
    speed_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    dataset_path = _resolve_dataset_path(split_dir)
    _ensure_dirs(results_dir, models_dir)

    columns, base_indices = _read_csv_structure(dataset_path)
    keep_positions, reduction_info = _select_feature_indices(dataset_path, columns, base_indices)
    selected_columns = [columns[idx] for idx in keep_positions]
    selected_indices = [base_indices[idx] for idx in keep_positions]

    feature_count = len(selected_columns)
    means_arr = [0.0] * feature_count
    m2_arr = [0.0] * feature_count
    sample_count = 0
    for row in _iter_numeric_rows(dataset_path, source_indices=selected_indices, limit=None):
        sample_count += 1
        for idx, value in enumerate(row):
            delta = value - means_arr[idx]
            means_arr[idx] += delta / sample_count
            delta2 = value - means_arr[idx]
            m2_arr[idx] += delta * delta2

    if sample_count == 0:
        raise RuntimeError("训练数据为空，无法继续。")

    if sample_count < 10:
        logger.warning(
            "检测到样本量仅 %d 条，建议至少提供 10 条记录以获得稳定阈值。继续训练以便调试。",
            sample_count,
        )

    stds_arr = []
    for idx in range(feature_count):
        variance = m2_arr[idx] / max(sample_count, 1)
        stds_arr.append(math.sqrt(variance) if variance > 0 else 1.0)

    means_map = {name: means_arr[idx] for idx, name in enumerate(selected_columns)}
    stds_map = {name: stds_arr[idx] for idx, name in enumerate(selected_columns)}

    model = _SimpleModel(
        columns=list(selected_columns),
        means=means_map,
        stds=stds_map,
        threshold=0.0,
        contamination=float(min(max(contamination, 0.001), 0.4)),
    )

    scores: List[float] = []
    for row in _iter_numeric_rows(dataset_path, source_indices=selected_indices, limit=None):
        scores.append(model.score_row(row))
    threshold = _quantile(scores, model.contamination)
    model.threshold = threshold
    score_std = _population_std(scores)

    predictions = [-1 if score <= threshold else 1 for score in scores]
    votes = [1.0 if pred == -1 else 0.0 for pred in predictions]
    risk, score_component, vote_component = compute_risk_components(
        scores, votes, threshold, model.vote_threshold, score_std
    )

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    results_csv = os.path.join(results_dir, f"{base_name}_results.csv")
    metadata_path = os.path.join(results_dir, f"{base_name}_metadata.json")
    model_path = os.path.join(models_dir, f"{base_name}_model.json")

    export_rows = _iter_numeric_rows(dataset_path, source_indices=selected_indices, limit=None)
    _stream_export_results(
        results_csv,
        selected_columns,
        export_rows,
        scores,
        votes,
        risk,
        predictions,
    )

    metadata = {
        "type": "simple_zscore",
        "columns": selected_columns,
        "means": means_map,
        "stds": stds_map,
        "threshold": threshold,
        "vote_threshold": model.vote_threshold,
        "score_std": score_std,
        "contamination": model.contamination,
        "samples": sample_count,
        "anomaly_count": int(sum(1 for pred in predictions if pred == -1)),
    }
    if reduction_info:
        metadata["feature_reduction"] = reduction_info
    _dump_json(metadata_path, metadata)
    _dump_json(model_path, metadata)

    log_training_run(
        {
            "strategy": "simple_zscore",
            "dataset": dataset_path,
            "samples": sample_count,
            "features": len(selected_columns),
            "contamination": model.contamination,
        }
    )

    if progress_cb:
        progress_cb(100)

    return {
        "results_csv": results_csv,
        "metadata_path": metadata_path,
        "pipeline_path": model_path,
        "packets": sample_count,
        "score_component": score_component,
        "vote_component": vote_component,
    }


def load_simple_model(path: str) -> _SimpleModel:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("type") != "simple_zscore":
        raise RuntimeError("不支持的简单模型格式。")
    return _SimpleModel(
        columns=list(payload.get("columns", [])),
        means={str(k): float(v) for k, v in dict(payload.get("means", {})).items()},
        stds={str(k): float(v) for k, v in dict(payload.get("stds", {})).items()},
        threshold=float(payload.get("threshold", 0.0)),
        contamination=float(payload.get("contamination", 0.05)),
    )


def simple_predict(
    model: _SimpleModel,
    feature_csv: str,
    *,
    output_path: Optional[str] = None,
) -> Tuple[str, Dict[str, object]]:
    file_columns, base_indices = _read_csv_structure(feature_csv)
    index_map = {name: idx for name, idx in zip(file_columns, base_indices)}
    missing = [name for name in model.columns if name not in index_map]
    if missing:
        raise RuntimeError("特征 CSV 缺少训练时的列: " + ", ".join(missing))

    selected_indices = [index_map[name] for name in model.columns]

    scores: List[float] = []
    for row in _iter_numeric_rows(feature_csv, source_indices=selected_indices, limit=None):
        scores.append(model.score_row(row))

    if not scores:
        raise RuntimeError("特征 CSV 为空，无法预测。")
    predictions = [-1 if score <= model.threshold else 1 for score in scores]
    votes = [1.0 if pred == -1 else 0.0 for pred in predictions]
    score_std = _population_std(scores)
    risk, score_component, vote_component = compute_risk_components(
        scores, votes, model.threshold, model.vote_threshold, score_std
    )

    if output_path is None:
        base = os.path.splitext(feature_csv)[0]
        output_path = f"{base}_predictions.csv"

    export_iter = _iter_numeric_rows(feature_csv, source_indices=selected_indices, limit=None)
    _stream_export_results(
        output_path,
        model.columns,
        export_iter,
        scores,
        votes,
        risk,
        predictions,
    )

    details = {
        "threshold": model.threshold,
        "vote_threshold": model.vote_threshold,
        "score_std": score_std,
        "risk_score": risk,
        "score_component": score_component,
        "vote_component": vote_component,
    }
    return output_path, details