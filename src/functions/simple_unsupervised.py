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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.functions.logging_utils import get_logger, log_training_run

logger = get_logger(__name__)


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


def _read_numeric_csv(path: str) -> Tuple[List[str], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise RuntimeError("训练数据为空。") from exc

        columns = [name.strip() for name in header if name.strip()]
        if not columns:
            raise RuntimeError("未发现可用于训练的特征列。")

        rows: List[List[float]] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) < len(columns):
                logger.debug("Skipping row %d with insufficient columns", line_no)
                continue
            try:
                numeric = [float(cell) for cell in row[: len(columns)]]
            except ValueError:
                logger.debug("Skipping non-numeric row %d", line_no)
                continue
            rows.append(numeric)

    if not rows:
        raise RuntimeError("训练数据为空，无法继续。")

    if len(rows) < 10:
        logger.warning(
            "检测到样本量仅 %d 条，建议至少提供 10 条记录以获得稳定阈值。继续训练以便调试。",
            len(rows),
        )

    return columns, rows


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
        for row, score, vote, rk, pred in zip(rows, scores, votes, risk, predictions):
            writer.writerow(list(row) + [
                f"{score:.6f}",
                f"{vote:.6f}",
                f"{rk:.6f}",
                pred,
                1 if pred == -1 else 0,
            ])


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
) -> Dict[str, object]:
    dataset_path = _resolve_dataset_path(split_dir)
    _ensure_dirs(results_dir, models_dir)

    columns, rows = _read_numeric_csv(dataset_path)
    column_values = list(zip(*rows))

    means = {name: _population_mean(vals) for name, vals in zip(columns, column_values)}
    stds = {name: _population_std(vals, means[name]) for name, vals in zip(columns, column_values)}

    model = _SimpleModel(
        columns=list(columns),
        means=means,
        stds=stds,
        threshold=0.0,
        contamination=float(min(max(contamination, 0.001), 0.4)),
    )

    scores = [model.score_row(row) for row in rows]
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

    _export_results(results_csv, columns, rows, scores, votes, risk, predictions)

    metadata = {
        "type": "simple_zscore",
        "columns": columns,
        "means": means,
        "stds": stds,
        "threshold": threshold,
        "vote_threshold": model.vote_threshold,
        "score_std": score_std,
        "contamination": model.contamination,
        "samples": len(rows),
        "anomaly_count": int(sum(1 for pred in predictions if pred == -1)),
    }
    _dump_json(metadata_path, metadata)
    _dump_json(model_path, metadata)

    log_training_run(
        {
            "strategy": "simple_zscore",
            "dataset": dataset_path,
            "samples": len(rows),
            "features": len(columns),
            "contamination": model.contamination,
        }
    )

    if progress_cb:
        progress_cb(100)

    return {
        "results_csv": results_csv,
        "metadata_path": metadata_path,
        "pipeline_path": model_path,
        "packets": len(rows),
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
    columns, rows = _read_numeric_csv(feature_csv)
    missing = [name for name in model.columns if name not in columns]
    if missing:
        raise RuntimeError("特征 CSV 缺少训练时的列: " + ", ".join(missing))

    column_indices = [columns.index(name) for name in model.columns]
    aligned_rows = [[row[idx] for idx in column_indices] for row in rows]

    scores = [model.score_row(row) for row in aligned_rows]
    predictions = [-1 if score <= model.threshold else 1 for score in scores]
    votes = [1.0 if pred == -1 else 0.0 for pred in predictions]
    score_std = _population_std(scores)
    risk, score_component, vote_component = compute_risk_components(
        scores, votes, model.threshold, model.vote_threshold, score_std
    )

    if output_path is None:
        base = os.path.splitext(feature_csv)[0]
        output_path = f"{base}_predictions.csv"

    _export_results(output_path, model.columns, aligned_rows, scores, votes, risk, predictions)

    details = {
        "threshold": model.threshold,
        "vote_threshold": model.vote_threshold,
        "score_std": score_std,
        "risk_score": risk,
        "score_component": score_component,
        "vote_component": vote_component,
    }
    return output_path, details