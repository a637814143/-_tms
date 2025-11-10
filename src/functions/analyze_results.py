"""Lightweight analysis utilities for anomaly detection outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import matplotlib

# 使用无界面的后端，避免在线程中绘图触发 GUI 报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configuration import get_path
from src.functions.logging_utils import get_logger

logger = get_logger(__name__)

# ---- 基础配置 --------------------------------------------------------------

SCORE_CANDIDATES: Sequence[str] = (
    "malicious_score",
    "anomaly_score",
    "score",
    "iforest_score",
)
LABEL_CANDIDATES: Sequence[str] = (
    "prediction_status",
    "prediction_label",
    "is_malicious",
    "is_anomaly",
    "predicted_label",
    "label",
    "anomaly",
)
FILE_CANDIDATES: Sequence[str] = (
    "pcap_file",
    "source_file",
    "__source_file__",
    "__source_path__",
)
ID_CANDIDATES: Sequence[str] = ("flow_id", "packet_id", "row_id", "id")
OPTIONAL_META_COLUMNS: Sequence[str] = (
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "timestamp",
    "capture_time",
    "attack",
    "category",
    "threat",
    "severity",
)
DEFAULT_TOP_PERCENT = 0.01
MAX_TOP_ROWS = 2000
TOP_REASON_COUNT = 5
CHUNK_SIZE = 50000

try:  # pragma: no cover - optional dependency
    import pyarrow  # type: ignore  # noqa: F401

    _DEFAULT_ENGINE = "pyarrow"
except Exception:  # pragma: no cover - pyarrow is optional
    _DEFAULT_ENGINE = "c"


# ---- 辅助函数 --------------------------------------------------------------


def _resolve_data_base() -> str:
    env = os.getenv("MALDET_DATA_DIR")
    if env and env.strip():
        try:
            base = Path(env).expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            return str(base)
        except Exception:  # pragma: no cover - fallback safety
            pass
    try:
        return str(get_path("data_dir"))
    except Exception:  # pragma: no cover - configuration fallback
        fallback = Path.home() / "maldet_data"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)


def _load_metadata(metadata: Optional[Dict[str, object]], metadata_path: Optional[str]) -> Dict[str, object]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception:  # pragma: no cover - best effort load
            logger.warning("Failed to load metadata from %s", metadata_path)
    base_dir = _resolve_data_base()
    candidate = os.path.join(base_dir, "models", "latest_iforest_metadata.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.warning("Failed to load fallback metadata at %s", candidate)
    return {}


def _read_columns(path: str, engine: str) -> List[str]:
    try:
        df = pd.read_csv(path, nrows=0, engine=engine)
    except Exception:
        df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def _select_first(candidates: Sequence[str], columns: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _prepare_meta_columns(
    available: Sequence[str],
    score_col: str,
    label_col: Optional[str],
    file_col: Optional[str],
    id_col: Optional[str],
) -> List[str]:
    cols: List[str] = [score_col]
    if label_col:
        cols.append(label_col)
    if file_col:
        cols.append(file_col)
    if id_col:
        cols.append(id_col)
    for name in OPTIONAL_META_COLUMNS:
        if name in available:
            cols.append(name)
    # 保留列顺序但去重
    seen: Set[str] = set()
    ordered: List[str] = []
    for col in cols:
        if col not in seen and col in available:
            ordered.append(col)
            seen.add(col)
    return ordered


def _read_meta_frame(path: str, usecols: Sequence[str], engine: str) -> pd.DataFrame:
    kwargs = {"usecols": list(usecols), "engine": engine}
    try:
        df = pd.read_csv(path, **kwargs)
    except Exception:
        kwargs.pop("engine", None)
        df = pd.read_csv(path, **kwargs)
    return df


def _ensure_float32(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.astype("float32", copy=False)


def _normalize_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return (numeric.fillna(0.0) > 0).astype("int32", copy=False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    true_tokens = {"1", "true", "yes", "attack", "anomaly", "malicious", "恶意", "异常"}
    values = normalized.isin(true_tokens).astype("int32", copy=False)
    return values


def _export_summary_by_file(
    df: pd.DataFrame,
    file_col: str,
    label_col: Optional[str],
    score_col: str,
    out_dir: str,
) -> str:
    grouped = df.groupby(file_col, dropna=False)
    payload = grouped[score_col].agg(["count", "mean", "min", "max"]).rename(
        columns={"count": "total"}
    )
    if label_col and label_col in df.columns:
        payload["malicious_count"] = grouped[label_col].sum()
        payload["malicious_ratio"] = payload["malicious_count"] / payload["total"].clip(lower=1)
    path = os.path.join(out_dir, "summary_by_file.csv")
    payload.reset_index().to_csv(path, index=False, encoding="utf-8")
    return path


def _plot_hist(scores: pd.Series, out_dir: str, *, descending: bool) -> str:
    plt.figure(figsize=(10, 6))
    plt.hist(scores.dropna().to_numpy(dtype=float), bins=50, color="#4C72B0", edgecolor="white")
    if descending:
        plt.xlabel("Anomaly score (higher = more anomalous)")
    else:
        plt.xlabel("Anomaly score (lower = more anomalous)")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution")
    plt.tight_layout()
    path = os.path.join(out_dir, "anomaly_score_distribution.png")
    plt.savefig(path)
    plt.close()
    return path


def _collect_top_feature_rows(
    path: str,
    id_col: str,
    ids: Iterable[object],
    feature_cols: Sequence[str],
) -> Optional[pd.DataFrame]:
    id_values = {val for val in ids if pd.notna(val)}
    if not id_values or not feature_cols:
        return None
    usecols = [id_col, *feature_cols]
    dtype_map = {col: "float32" for col in feature_cols}
    rows: List[pd.DataFrame] = []
    try:
        reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, dtype=dtype_map)
    except Exception:
        reader = pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE)
    for chunk in reader:
        mask = chunk[id_col].isin(id_values)
        if not mask.any():
            continue
        sub = chunk.loc[mask, usecols].copy()
        for col in feature_cols:
            if col in sub.columns:
                sub[col] = _ensure_float32(sub[col])
        rows.append(sub)
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=[id_col], keep="first")
    return combined


def _compute_top_reasons(df: pd.DataFrame, feature_cols: Sequence[str], k: int = TOP_REASON_COUNT) -> pd.Series:
    if not feature_cols:
        return pd.Series([], dtype=object)
    feature_frame = df.loc[:, feature_cols].copy()
    feature_frame = feature_frame.fillna(0.0).astype("float32", copy=False)
    if feature_frame.empty:
        return pd.Series([], dtype=object)
    values = feature_frame.to_numpy(dtype=np.float32, copy=False)
    mu = values.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)
    std = values.std(axis=0, dtype=np.float64)
    std = np.where(std < 1e-6, 1e-6, std).astype(np.float32, copy=False)
    z_scores = np.abs((values - mu) / std)
    reasons: List[str] = []
    feat_arr = np.asarray(feature_cols)
    for row in z_scores:
        if not np.isfinite(row).any():
            reasons.append("")
            continue
        count = min(k, len(row))
        if count == 0:
            reasons.append("")
            continue
        top_idx = np.argpartition(row, -count)[-count:]
        sorted_idx = top_idx[np.argsort(-row[top_idx])]
        parts = [f"{feat_arr[i]}:{row[i]:.2f}" for i in sorted_idx if row[i] > 0]
        reasons.append(", ".join(parts))
    return pd.Series(reasons, dtype=object)


def _to_float_dict(payload: Dict[object, object]) -> Dict[float, float]:
    result: Dict[float, float] = {}
    for key, value in payload.items():
        try:
            key_float = float(key)
            result[key_float] = float(value)
        except (TypeError, ValueError):
            continue
    return result


# ---- 主函数 ---------------------------------------------------------------


def analyze_results(
    results_csv: str,
    out_dir: str,
    *,
    metadata_path: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    progress_cb=None,
) -> dict:
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"找不到结果文件: {results_csv}")

    os.makedirs(out_dir, exist_ok=True)
    if progress_cb:
        progress_cb(5)

    meta_payload = _load_metadata(metadata, metadata_path)

    engine = _DEFAULT_ENGINE
    available_columns = _read_columns(results_csv, engine)

    score_col = _select_first(SCORE_CANDIDATES, available_columns)
    if not score_col:
        raise ValueError("结果文件缺少 anomaly_score/score 列")

    label_col = _select_first(LABEL_CANDIDATES, available_columns)
    file_col = _select_first(FILE_CANDIDATES, available_columns)
    id_col = _select_first(ID_CANDIDATES, available_columns)

    meta_cols = _prepare_meta_columns(available_columns, score_col, label_col, file_col, id_col)
    df_meta = _read_meta_frame(results_csv, meta_cols, engine)

    df_meta[score_col] = _ensure_float32(df_meta[score_col])
    for col in meta_cols:
        if col == score_col or col not in df_meta.columns:
            continue
        if df_meta[col].dtype.kind in {"i", "u", "f"}:
            df_meta[col] = _ensure_float32(df_meta[col])
    df_meta = df_meta.replace([np.inf, -np.inf], np.nan)
    df_meta = df_meta.dropna(subset=[score_col])

    if label_col and label_col in df_meta.columns:
        df_meta[label_col] = _normalize_label(df_meta[label_col])
    else:
        label_col = None

    total_rows = int(len(df_meta))
    if total_rows == 0:
        raise ValueError("结果文件为空或缺少有效分数列")

    if progress_cb:
        progress_cb(20)

    summary_csv_path = None
    unique_files = 0
    malicious_total = 0
    malicious_ratio = 0.0
    if file_col:
        summary_csv_path = _export_summary_by_file(df_meta, file_col, label_col, score_col, out_dir)
        unique_files = int(df_meta[file_col].nunique())
    if label_col:
        malicious_total = int(df_meta[label_col].sum())
        malicious_ratio = float(malicious_total / total_rows)

    score_column_meta = ""
    if isinstance(meta_payload, dict):
        score_column_meta = str(meta_payload.get("score_column", "")).strip().lower()
    descending_scores = score_col.lower() in {"malicious_score", "prob_malicious", "proba_malicious"}
    if score_column_meta in {"malicious_score", "prob_malicious", "proba_malicious"}:
        descending_scores = True

    hist_path = _plot_hist(df_meta[score_col], out_dir, descending=descending_scores)

    meta_top_percent = meta_payload.get("refine_top_percent") if isinstance(meta_payload, dict) else None
    if isinstance(meta_top_percent, (int, float)) and meta_top_percent > 0:
        target_percent = float(meta_top_percent)
    else:
        target_percent = DEFAULT_TOP_PERCENT * 100
    top_fraction = max(target_percent / 100.0, DEFAULT_TOP_PERCENT)
    top_n = int(max(1, total_rows * top_fraction))
    if total_rows >= 20:
        top_n = max(20, top_n)
    top_n = min(top_n, MAX_TOP_ROWS, total_rows)
    if descending_scores:
        top_meta = df_meta.nlargest(top_n, score_col).copy()
    else:
        top_meta = df_meta.nsmallest(top_n, score_col).copy()

    feature_columns_meta = meta_payload.get("feature_columns") if isinstance(meta_payload, dict) else None
    feature_cols: List[str] = []
    if isinstance(feature_columns_meta, list):
        feature_cols = [col for col in feature_columns_meta if col in available_columns and col not in meta_cols]
        if len(feature_cols) > 256:
            feature_cols = feature_cols[:256]

    top_reasons_series = None
    if id_col and feature_cols:
        feature_rows = _collect_top_feature_rows(results_csv, id_col, top_meta[id_col], feature_cols)
        if feature_rows is not None:
            merged = top_meta.merge(feature_rows, on=id_col, how="left", suffixes=("", "_feat"))
            # Remove duplicate columns potentially introduced by merge
            dup_cols = [f"{col}_feat" for col in feature_cols if f"{col}_feat" in merged.columns]
            if dup_cols:
                merged.drop(columns=dup_cols, inplace=True)
            top_meta = merged
            top_reasons_series = _compute_top_reasons(top_meta, feature_cols)
            if not top_reasons_series.empty:
                top_meta["top_reasons"] = top_reasons_series.values
    if "top_reasons" not in top_meta.columns:
        top_meta["top_reasons"] = ""

    export_cols = [score_col]
    for col in (label_col, file_col, id_col):
        if col and col in top_meta.columns:
            export_cols.append(col)
    for name in OPTIONAL_META_COLUMNS:
        if name in top_meta.columns:
            export_cols.append(name)
    export_cols.append("top_reasons")
    export_cols = [col for col in export_cols if col in top_meta.columns]
    top_csv_base = "top20_packets.csv"
    top_csv_path = os.path.join(out_dir, top_csv_base)
    try:
        top_meta.loc[:, export_cols].to_csv(top_csv_path, index=False, encoding="utf-8")
    except PermissionError:
        alt_name = f"top20_packets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        top_csv_path = os.path.join(out_dir, alt_name)
        top_meta.loc[:, export_cols].to_csv(top_csv_path, index=False, encoding="utf-8")

    quantile_points = [0.01, 0.05, 0.5, 0.9, 0.95, 0.99]
    score_quantiles = df_meta[score_col].quantile(quantile_points).to_dict()
    train_threshold_val = meta_payload.get("threshold") if isinstance(meta_payload, dict) else None
    if isinstance(train_threshold_val, (int, float)):
        score_threshold = float(train_threshold_val)
    else:
        if descending_scores:
            score_threshold = float(
                score_quantiles.get(0.95, float(df_meta[score_col].quantile(0.95)))
            )
        else:
            score_threshold = float(score_quantiles.get(0.05, float(df_meta[score_col].min())))
    ratio_threshold = float(meta_payload.get("ratio_threshold", 0.05)) if isinstance(meta_payload, dict) else 0.05

    drift_alerts = None
    train_quantiles_raw = meta_payload.get("score_quantiles") if isinstance(meta_payload, dict) else None
    if isinstance(train_quantiles_raw, dict):
        try:
            train_quantiles = _to_float_dict(train_quantiles_raw)
            current_quantiles = _to_float_dict(
                df_meta[score_col].quantile([0.01, 0.05, 0.5, 0.9]).to_dict()
            )
            alerts = {}
            for key in (0.01, 0.05, 0.5, 0.9):
                if key not in train_quantiles or key not in current_quantiles:
                    continue
                base = train_quantiles[key]
                current = current_quantiles[key]
                drift = abs(current - base) / max(abs(base), 1e-6)
                if drift > 0.15:
                    alerts[f"{key:.2f}".rstrip("0").rstrip(".")] = float(drift)
            drift_alerts = alerts or None
        except Exception:
            drift_alerts = None

    summary_lines = [
        f"总检测样本数：{total_rows}",
        f"Top {top_n} 异常样本已导出至 {os.path.basename(top_csv_path)}",
    ]
    if unique_files:
        summary_lines.append(f"涉及文件数：{unique_files}")
    if label_col:
        summary_lines.append(f"异常样本数量：{malicious_total} ({malicious_ratio:.2%})")
    if descending_scores:
        summary_lines.append(f"建议异常得分阈值（越大越异常）：≥ {score_threshold:.6f}")
    else:
        summary_lines.append(f"建议异常得分阈值（越小越异常）：≤ {score_threshold:.6f}")
    summary_lines.append(f"文件级判定阈值：恶意占比 ≥ {ratio_threshold:.2%}")
    if drift_alerts:
        summary_lines.append("⚠ 分布漂移告警：" + json.dumps(drift_alerts, ensure_ascii=False))
    summary_text = "\n".join(summary_lines)

    details_payload = {
        "total_rows": total_rows,
        "unique_files": unique_files,
        "malicious_total": malicious_total,
        "malicious_ratio": malicious_ratio,
        "score_threshold": score_threshold,
        "ratio_threshold": ratio_threshold,
        "anomaly_score_quantiles": {str(k): float(v) for k, v in score_quantiles.items()},
        "summary_csv": summary_csv_path,
        "train_threshold": float(train_threshold_val) if isinstance(train_threshold_val, (int, float)) else None,
        "score_quantiles_train": train_quantiles_raw,
        "drift_alerts": drift_alerts,
        "top_n": top_n,
        "top_csv": top_csv_path,
        "model_metrics": None,
        "roc_auc": None,
        "pr_auc": None,
        "score_column": score_col,
        "score_direction": "descending" if descending_scores else "ascending",
    }

    export_payload = {
        "anomaly_count": malicious_total,
        "anomaly_files": [],
        "single_file_status": None,
        "status_text": "异常" if malicious_total > 0 else "正常",
    }
    if file_col and file_col in top_meta.columns:
        anomaly_files = [
            str(value)
            for value in top_meta[file_col].dropna().astype(str).unique().tolist()[:10]
        ]
        export_payload["anomaly_files"] = anomaly_files

    summary_json_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary_text, "details": details_payload}, fh, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100)

    logger.info(
        "Analysis completed: rows=%d anomalies=%d top_n=%d", total_rows, malicious_total, top_n
    )

    payload = {
        "plots": [hist_path],
        "top20_csv": top_csv_path,
        "summary_csv": summary_csv_path,
        "summary_json": summary_json_path,
        "summary_text": summary_text,
        "metrics": details_payload,
        "out_dir": out_dir,
        "export_payload": export_payload,
        "hist_compare_plot": None,
        "metrics_csv": None,
        "metrics_json": summary_json_path,
    }
    return payload


__all__ = ["analyze_results"]