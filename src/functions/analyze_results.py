
# src/functions/analyze_results.py

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

# 使用无界面的后端，避免在线程中绘图触发 GUI 报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from joblib import load
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.configuration import get_path
from src.functions.logging_utils import get_logger

try:  # optional dependency
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional import guard
    shap = None

logger = get_logger(__name__)

EXPLANATION_EXCLUDE_COLUMNS = {
    "anomaly_score",
    "anomaly_confidence",
    "vote_ratio",
    "risk_score",
    "is_malicious",
    "__TAG__",
    "pcap_file",
    "__source_file__",
    "__source_path__",
    "flow_id",
}

TIME_COLUMN_CANDIDATES = (
    "timestamp",
    "time",
    "frame_time",
    "frame.time",
    "capture_time",
    "flow_start",
    "flow_time",
    "start_time",
    "end_time",
    "ts",
    "epoch",
)


def _resolve_data_base() -> str:
    env = os.getenv("MALDET_DATA_DIR")
    if env and env.strip():
        try:
            base = Path(env).expanduser().resolve()
            base.mkdir(parents=True, exist_ok=True)
            return str(base)
        except Exception:
            pass
    try:
        return str(get_path("data_dir"))
    except Exception:
        fallback = Path.home() / "maldet_data"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

GROUND_TRUTH_COLUMN_CANDIDATES = (
    "label",
    "labels",
    "ground_truth",
    "attack",
    "attacks",
    "is_attack",
    "is_malicious",
    "malicious",
    "malware",
    "anomaly",
)

NORMAL_TOKENS = {
    "normal",
    "normal.",
    "benign",
    "benign.",
    "good",
    "legit",
    "legitimate",
    "clean",
    "ok",
    "allow",
    "allowed",
    "pass",
    "passed",
    "success",
    "success.",
    "successful",
    "合法",
    "良性",
    "正常",
    "正常.",
    "正常流量",
    "成功",
    "登录成功",
    "登陆成功",
    "成功登陆",
    "成功登入",
    "4399",
    "4399登录",
    "4399成功登录",
    "baidu",
    "baidu.",
    "百度",
    "百度.",
    "百度登录",
    "百度成功登录",
    "0",
    "false",
    "no",
}

ANOMALY_TOKENS = {
    "attack",
    "attack.",
    "attacks",
    "intrusion",
    "intrusion.",
    "anomaly",
    "anomaly.",
    "abnormal",
    "malicious",
    "malicious.",
    "malware",
    "botnet",
    "spam",
    "ddos",
    "dos",
    "denied",
    "blocked",
    "fail",
    "failed",
    "failure",
    "error",
    "失败",
    "登录失败",
    "登陆失败",
    "登入失败",
    "非法",
    "恶意",
    "恶意流量",
    "攻击",
    "攻击流量",
    "异常",
    "异常流量",
    "异常登录",
    "可疑",
    "可疑流量",
    "嫌疑",
    "嫌疑流量",
    "1",
    "true",
    "yes",
}


def _build_token_variants(tokens: set[str]) -> set[str]:
    variants: set[str] = set()
    strip_chars = "。.,，!！?？；;:"
    for token in tokens:
        if not token:
            continue
        base = token.strip().lower()
        collapsed = base.replace(" ", "")
        trimmed = base.strip(strip_chars)
        collapsed_trimmed = collapsed.strip(strip_chars)
        for item in (base, collapsed, trimmed, collapsed_trimmed):
            if item:
                variants.add(item)
    return variants


NORMAL_TOKEN_VARIANTS = _build_token_variants(NORMAL_TOKENS)
ANOMALY_TOKEN_VARIANTS = _build_token_variants(ANOMALY_TOKENS)


def _normalize_hist(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts / total


def _population_stability_index(base: np.ndarray, current: np.ndarray) -> float:
    base_prob = _normalize_hist(base)
    curr_prob = _normalize_hist(current)
    mask = (base_prob > 0) & (curr_prob > 0)
    if not np.any(mask):
        return 0.0
    psi = np.sum((curr_prob[mask] - base_prob[mask]) * np.log(curr_prob[mask] / base_prob[mask]))
    return float(psi)


def _kl_divergence(base: np.ndarray, current: np.ndarray) -> float:
    base_prob = _normalize_hist(base)
    curr_prob = _normalize_hist(current)
    mask = (base_prob > 0) & (curr_prob > 0)
    if not np.any(mask):
        return 0.0
    kl = np.sum(curr_prob[mask] * np.log(curr_prob[mask] / base_prob[mask]))
    return float(kl)


def _extract_time_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in TIME_COLUMN_CANDIDATES:
        if column not in df.columns:
            continue
        series = df[column]
        if series.isnull().all():
            continue
        if np.issubdtype(series.dtype, np.datetime64):
            return pd.to_datetime(series, errors="coerce")
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                if numeric.abs().max() > 1e11:
                    converted = pd.to_datetime(numeric, unit="ns", errors="coerce")
                elif numeric.abs().max() > 1e9:
                    converted = pd.to_datetime(numeric, unit="s", errors="coerce")
                else:
                    converted = pd.to_datetime(numeric, unit="s", origin="unix", errors="coerce")
                if converted.notna().any():
                    return converted
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().any():
                return parsed
        except Exception:
            continue
    return None


def _select_numeric_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    columns: List[str] = []
    exclude_set = {str(col) for col in exclude}
    for col in df.columns:
        if col in exclude_set:
            continue
        series = df[col]
        if is_numeric_dtype(series):
            columns.append(col)
            continue
        try:
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any():
                columns.append(col)
        except Exception:
            continue
    return columns


def _prepare_numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> Optional[np.ndarray]:
    if not columns:
        return None
    try:
        subset = df.loc[:, columns]
    except Exception:
        return None
    try:
        numeric = subset.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    except Exception:
        return None
    if numeric.empty:
        return None
    return numeric.to_numpy(dtype=float, copy=False)


def _drift_alert(
    train_quantiles: Optional[Dict[str, object]],
    current_scores: pd.Series,
    *,
    tolerance: float = 0.15,
) -> Optional[Dict[str, float]]:
    """Compare quantiles between train metadata and current scores."""

    if not isinstance(train_quantiles, dict) or current_scores.empty:
        return None

    try:
        curr_q_raw = current_scores.quantile([0.01, 0.05, 0.5, 0.9]).to_dict()
    except Exception:
        return None

    curr_q = {str(k): float(v) for k, v in curr_q_raw.items() if np.isfinite(v)}
    alerts: Dict[str, float] = {}
    for key in ("0.01", "0.05", "0.5", "0.9"):
        if key not in train_quantiles or key not in curr_q:
            continue
        try:
            base = float(train_quantiles[key])
            current = float(curr_q[key])
        except (TypeError, ValueError):
            continue
        denom = max(abs(base), 1e-9)
        drift = abs(current - base) / denom
        if drift > tolerance:
            alerts[key] = float(drift)
    return alerts or None


def _series_to_binary(series: pd.Series) -> Optional[np.ndarray]:
    if series.empty:
        return None

    if series.dtype == bool:
        return series.fillna(False).astype(int).to_numpy()

    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except Exception:
        numeric = None

    if numeric is not None:
        valid = numeric.dropna()
        if not valid.empty:
            unique_values = set(int(v) for v in valid.astype(int).tolist())
            if unique_values.issubset({0, 1}):
                return numeric.fillna(0).astype(int).to_numpy()

    normalized = series.fillna("").astype(str).str.strip().str.lower()
    normalized = normalized.replace("", "unknown")
    tokens = normalized.str.replace(" ", "", regex=False)

    values: List[int] = []
    for value in tokens:
        if value in NORMAL_TOKEN_VARIANTS:
            values.append(0)
        elif value in ANOMALY_TOKEN_VARIANTS:
            values.append(1)
        else:
            return None
    return np.asarray(values, dtype=int)


def _calc_metrics(preds: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    preds = preds.astype(int)
    truth = truth.astype(int)
    tp = float(np.sum((preds == 1) & (truth == 1)))
    fp = float(np.sum((preds == 1) & (truth == 0)))
    fn = float(np.sum((preds == 0) & (truth == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def analyze_results(
    results_csv: str,
    out_dir: str,
    *,
    metadata_path: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    progress_cb=None,
) -> dict:
    """
    读取 iforest 逐包结果 CSV，输出两张图与一个 Top20 异常包清单：
    - top10_malicious_ratio.png
    - anomaly_score_distribution.png
    - top20_packets.csv
    """
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"找不到结果文件: {results_csv}")

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)

    if metadata is None:
        if not metadata_path or not os.path.exists(metadata_path):
            base_dir = _resolve_data_base()
            candidate = os.path.join(base_dir, "models", "latest_iforest_metadata.json")
            if os.path.exists(candidate):
                metadata_path = candidate

    loaded_metadata: Dict[str, object] = {}
    if isinstance(metadata, dict):
        loaded_metadata = metadata
    elif metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                loaded_metadata = payload
        except Exception:
            loaded_metadata = {}

    threshold_breakdown_meta = (
        loaded_metadata.get("threshold_breakdown")
        if isinstance(loaded_metadata, dict)
        else None
    )
    train_threshold = None
    if isinstance(threshold_breakdown_meta, dict) and threshold_breakdown_meta.get("adaptive") is not None:
        train_threshold = float(threshold_breakdown_meta.get("adaptive"))
    elif isinstance(loaded_metadata, dict) and loaded_metadata.get("threshold") is not None:
        train_threshold = float(loaded_metadata.get("threshold"))

    vote_threshold_train = None
    if isinstance(loaded_metadata, dict) and loaded_metadata.get("vote_threshold") is not None:
        vote_threshold_train = float(loaded_metadata.get("vote_threshold"))

    score_std_train = (
        float(loaded_metadata.get("score_std"))
        if isinstance(loaded_metadata, dict) and loaded_metadata.get("score_std") is not None
        else None
    )

    train_score_quantiles = (
        loaded_metadata.get("score_quantiles")
        if isinstance(loaded_metadata, dict)
        else None
    )
    train_score_histogram = (
        loaded_metadata.get("score_histogram")
        if isinstance(loaded_metadata, dict)
        else None
    )

    if progress_cb: progress_cb(30)

    # 1) 按文件的恶意占比（如果训练阶段已导出 summary_by_file.csv 也可以直接使用）
    if "pcap_file" not in df.columns or "is_malicious" not in df.columns:
        raise ValueError("结果文件缺少必要字段：pcap_file / is_malicious")

    def _clean_number(value) -> float | None:
        try:
            num = float(value)
        except Exception:
            return None
        if math.isnan(num) or math.isinf(num):
            return None
        return num

    total_rows = int(len(df))
    unique_files = int(df["pcap_file"].nunique()) if total_rows else 0

    malicious_raw = df["is_malicious"].astype("float32", copy=False)
    malicious_mask = malicious_raw > 0
    malicious_total = int(malicious_mask.sum())
    malicious_ratio_global = float(malicious_total / total_rows) if total_rows else 0.0

    avg_confidence = None
    avg_risk = None
    vote_ratio_quantiles = None
    vote_threshold_hint = None
    if "anomaly_confidence" in df.columns:
        try:
            conf_series = df["anomaly_confidence"].astype("float32", copy=False)
            if malicious_mask.any():
                avg_confidence = float(conf_series[malicious_mask].mean())
            else:
                avg_confidence = float(conf_series.mean())
        except Exception:
            avg_confidence = None
    if "risk_score" in df.columns:
        try:
            risk_series = df["risk_score"].astype("float32", copy=False)
            if malicious_mask.any():
                avg_risk = float(risk_series[malicious_mask].mean())
            else:
                avg_risk = float(risk_series.mean())
        except Exception:
            avg_risk = None
    if "vote_ratio" in df.columns:
        try:
            vote_series = df["vote_ratio"].astype("float32", copy=False)
            vote_ratio_quantiles = vote_series.quantile([0.5, 0.75, 0.9]).to_dict()
            vote_threshold_hint = float(vote_ratio_quantiles.get(0.75, vote_series.mean()))
        except Exception:
            vote_ratio_quantiles = None
            vote_threshold_hint = None

    if vote_threshold_train is not None:
        vote_threshold_hint = vote_threshold_train

    grouped = df.groupby("pcap_file", dropna=False)
    summary = grouped["is_malicious"].agg([("malicious_count", lambda s: int((s.astype("float32") > 0).sum())),
                                            ("total", "count")])
    summary["malicious_ratio"] = summary["malicious_count"] / summary["total"].clip(lower=1)
    if "anomaly_score" in df.columns:
        summary["avg_anomaly_score"] = grouped["anomaly_score"].mean()
        summary["min_anomaly_score"] = grouped["anomaly_score"].min()
    summary = summary.reset_index().sort_values("malicious_ratio", ascending=False)
    top10 = summary.head(10)

    summary_path = os.path.join(out_dir, "summary_by_file.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    out1 = None
    if not top10.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(top10["pcap_file"], top10["malicious_ratio"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Malicious Ratio")
        plt.title("Top 10 Files by Malicious Ratio")
        plt.tight_layout()
        out1 = os.path.join(out_dir, "top10_malicious_ratio.png")
        plt.savefig(out1)
        plt.close()

    if progress_cb: progress_cb(70)

    # 2) 分数分布
    if "anomaly_score" not in df.columns:
        raise ValueError("结果文件缺少 anomaly_score 字段")

    score_series = df["anomaly_score"].astype("float32", copy=False)
    # 大数据集时仅采样 200k 条用于绘图，加快速度
    if len(score_series) > 200_000:
        score_plot_sample = score_series.sample(200_000, random_state=0)
    else:
        score_plot_sample = score_series

    plt.figure(figsize=(8, 5))
    plt.hist(score_plot_sample, bins=50, alpha=0.8)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution")
    plt.tight_layout()
    out2 = os.path.join(out_dir, "anomaly_score_distribution.png")
    plt.savefig(out2)
    plt.close()

    psi_value: Optional[float] = None
    kl_divergence_val: Optional[float] = None
    hist_overlay_path: Optional[str] = None
    if isinstance(train_score_histogram, dict):
        bins_raw = train_score_histogram.get("bins")
        counts_raw = train_score_histogram.get("counts")
        try:
            bins_arr = np.asarray(bins_raw, dtype=float)
            counts_arr = np.asarray(counts_raw, dtype=float)
        except Exception:
            bins_arr = np.empty(0)
            counts_arr = np.empty(0)
        if bins_arr.size >= 2 and counts_arr.size == bins_arr.size - 1:
            try:
                current_counts, _ = np.histogram(score_series, bins=bins_arr)
                psi_value = _population_stability_index(counts_arr, current_counts)
                psi_entry = {
                    "value": float(psi_value),
                    "level": "low",
                }
                if psi_value >= 0.25:
                    psi_entry["level"] = "severe"
                    drift_retrain_reasons.append(f"PSI={psi_value:.3f}")
                elif psi_value >= 0.1:
                    psi_entry["level"] = "moderate"
                drift_details["psi"] = psi_entry

                kl_divergence_val = _kl_divergence(counts_arr, current_counts)
                kl_entry = {
                    "value": float(kl_divergence_val),
                    "level": "low",
                }
                if kl_divergence_val >= 0.5:
                    kl_entry["level"] = "severe"
                    drift_retrain_reasons.append(f"KL={kl_divergence_val:.3f}")
                elif kl_divergence_val >= 0.2:
                    kl_entry["level"] = "moderate"
                drift_details["kl_divergence"] = kl_entry
                plt.figure(figsize=(8, 5))
                width = np.diff(bins_arr)
                base_prob = _normalize_hist(counts_arr)
                curr_prob = _normalize_hist(current_counts)
                plt.bar(
                    bins_arr[:-1],
                    base_prob,
                    width=width,
                    alpha=0.4,
                    align="edge",
                    label="Train",
                )
                plt.bar(
                    bins_arr[:-1],
                    curr_prob,
                    width=width,
                    alpha=0.4,
                    align="edge",
                    label="Current",
                )
                plt.xlabel("Anomaly Score")
                plt.ylabel("Probability")
                plt.title("Score Distribution Drift Comparison")
                plt.legend()
                plt.tight_layout()
                hist_overlay_path = os.path.join(out_dir, "anomaly_score_hist_compare.png")
                plt.savefig(hist_overlay_path)
                plt.close()
            except Exception as exc:
                logger.warning("Failed to compute PSI/KL drift metrics: %s", exc)

    # 3) Top20 异常包（分数越小越异常）
    explanation_numeric_cols = [
        col
        for col in df.columns
        if col not in EXPLANATION_EXCLUDE_COLUMNS and is_numeric_dtype(df[col])
    ]
    top20 = df.sort_values("anomaly_score").head(20).copy()
    if explanation_numeric_cols:
        mu = df[explanation_numeric_cols].mean()
        sigma = df[explanation_numeric_cols].std(ddof=0).replace(0, 1e-6)

        def _top_reasons(row: pd.Series, k: int = 3) -> str:
            try:
                z_scores = ((row[explanation_numeric_cols] - mu) / sigma).abs()
            except Exception:
                return ""
            z_scores = z_scores.sort_values(ascending=False)
            pairs = [f"{col}:{z_scores[col]:.2f}" for col in z_scores.head(k).index]
            return ";".join(pairs)

        top20["top_reasons"] = top20.apply(_top_reasons, axis=1)
    out3 = os.path.join(out_dir, "top20_packets.csv")
    top20.to_csv(out3, index=False, encoding="utf-8")

    confusion_plot_path: Optional[str] = None
    permutation_plot_path: Optional[str] = None
    importance_source: Optional[str] = None
    if isinstance(loaded_metadata, dict):
        evaluation_meta = loaded_metadata.get("evaluation")
        if isinstance(evaluation_meta, dict):
            cm = evaluation_meta.get("confusion_matrix")
            if isinstance(cm, (list, tuple)):
                arr = np.asarray(cm, dtype=float)
                if arr.size == 4:
                    arr = arr.reshape(2, 2)
                if arr.shape == (2, 2):
                    plt.figure(figsize=(4, 3))
                    plt.imshow(arr, cmap="Blues")
                    for i in range(2):
                        for j in range(2):
                            plt.text(j, i, f"{int(arr[i, j])}", ha="center", va="center", color="black")
                    plt.xticks([0, 1], ["Normal", "Anomaly"])
                    plt.yticks([0, 1], ["Normal", "Anomaly"])
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix")
                    plt.tight_layout()
                    confusion_plot_path = os.path.join(out_dir, "confusion_matrix.png")
                    plt.savefig(confusion_plot_path)
                    plt.close()

        for key in ("permutation_importance_topk", "feature_importances_topk"):
            data = loaded_metadata.get(key)
            if isinstance(data, list) and data:
                importance_source = key
                df_imp = pd.DataFrame(data)
                if {"feature", "importance"}.issubset(df_imp.columns):
                    df_imp = df_imp.sort_values("importance", ascending=False).head(20)
                    plt.figure(figsize=(8, max(4, len(df_imp) * 0.35)))
                    plt.barh(df_imp["feature"], df_imp["importance"], color="#4C72B0")
                    plt.gca().invert_yaxis()
                    plt.xlabel("Importance")
                    title_label = "Permutation Importance" if key == "permutation_importance_topk" else "Feature Importance"
                    plt.title(title_label)
                    plt.tight_layout()
                    permutation_plot_path = os.path.join(out_dir, "feature_importance.png")
                    plt.savefig(permutation_plot_path)
                    plt.close()
                break

    top_dst_path = None
    dst_group_cols = [col for col in ("dst_ip", "dst_port") if col in df.columns]
    if dst_group_cols:
        try:
            by_dst = (
                df.groupby(dst_group_cols, dropna=False)["is_malicious"].mean().reset_index()
            )
            by_dst = by_dst.sort_values("is_malicious", ascending=False).head(50)
            top_dst_path = os.path.join(out_dir, "top50_dst_pairs.csv")
            by_dst.to_csv(top_dst_path, index=False, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to export top dst pairs: %s", exc)

    risk_results_path = None
    risk_source_col = None
    if "risk_score" in df.columns:
        risk_source_col = "risk_score"
    elif "anomaly_confidence" in df.columns:
        risk_source_col = "anomaly_confidence"
    if risk_source_col is not None:
        def _risk_bucket(value: object) -> str:
            try:
                val = float(value)
            except (TypeError, ValueError):
                val = 0.0
            if val >= 0.8:
                return "HIGH"
            if val >= 0.5:
                return "MEDIUM"
            return "LOW"

        try:
            df["risk_bucket"] = df[risk_source_col].apply(_risk_bucket)
            risk_results_path = os.path.join(out_dir, "results_with_risk.csv")
            df.to_csv(risk_results_path, index=False, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to export risk bucket results: %s", exc)

    shap_plot_path: Optional[str] = None
    shap_summary_csv: Optional[str] = None
    shap_summary: Optional[List[Dict[str, float]]] = None
    shap_status: Optional[str] = None
    feature_columns_meta = None
    if isinstance(loaded_metadata, dict):
        feature_columns_meta = loaded_metadata.get("feature_columns")
    pipeline_path_meta = None
    if isinstance(loaded_metadata, dict):
        pipeline_path_meta = (
            loaded_metadata.get("pipeline_path")
            or loaded_metadata.get("pipeline_latest")
        )

    if shap is None:
        shap_status = "missing_dependency"
    elif not feature_columns_meta or not isinstance(feature_columns_meta, list):
        shap_status = "missing_feature_columns"
    elif not pipeline_path_meta or not isinstance(pipeline_path_meta, str):
        shap_status = "pipeline_missing"
    elif not os.path.exists(pipeline_path_meta):
        shap_status = "pipeline_missing"
    else:
        align_columns = [col for col in feature_columns_meta if col in df.columns]
        if len(align_columns) < 3:
            shap_status = "insufficient_features"
        elif len(align_columns) > 120:
            shap_status = "too_many_features"
        else:
            try:
                pipeline = load(pipeline_path_meta)
                feature_frame = (
                    df.loc[:, align_columns]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                )
                if feature_frame.empty:
                    shap_status = "empty_features"
                else:
                    background_size = min(80, len(feature_frame))
                    sample_size = min(40, len(feature_frame))
                    if background_size < 5 or sample_size < 5:
                        shap_status = "insufficient_samples"
                    else:
                        background = feature_frame.sample(
                            n=background_size, random_state=0, replace=False
                        )
                        focus_df = (
                            df.sort_values("anomaly_score")
                            .head(sample_size)
                            .loc[:, align_columns]
                            .apply(pd.to_numeric, errors="coerce")
                            .fillna(0.0)
                        )

                        def _score_fn(values: np.ndarray) -> np.ndarray:
                            data = pd.DataFrame(values, columns=align_columns)
                            return np.asarray(pipeline.decision_function(data), dtype=float)

                        explainer = shap.KernelExplainer(
                            _score_fn,
                            background.to_numpy(dtype=float, copy=False),
                        )
                        nsamples = min(256, max(len(align_columns) * 2, 64))
                        shap_values = explainer.shap_values(
                            focus_df.to_numpy(dtype=float, copy=False),
                            nsamples=nsamples,
                        )
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        shap_abs = np.mean(np.abs(shap_values), axis=0)
                        shap_summary_df = pd.DataFrame(
                            {
                                "feature": align_columns,
                                "mean_abs_shap": shap_abs,
                            }
                        ).sort_values("mean_abs_shap", ascending=False)
                        shap_summary = [
                            {
                                "feature": str(row["feature"]),
                                "mean_abs_shap": float(row["mean_abs_shap"]),
                            }
                            for _, row in shap_summary_df.head(40).iterrows()
                        ]
                        shap_summary_csv = os.path.join(out_dir, "shap_summary.csv")
                        shap_summary_df.to_csv(
                            shap_summary_csv, index=False, encoding="utf-8"
                        )
                        try:
                            shap.summary_plot(
                                shap_values,
                                focus_df.to_numpy(dtype=float, copy=False),
                                feature_names=align_columns,
                                show=False,
                                plot_size=(10, max(4, len(align_columns) * 0.15)),
                            )
                            shap_plot_path = os.path.join(out_dir, "shap_beeswarm.png")
                            plt.tight_layout()
                            plt.savefig(shap_plot_path, bbox_inches="tight")
                            plt.close()
                            shap_status = "ok"
                        except Exception as exc:
                            shap_status = f"plot_failed:{exc}"
            except Exception as exc:
                shap_status = f"error:{exc}"

    cluster_plot_path: Optional[str] = None
    cluster_summary: Optional[List[Dict[str, object]]] = None
    numeric_columns = _select_numeric_columns(df, EXPLANATION_EXCLUDE_COLUMNS)
    numeric_matrix = _prepare_numeric_matrix(df, numeric_columns)
    if numeric_matrix is not None and numeric_matrix.shape[0] >= 20 and numeric_matrix.shape[1] >= 2:
        try:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(numeric_matrix)
            reducer = TruncatedSVD(n_components=2, random_state=42)
            reduced = reducer.fit_transform(scaled)
            cluster_count = min(8, max(2, reduced.shape[0] // 500 + 2))
            kmeans = MiniBatchKMeans(
                n_clusters=cluster_count,
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(reduced)
            anomaly_mask = df["is_malicious"].astype(int, copy=False) > 0
            plt.figure(figsize=(8, 6))
            plt.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=labels,
                cmap="tab10",
                s=16,
                alpha=0.7,
            )
            if anomaly_mask.any():
                plt.scatter(
                    reduced[anomaly_mask, 0],
                    reduced[anomaly_mask, 1],
                    facecolors="none",
                    edgecolors="red",
                    s=40,
                    linewidths=1.0,
                    label="Anomaly",
                )
                plt.legend()
            plt.xlabel("Temporal component 1")
            plt.ylabel("Temporal component 2")
            plt.title("Flow Cluster Map")
            plt.tight_layout()
            cluster_plot_path = os.path.join(out_dir, "flow_cluster_map.png")
            plt.savefig(cluster_plot_path)
            plt.close()

            cluster_df = pd.DataFrame({"cluster": labels, "is_malicious": anomaly_mask.astype(int)})
            cluster_stats = (
                cluster_df.groupby("cluster")
                .agg(total=("is_malicious", "count"), anomalies=("is_malicious", "sum"))
                .reset_index()
            )
            cluster_stats["anomaly_ratio"] = cluster_stats["anomalies"] / cluster_stats["total"].clip(lower=1)
            cluster_summary = cluster_stats.to_dict("records")
        except Exception as exc:
            logger.warning("Failed to build cluster map: %s", exc)

    timeline_plot_path: Optional[str] = None
    timeline_points: Optional[int] = None
    time_series = _extract_time_series(df)
    if time_series is not None:
        try:
            if risk_source_col is not None:
                timeline_scores = pd.to_numeric(df[risk_source_col], errors="coerce")
            else:
                timeline_scores = -pd.to_numeric(df["anomaly_score"], errors="coerce")
            timeline_df = pd.DataFrame(
                {"time": time_series, "score": timeline_scores}
            ).dropna()
            timeline_df = timeline_df.sort_values("time")
            if len(timeline_df) >= 10:
                timeline_points = int(len(timeline_df))
                plt.figure(figsize=(10, 4))
                plt.plot(
                    timeline_df["time"],
                    timeline_df["score"].rolling(window=10, min_periods=1).mean(),
                    label="Rolling mean",
                )
                plt.scatter(
                    timeline_df["time"],
                    timeline_df["score"],
                    s=6,
                    alpha=0.3,
                    label="Raw score",
                )
                plt.xlabel("Time")
                plt.ylabel("Risk Score" if risk_source_col else "-Anomaly Score")
                plt.title("Risk Trend Over Time")
                plt.legend()
                plt.tight_layout()
                timeline_plot_path = os.path.join(out_dir, "risk_time_series.png")
                plt.savefig(timeline_plot_path)
                plt.close()
        except Exception as exc:
            logger.warning("Failed to render timeline chart: %s", exc)

    # -------- 补充信息 --------
    raw_quantiles = score_series.quantile([0.01, 0.05, 0.1, 0.5, 0.9]).to_dict()
    anomaly_score_quantiles = {str(k): _clean_number(v) for k, v in raw_quantiles.items()}
    drift_quantiles = _drift_alert(train_score_quantiles, score_series)
    drift_details: Dict[str, object] = {}
    drift_retrain_reasons: List[str] = []
    if drift_quantiles:
        drift_details["quantiles"] = drift_quantiles
        max_shift = max(float(v) for v in drift_quantiles.values()) if drift_quantiles else 0.0
        if max_shift > 0.3:
            drift_retrain_reasons.append(f"分位数偏移 {max_shift:.2f}")
    drift_alerts = drift_details or None
    drift_retrain = bool(drift_retrain_reasons)
    if drift_alerts:
        logger.warning("Anomaly score drift detected: %s", drift_alerts)

    score_threshold = train_threshold if train_threshold is not None else anomaly_score_quantiles.get("0.05")
    if score_threshold is None:
        fallback = _clean_number(score_series.min()) if len(score_series) else None
        score_threshold = fallback if fallback is not None else 0.0
    ratio_std = float(summary["malicious_ratio"].std(ddof=0) or 0.0)
    ratio_threshold = min(1.0, malicious_ratio_global + 2 * ratio_std)

    suspect_files = summary[summary["malicious_count"] > 0].copy()
    suspect_files = suspect_files[suspect_files["malicious_ratio"] >= max(0.0, ratio_threshold)]
    if suspect_files.empty:
        # 如果阈值过高，退化为所有有异常包的文件
        suspect_files = summary[summary["malicious_count"] > 0]

    anomalous_files: list[dict[str, object]] = []
    for _, row in suspect_files.iterrows():
        anomalous_files.append(
            {
                "pcap_file": row.get("pcap_file"),
                "malicious_count": int(row.get("malicious_count", 0)),
                "total": int(row.get("total", 0)),
                "malicious_ratio": float(row.get("malicious_ratio", 0.0)),
                "avg_anomaly_score": _clean_number(row.get("avg_anomaly_score")) if "avg_anomaly_score" in row else None,
            }
        )

    if anomalous_files:
        anomalous_files = sorted(anomalous_files, key=lambda x: x["malicious_ratio"], reverse=True)

    train_decision = None
    if train_threshold is not None and "anomaly_score" in df.columns:
        try:
            score_arr = df["anomaly_score"].astype("float32", copy=False)
            decision = (score_arr <= float(train_threshold)).astype(int)
            if vote_threshold_train is not None and "vote_ratio" in df.columns:
                vote_arr = df["vote_ratio"].astype("float32", copy=False)
                vote_mask = vote_arr >= float(vote_threshold_train)
                decision = (decision & vote_mask.astype(int)).astype(int)
            train_decision = decision.to_numpy(dtype=int, copy=False)
        except Exception:
            train_decision = None

    ground_truth = None
    ground_truth_column: Optional[str] = None
    candidate_columns: List[str] = list(GROUND_TRUTH_COLUMN_CANDIDATES)
    preferred_column = None
    if isinstance(loaded_metadata, dict) and loaded_metadata.get("ground_truth_column"):
        preferred_column = str(loaded_metadata.get("ground_truth_column"))
    if preferred_column:
        candidate_columns = [preferred_column] + [c for c in candidate_columns if c != preferred_column]

    for col in candidate_columns:
        if col not in df.columns:
            continue
        arr = _series_to_binary(df[col])
        if arr is not None and len(arr) == len(df):
            ground_truth = arr
            ground_truth_column = col
            break

    metrics_rows: List[Dict[str, object]] = []
    base_metrics: Optional[Dict[str, float]] = None
    train_metrics: Optional[Dict[str, float]] = None
    roc_plot_path: Optional[str] = None
    pr_plot_path: Optional[str] = None
    roc_auc_val: Optional[float] = None
    pr_auc_val: Optional[float] = None
    avg_precision_val: Optional[float] = None

    if ground_truth is not None:
        preds_actual = df["is_malicious"].astype(int, copy=False).to_numpy()
        base_metrics = _calc_metrics(preds_actual, ground_truth)
        metrics_rows.append(
            {
                "variant": "model_prediction",
                "precision": base_metrics["precision"],
                "recall": base_metrics["recall"],
                "f1": base_metrics["f1"],
                "score_threshold": float(train_threshold) if train_threshold is not None else None,
                "vote_threshold": float(vote_threshold_train) if vote_threshold_train is not None else None,
            }
        )
        if train_decision is not None:
            train_metrics = _calc_metrics(train_decision, ground_truth)
            metrics_rows.append(
                {
                    "variant": "train_threshold",
                    "precision": train_metrics["precision"],
                    "recall": train_metrics["recall"],
                    "f1": train_metrics["f1"],
                    "score_threshold": float(train_threshold) if train_threshold is not None else None,
                    "vote_threshold": float(vote_threshold_train) if vote_threshold_train is not None else None,
                }
            )

        unique_truth = np.unique(ground_truth)
        if unique_truth.size > 1:
            if "risk_score" in df.columns:
                score_like = df["risk_score"].astype("float64", copy=False)
            elif "anomaly_confidence" in df.columns:
                score_like = df["anomaly_confidence"].astype("float64", copy=False)
            else:
                score_like = -df["anomaly_score"].astype("float64", copy=False)

            score_array = np.asarray(score_like, dtype=float)
            try:
                fpr, tpr, _ = roc_curve(ground_truth, score_array)
                roc_auc_val = float(auc(fpr, tpr)) if fpr.size and tpr.size else None
                if roc_auc_val is not None:
                    plt.figure(figsize=(6, 5))
                    plt.plot(fpr, tpr, label=f"AUC={roc_auc_val:.3f}")
                    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    roc_plot_path = os.path.join(out_dir, "roc_curve.png")
                    plt.savefig(roc_plot_path)
                    plt.close()

                precision_curve, recall_curve, _ = precision_recall_curve(
                    ground_truth, score_array
                )
                if recall_curve.size and precision_curve.size:
                    pr_auc_val = float(auc(recall_curve, precision_curve))
                    avg_precision_val = float(
                        average_precision_score(ground_truth, score_array)
                    )
                    plt.figure(figsize=(6, 5))
                    plt.plot(
                        recall_curve,
                        precision_curve,
                        label=f"PR AUC={pr_auc_val:.3f}",
                    )
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    pr_plot_path = os.path.join(out_dir, "precision_recall_curve.png")
                    plt.savefig(pr_plot_path)
                    plt.close()
            except Exception as exc:
                logger.warning("Failed to compute ROC/PR curves: %s", exc)

    metrics_csv_path = None
    metrics_json_path = None
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = os.path.join(out_dir, "threshold_evaluation.csv")
        metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")

    if ground_truth is not None:
        metrics_summary_payload = {
            "ground_truth_column": ground_truth_column,
            "model_metrics": base_metrics,
            "train_threshold_metrics": train_metrics,
            "roc_auc": roc_auc_val,
            "pr_auc": pr_auc_val,
            "average_precision": avg_precision_val,
        }
        metrics_json_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_json_path, "w", encoding="utf-8") as fh:
            json.dump(metrics_summary_payload, fh, ensure_ascii=False, indent=2)

    single_file_status = None
    if unique_files == 1:
        single_file_status = "异常" if malicious_total > 0 else "正常"

    summary_lines = [
        f"总检测包数：{total_rows}",
        f"涉及文件数：{unique_files}",
        f"异常包数量：{malicious_total} ({malicious_ratio_global:.2%})",
        f"建议异常得分阈值（越小越异常）：≤ {score_threshold:.6f}",
        f"文件级判定阈值：恶意占比 ≥ {ratio_threshold:.2%}",
    ]
    if drift_alerts:
        summary_lines.append(
            "⚠ 分布漂移告警：" + json.dumps(drift_alerts, ensure_ascii=False)
        )
    if drift_retrain:
        reason_text = "; ".join(drift_retrain_reasons) if drift_retrain_reasons else "分布漂移超过阈值"
        summary_lines.append(f"建议：检测到显著分布漂移（{reason_text}），请重新训练模型。")
    if psi_value is not None:
        summary_lines.append(f"PSI：{psi_value:.4f}")
    if kl_divergence_val is not None:
        summary_lines.append(f"KL 散度：{kl_divergence_val:.4f}")
    if avg_risk is not None:
        summary_lines.append(f"风险分均值：{avg_risk:.2%}")
    elif avg_confidence is not None:
        summary_lines.append(f"异常置信度均值：{avg_confidence:.2%}")
    if vote_threshold_hint is not None:
        summary_lines.append(f"投票占比建议阈值：≥ {vote_threshold_hint:.2f}")
    if train_metrics is not None:
        summary_lines.append(
            "训练阈值复算 Precision/Recall/F1：{:.2%}/{:.2%}/{:.2%}".format(
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
            )
        )
    if roc_auc_val is not None and pr_auc_val is not None:
        summary_lines.append(
            f"ROC AUC：{roc_auc_val:.3f}；PR AUC：{pr_auc_val:.3f}"
        )
    if avg_precision_val is not None:
        summary_lines.append(f"平均精度 (AP)：{avg_precision_val:.3f}")
    if single_file_status:
        summary_lines.append(f"单文件判定：{single_file_status}")
    if anomalous_files:
        summary_lines.append("疑似异常文件 TOP: ")
        for item in anomalous_files[:5]:
            avg_value = item.get("avg_anomaly_score")
            if avg_value is not None:
                avg_score_txt = f"{avg_value:.6f}"
            else:
                avg_score_txt = "N/A"
            summary_lines.append(
                f"  - {item['pcap_file']}: {item['malicious_count']}/{item['total']} ({item['malicious_ratio']:.2%}), 平均得分 {avg_score_txt}"
            )

    if malicious_total > 0:
        summary_lines.append("诊断：检测到异常流量，请重点排查。")
    else:
        summary_lines.append("诊断：未检测到明显异常流量。")

    summary_text = "\n".join(summary_lines)

    details_payload = {
        "total_rows": total_rows,
        "unique_files": unique_files,
        "malicious_total": malicious_total,
        "malicious_ratio": malicious_ratio_global,
        "score_threshold": score_threshold,
        "ratio_threshold": ratio_threshold,
        "avg_confidence": avg_confidence,
        "avg_risk": avg_risk,
        "vote_ratio_quantiles": vote_ratio_quantiles,
        "vote_threshold_hint": vote_threshold_hint,
        "anomalous_files": anomalous_files,
        "anomaly_score_quantiles": anomaly_score_quantiles,
        "single_file_status": single_file_status,
        "summary_csv": summary_path,
        "train_threshold": float(train_threshold) if train_threshold is not None else None,
        "train_vote_threshold": float(vote_threshold_train) if vote_threshold_train is not None else None,
        "train_metrics": train_metrics,
        "model_metrics": base_metrics,
        "ground_truth_column": ground_truth_column,
        "metadata_path": metadata_path,
        "score_std_train": score_std_train,
        "score_quantiles_train": train_score_quantiles,
        "drift_alerts": drift_alerts,
        "drift_metrics": {
            "psi": float(psi_value) if psi_value is not None else None,
            "kl_divergence": float(kl_divergence_val) if kl_divergence_val is not None else None,
        },
        "drift_retrain": drift_retrain,
        "drift_retrain_reasons": drift_retrain_reasons,
        "score_hist_compare_plot": hist_overlay_path,
        "roc_auc": roc_auc_val,
        "pr_auc": pr_auc_val,
        "average_precision": avg_precision_val,
        "metrics_json": metrics_json_path,
        "roc_plot": roc_plot_path,
        "pr_plot": pr_plot_path,
        "top_dst_csv": top_dst_path,
        "risk_bucket_csv": risk_results_path,
        "confusion_plot": confusion_plot_path,
        "feature_importance_plot": permutation_plot_path,
        "feature_importance_source": importance_source,
        "shap_plot": shap_plot_path,
        "shap_summary_csv": shap_summary_csv,
        "shap_status": shap_status,
        "shap_top_features": shap_summary,
        "cluster_plot": cluster_plot_path,
        "cluster_summary": cluster_summary,
        "timeline_plot": timeline_plot_path,
        "timeline_points": timeline_points,
    }

    export_payload = {
        "anomaly_count": malicious_total,
        "anomaly_files": [str(item.get("pcap_file")) for item in anomalous_files if item.get("pcap_file")],
        "single_file_status": single_file_status,
        "status_text": "异常" if malicious_total > 0 else "正常",
    }

    summary_json_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary_text, "details": details_payload}, fh, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100)

    logger.info(
        "Analysis completed: rows=%d anomalies=%d score_threshold=%.6f train_threshold=%s vote_threshold=%s",
        total_rows,
        malicious_total,
        float(score_threshold),
        "%.6f" % train_threshold if train_threshold is not None else "None",
        "%.3f" % vote_threshold_train if vote_threshold_train is not None else "None",
    )
    if train_metrics is not None:
        logger.info(
            "Train-threshold metrics precision=%.3f recall=%.3f f1=%.3f",
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
        )

    payload = {
        "plots": [
            p
            for p in (
                out1,
                out2,
                confusion_plot_path,
                permutation_plot_path,
                roc_plot_path,
                pr_plot_path,
                hist_overlay_path,
                shap_plot_path,
                cluster_plot_path,
                timeline_plot_path,
            )
            if p
        ],
        "top20_csv": out3,
        "top_dst_csv": top_dst_path,
        "risk_bucket_csv": risk_results_path,
        "hist_compare_plot": hist_overlay_path,
        "shap_summary_csv": shap_summary_csv,
        "shap_plot": shap_plot_path,
        "cluster_plot": cluster_plot_path,
        "timeline_plot": timeline_plot_path,
        "out_dir": out_dir,
        "summary_csv": summary_path,
        "summary_json": summary_json_path,
        "summary_text": summary_text,
        "metrics": details_payload,
        "drift_alerts": drift_alerts,
        "drift_retrain": drift_retrain,
        "drift_retrain_reasons": drift_retrain_reasons,
        "export_payload": export_payload,
        "metrics_csv": metrics_csv_path,
        "metrics_json": metrics_json_path,
    }

    return payload
