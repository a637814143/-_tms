"""无监督异常检测训练流程。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Number
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.functions.feature_extractor import extract_features, extract_features_dir

META_COLUMNS = {
    "pcap_file",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
}

LABEL_KEYWORDS = (
    "label",
    "标签",
    "ground",
    "truth",
    "result",
    "status",
    "attack",
    "threat",
    "tag",
    "category",
    "type",
    "标记",
)

POSITIVE_TOKENS = {
    "1",
    "true",
    "yes",
    "y",
    "异常",
    "恶意",
    "攻击",
    "攻击流量",
    "anomaly",
    "abnormal",
    "malicious",
    "threat",
    "suspicious",
    "bot",
    "botnet",
    "infected",
    "compromised",
    "攻击行为",
}

NEGATIVE_TOKENS = {
    "0",
    "false",
    "no",
    "n",
    "normal",
    "benign",
    "legit",
    "合法",
    "正常",
    "safe",
    "clean",
    "good",
    "none",
    "无",
    "ok",
}

POSITIVE_CONTAINS = (
    "anomaly",
    "abnormal",
    "attack",
    "malicious",
    "threat",
    "suspicious",
    "bot",
    "恶意",
    "异常",
    "攻击",
    "木马",
    "病毒",
    "后门",
    "入侵",
    "泄露",
    "exploit",
    "c2",
    "command",
    "shell",
)

NEGATIVE_CONTAINS = (
    "normal",
    "benign",
    "合法",
    "正常",
    "good",
    "clean",
    "safe",
    "whitelist",
)


def _value_indicates_anomaly(value) -> Optional[bool]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, Number):
        if float(value) == -1:
            return True
        if float(value) == 0:
            return False
        if float(value) == 1:
            return True
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if text in POSITIVE_TOKENS:
        return True
    if text in NEGATIVE_TOKENS:
        return False
    for key in POSITIVE_CONTAINS:
        if key in text:
            return True
    for key in NEGATIVE_CONTAINS:
        if key in text:
            return False
    return None


def _normalize_truth_series(series: pd.Series) -> Optional[pd.Series]:
    if series is None or len(series) == 0:
        return None
    if is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().all():
            return None
        unique = set(int(v) for v in numeric.dropna().astype(int).unique())
        if unique.issubset({0, 1}):
            return numeric.fillna(0).astype(int).astype(bool)
        if unique.issubset({-1, 0}):
            return (numeric.fillna(0).astype(int) == -1)
        if unique.issubset({-1, 1}):
            return (numeric.fillna(1).astype(int) == -1)
        return None

    try:
        mapped = series.map(_value_indicates_anomaly)
    except Exception:
        mapped = series.apply(_value_indicates_anomaly)

    if mapped is None:
        return None

    valid = mapped.dropna()
    if valid.empty:
        return None

    coverage = len(valid) / max(1, len(series))
    if len(series) > 10 and coverage < 0.5:
        return None

    return mapped.fillna(False).astype(bool)


def infer_ground_truth_labels(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    if df is None or df.empty:
        return None, None

    for col in df.columns:
        name = str(col)
        if name in META_COLUMNS:
            continue
        lower = name.lower()
        if lower.startswith("__"):
            continue
        if lower in {"prediction", "anomaly_score", "is_malicious"}:
            continue
        if not any(key in lower for key in LABEL_KEYWORDS):
            continue
        series = df[col]
        normalized = _normalize_truth_series(series)
        if normalized is not None:
            return name, normalized

    return None, None


def normalize_prediction_output(values) -> Tuple[List[str], List[int]]:
    labels: List[str] = []
    flags: List[int] = []
    for value in values:
        is_anomaly = None
        if isinstance(value, Number):
            try:
                num = float(value)
            except Exception:
                num = None
            if num is not None:
                if num == -1:
                    is_anomaly = True
                elif num in {0.0, 1.0}:
                    is_anomaly = False
        if is_anomaly is None:
            text = str(value).strip().lower()
            if text in {"-1", "异常", "恶意", "malicious", "attack", "anomaly", "abnormal", "threat"}:
                is_anomaly = True
            elif text in {"1", "0", "normal", "benign", "正常", "合法", "safe", "ok", "none", "无"}:
                is_anomaly = False
        if is_anomaly is None:
            try:
                is_anomaly = float(value) < 0
            except Exception:
                is_anomaly = False

        labels.append("异常" if is_anomaly else "正常")
        flags.append(1 if is_anomaly else 0)

    return labels, flags


def _is_npz(path: str) -> bool:
    return path.lower().endswith(".npz")


def _latest_npz_in_dir(directory: str) -> Optional[str]:
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if _is_npz(name)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _load_npz_dataset(dataset_path: str) -> Tuple[np.ndarray, list]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    if "X" not in data:
        raise ValueError(f"NPZ 文件缺少 'X' 数据: {dataset_path}")

    X = data["X"]
    columns_raw = data.get("columns", None)
    if columns_raw is None:
        columns = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        columns = [str(c) for c in list(columns_raw)]

    if not isinstance(X, np.ndarray):
        raise ValueError("X 必须为 numpy.ndarray")

    return X.astype(float), columns


def _ensure_dirs(results_dir: str, models_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


def _load_feature_frames(paths: List[str], workers: int) -> List[pd.DataFrame]:
    if not paths:
        return []

    def _read_csv(path: str) -> Tuple[str, pd.DataFrame]:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")
        return path, df

    frames: List[Tuple[str, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_read_csv, path): path for path in paths}
        for future in as_completed(futures):
            frames.append(future.result())

    frames.sort(key=lambda item: item[0])
    return [df for _, df in frames]


def _prepare_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    numeric_cols: List[str] = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        lower = str(col).lower()
        if lower.startswith("__"):
            continue
        if any(key in lower for key in LABEL_KEYWORDS):
            continue
        if is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        raise RuntimeError("未发现可用于训练的数值特征。")

    matrix = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return matrix.to_numpy(dtype=float, copy=False), numeric_cols


def _train_from_dataframe(
    df: pd.DataFrame,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
    group_column: Optional[str] = None,
) -> dict:
    if df.empty:
        raise RuntimeError("训练数据为空。")

    working_df = df.copy()
    truth_col, truth_series = infer_ground_truth_labels(df)
    truth_bool = truth_series.astype(bool) if truth_series is not None else None

    X, feature_columns = _prepare_feature_matrix(working_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=base_estimators,
        contamination=contamination,
        random_state=42,
        warm_start=True,
        n_jobs=-1,
    )
    if progress_cb:
        progress_cb(80)
    model.fit(X_scaled)
    if progress_cb:
        progress_cb(90)

    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    pred_labels, pred_flags = normalize_prediction_output(preds)
    is_malicious = np.array(pred_flags, dtype=int)

    working_df["prediction"] = pred_labels
    working_df["anomaly_score"] = -scores
    working_df["is_malicious"] = is_malicious

    accuracy: Optional[float] = None
    if truth_bool is not None:
        truth_array = truth_bool.to_numpy(dtype=int)
        comparison = (is_malicious == truth_array)
        accuracy = float(comparison.mean())
        truth_labels = ["异常" if flag else "正常" for flag in truth_bool.tolist()]
        gt_name = truth_col or "ground_truth"
        if gt_name in working_df.columns:
            gt_name = f"{gt_name}_ground_truth"
        working_df[gt_name] = truth_labels

    results_csv = os.path.join(results_dir, "iforest_results.csv")
    working_df.to_csv(results_csv, index=False, encoding="utf-8")

    if group_column and group_column in working_df.columns:
        summary = (
            working_df.groupby(group_column)
            .agg(
                total_packets=(group_column, "count"),
                malicious_packets=("is_malicious", "sum"),
                malicious_ratio=("is_malicious", "mean"),
            )
            .reset_index()
            .sort_values("malicious_ratio", ascending=False)
        )
        if truth_bool is not None:
            compare_df = pd.DataFrame(
                {
                    group_column: working_df[group_column],
                    "_pred": is_malicious,
                    "_truth": truth_bool.astype(int),
                }
            )
            group_acc = (
                compare_df.groupby(group_column)
                .apply(lambda g: (g["_pred"] == g["_truth"]).mean())
                .rename("accuracy")
                .reset_index()
            )
            summary = summary.merge(group_acc, on=group_column, how="left")
    else:
        summary = pd.DataFrame(
            [
                {
                    "source": group_column or "all",
                    "total_packets": len(working_df),
                    "malicious_packets": int(is_malicious.sum()),
                    "malicious_ratio": float(is_malicious.mean()),
                    "accuracy": accuracy if accuracy is not None else pd.NA,
                }
            ]
        )

    if accuracy is not None and "accuracy" not in summary.columns:
        summary["accuracy"] = accuracy

    summary_csv = os.path.join(results_dir, "summary_by_file.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    model_path = os.path.join(models_dir, "isoforest.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    dump(model, model_path)
    dump(scaler, scaler_path)

    if progress_cb:
        progress_cb(100)

    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "packets": len(working_df),
        "flows": len(working_df),
        "malicious": int(is_malicious.sum()),
        "contamination": contamination,
        "feature_columns": feature_columns,
        "accuracy": accuracy,
        "accuracy_column": truth_col,
    }


def _train_from_npz(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
) -> dict:
    X, columns = _load_npz_dataset(dataset_path)
    df = pd.DataFrame(X, columns=columns)

    manifest_path = dataset_path.replace(".npz", "_manifest.csv")
    manifest_col = None
    if os.path.exists(manifest_path):
        try:
            manifest_df = pd.read_csv(manifest_path, encoding="utf-8")
            expanded = []
            for _, row in manifest_df.iterrows():
                rows = int(row.get("rows", 0))
                source_file = row.get("source_file") or row.get("source_path") or "unknown"
                expanded.extend([source_file] * rows)
            if len(expanded) == len(df):
                df["__source_file__"] = expanded
                manifest_col = "__source_file__"
        except Exception as exc:
            print(f"[WARN] 读取 manifest 失败 {manifest_path}: {exc}")

    if progress_cb:
        progress_cb(40)

    return _train_from_dataframe(
        df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column=manifest_col,
    )


def train_unsupervised_on_split(
    split_dir: str,
    results_dir: str,
    models_dir: str,
    contamination: float = 0.05,
    base_estimators: int = 50,
    progress_cb=None,
    workers: int = 4,
):
    """
    无监督训练：
    - 支持直接加载 NPZ 向量数据集
    - 对 PCAP 目录进行特征提取（多线程）后再训练
    """

    _ensure_dirs(results_dir, models_dir)

    dataset_path: Optional[str] = None
    pcap_dir: Optional[str] = None
    pcap_files: List[str] = []

    if not split_dir:
        raise FileNotFoundError("未提供训练数据路径")

    if os.path.isfile(split_dir):
        if _is_npz(split_dir):
            dataset_path = split_dir
        elif split_dir.lower().endswith((".pcap", ".pcapng")):
            pcap_dir = os.path.dirname(split_dir) or os.path.abspath(os.path.join(split_dir, os.pardir))
            pcap_files = [split_dir]
        else:
            raise FileNotFoundError(f"不支持的训练文件: {split_dir}")
    elif os.path.isdir(split_dir):
        dataset_path = _latest_npz_in_dir(split_dir)
        if dataset_path is None:
            pcap_dir = split_dir
    else:
        raise FileNotFoundError(f"路径不存在: {split_dir}")

    if dataset_path:
        return _train_from_npz(
            dataset_path,
            results_dir=results_dir,
            models_dir=models_dir,
            contamination=contamination,
            base_estimators=base_estimators,
            progress_cb=progress_cb,
        )

    if not pcap_dir or not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"目录不存在: {pcap_dir or split_dir}")

    feature_dir = os.path.join(results_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)

    feature_csvs: List[str] = []
    if pcap_files:
        for idx, pcap in enumerate(pcap_files, 1):
            base = os.path.splitext(os.path.basename(pcap))[0]
            csv_path = os.path.join(feature_dir, f"{base}_features.csv")
            extract_features(pcap, csv_path, progress_cb=None)
            feature_csvs.append(csv_path)
            if progress_cb:
                progress_cb(min(60, int(60 * idx / len(pcap_files))))
    else:
        if progress_cb:
            def _extract_progress(pct: int) -> None:
                pct = max(0, min(100, pct))
                progress_cb(min(60, int(pct * 0.6)))
        else:
            _extract_progress = None

        dir_result = extract_features_dir(
            pcap_dir,
            feature_dir,
            workers=workers,
            progress_cb=_extract_progress,
        )
        if not isinstance(dir_result, dict):
            raise RuntimeError("目录特征提取失败，未获得输出文件信息。")
        csv_path = dir_result.get("csv_path")
        if not csv_path:
            raise RuntimeError("目录特征提取未返回有效的 CSV 路径。")
        feature_csvs = [csv_path]

    if progress_cb:
        progress_cb(70)

    dfs = _load_feature_frames(feature_csvs, workers)

    if not dfs:
        raise RuntimeError("未能读取到任何特征 CSV。")

    full_df = pd.concat(dfs, ignore_index=True)

    return _train_from_dataframe(
        full_df,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=contamination,
        base_estimators=base_estimators,
        progress_cb=progress_cb,
        group_column="pcap_file" if "pcap_file" in full_df.columns else None,
    )