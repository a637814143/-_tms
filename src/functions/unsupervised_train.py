# src/functions/unsupervised_train.py
import os
import glob
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from joblib import dump

from src.functions.feature_extractor import extract_features

NUMERIC_FEATURES = ["length"]  # 采用轻量特征：每包长度


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

def _ensure_dirs(results_dir: str, models_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _load_or_extract_feature_csv(pcap_path: str, feature_dir: str, progress_cb=None) -> str:
    base = os.path.splitext(os.path.basename(pcap_path))[0]
    out_csv = os.path.join(feature_dir, f"{base}_features.csv")
    os.makedirs(feature_dir, exist_ok=True)
    if not os.path.exists(out_csv):
        extract_features(pcap_path, out_csv, progress_cb=None)  # 单文件内部进度无需重复冒泡
    return out_csv

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
    for c in NUMERIC_FEATURES:
        if c in working_df.columns:
            working_df[c] = working_df[c].apply(_safe_float)
        else:
            working_df[c] = 0.0

    X = working_df[NUMERIC_FEATURES].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=base_estimators,
        contamination=contamination,
        random_state=42,
        warm_start=True,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    if progress_cb:
        progress_cb(70)

    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    is_malicious = (preds == -1).astype(int)

    working_df["anomaly_score"] = scores
    working_df["is_malicious"] = is_malicious

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
    else:
        summary = pd.DataFrame(
            [
                {
                    "source": group_column or "all",
                    "total_packets": len(working_df),
                    "malicious_packets": int(is_malicious.sum()),
                    "malicious_ratio": float(is_malicious.mean()),
                }
            ]
        )

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
):
    """
    无监督训练：
    - 优先支持向量化目录/NPZ 数据集（results/vector）
    - 兼容旧的 split 目录下 PCAP 文件批量训练
    """

    _ensure_dirs(results_dir, models_dir)

    dataset_path: Optional[str] = None
    pcap_dir: Optional[str] = None

    if not split_dir:
        raise FileNotFoundError("未提供训练数据路径")

    if os.path.isfile(split_dir):
        if _is_npz(split_dir):
            dataset_path = split_dir
        elif split_dir.lower().endswith((".pcap", ".pcapng")):
            pcap_dir = os.path.dirname(split_dir) or os.path.abspath(os.path.join(split_dir, os.pardir))
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

    pcaps = sorted(glob.glob(os.path.join(pcap_dir, "*.pcap"))) + sorted(
        glob.glob(os.path.join(pcap_dir, "*.pcapng"))
    )

    if len(pcaps) == 0:
        raise RuntimeError(f"未在目录中发现 pcap/pcapng: {pcap_dir}")

    feature_csvs = []
    for i, pcap in enumerate(pcaps, 1):
        out_csv = _load_or_extract_feature_csv(pcap, feature_dir)
        feature_csvs.append(out_csv)
        if progress_cb:
            progress_cb(int(min(40, 40 * i / len(pcaps))))

    dfs = []
    for csv_path in feature_csvs:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 读取特征失败 {csv_path}: {e}")

    if len(dfs) == 0:
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



