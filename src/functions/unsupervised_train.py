
"""无监督异常检测训练流程。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
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
    "__source_file__",
    "__source_path__",
}


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


def _is_preprocessed_csv(path: str) -> bool:
    if not path.lower().endswith(".csv"):
        return False
    name = os.path.basename(path)
    if not name.startswith("dataset_preprocessed_"):
        return False
    base, _ = os.path.splitext(path)
    return os.path.exists(f"{base}_meta.json")


def _latest_preprocessed_csv(directory: str) -> Optional[str]:
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if _is_preprocessed_csv(os.path.join(directory, name))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _manifest_path_for(dataset_path: str) -> str:
    base, _ = os.path.splitext(dataset_path)
    return f"{base}_manifest.csv"


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
        "feature_columns": feature_columns,
    }


def _train_from_preprocessed_csv(
    dataset_path: str,
    *,
    results_dir: str,
    models_dir: str,
    contamination: float,
    base_estimators: int,
    progress_cb=None,
) -> dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"预处理数据集不存在: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    if df.empty:
        raise RuntimeError("预处理数据集为空，无法训练。")

    manifest_path = _manifest_path_for(dataset_path)
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

    if "__source_file__" in df.columns:
        manifest_col = "__source_file__"
    elif "pcap_file" in df.columns:
        manifest_col = "pcap_file"

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
    - 支持直接加载预处理 CSV 或 NPZ 数据集
    - 对 PCAP 目录进行特征提取（多线程）后再训练
    """

    _ensure_dirs(results_dir, models_dir)

    dataset_path: Optional[str] = None
    pcap_dir: Optional[str] = None
    pcap_files: List[str] = []

    if not split_dir:
        raise FileNotFoundError("未提供训练数据路径")

    if os.path.isfile(split_dir):
        if _is_preprocessed_csv(split_dir):
            dataset_path = split_dir
        elif _is_npz(split_dir):
            dataset_path = split_dir
        elif split_dir.lower().endswith((".pcap", ".pcapng")):
            pcap_dir = os.path.dirname(split_dir) or os.path.abspath(os.path.join(split_dir, os.pardir))
            pcap_files = [split_dir]
        else:
            raise FileNotFoundError(f"不支持的训练文件: {split_dir}")
    elif os.path.isdir(split_dir):
        dataset_path = _latest_preprocessed_csv(split_dir)
        if dataset_path is None:
            dataset_path = _latest_npz_in_dir(split_dir)
        if dataset_path is None:
            pcap_dir = split_dir
    else:
        raise FileNotFoundError(f"路径不存在: {split_dir}")

    if dataset_path:
        if dataset_path.lower().endswith(".csv"):
            return _train_from_preprocessed_csv(
                dataset_path,
                results_dir=results_dir,
                models_dir=models_dir,
                contamination=contamination,
                base_estimators=base_estimators,
                progress_cb=progress_cb,
            )
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

        feature_csvs = extract_features_dir(
            pcap_dir,
            feature_dir,
            workers=workers,
            progress_cb=_extract_progress,
        )

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
