# src/functions/unsupervised_train.py
import os
import glob
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from joblib import dump

from src.functions.feature_extractor import extract_features

NUMERIC_FEATURES = ["length"]  # 采用轻量特征：每包长度

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

def train_unsupervised_on_split(
    split_dir: str,
    results_dir: str,
    models_dir: str,
    contamination: float = 0.05,
    base_estimators: int = 50,
    progress_cb=None
):
    """
    无监督训练：顺序处理 split_dir 下每个 pcap，生成/读取 packet-level 特征，合并训练 IsolationForest。
    输出：results/iforest_results.csv, results/summary_by_file.csv, models/isoforest.joblib, models/scaler.joblib
    """
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"目录不存在: {split_dir}")

    _ensure_dirs(results_dir, models_dir)
    feature_dir = os.path.join(results_dir, "features")
    os.makedirs(feature_dir, exist_ok=True)

    pcaps = sorted(glob.glob(os.path.join(split_dir, "*.pcap"))) + \
            sorted(glob.glob(os.path.join(split_dir, "*.pcapng")))

    if len(pcaps) == 0:
        raise RuntimeError(f"未在目录中发现 pcap/pcapng: {split_dir}")

    # 1) 逐文件生成/复用特征 CSV
    feature_csvs = []
    for i, pcap in enumerate(pcaps, 1):
        out_csv = _load_or_extract_feature_csv(pcap, feature_dir)
        feature_csvs.append(out_csv)
        if progress_cb:
            progress_cb(int( min(50, 50 * i / len(pcaps)) ))

    # 2) 聚合特征
    dfs = []
    for csv_path in feature_csvs:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            # 数值清洗
            for c in NUMERIC_FEATURES:
                if c in df.columns:
                    df[c] = df[c].apply(_safe_float)
                else:
                    df[c] = 0.0
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] 读取特征失败 {csv_path}: {e}")

    if len(dfs) == 0:
        raise RuntimeError("未能读取到任何特征 CSV。")

    full_df = pd.concat(dfs, ignore_index=True)

    # 3) 特征选择与标准化
    X = full_df[NUMERIC_FEATURES].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) IsolationForest 顺序训练（warm_start 用于可视化进度）
    rounds = max(1, math.ceil(len(pcaps)))
    model = IsolationForest(
        n_estimators=base_estimators,
        contamination=contamination,
        random_state=42,
        warm_start=True,
        n_jobs=-1
    )
    model.fit(X_scaled)
    if progress_cb:
        progress_cb(60)

    current_estimators = base_estimators
    for r in range(2, rounds + 1):
        current_estimators += base_estimators
        model.set_params(n_estimators=current_estimators)
        model.fit(X_scaled)
        if progress_cb:
            progress_cb(60 + int(30 * r / rounds))

    # 5) 推断
    scores = model.decision_function(X_scaled)  # 分数越小越异常
    preds = model.predict(X_scaled)             # 1=正常，-1=异常
    is_malicious = (preds == -1).astype(int)

    full_df["anomaly_score"] = scores
    full_df["is_malicious"] = is_malicious

    # 6) 导出结果
    results_csv = os.path.join(results_dir, "iforest_results.csv")
    full_df.to_csv(results_csv, index=False, encoding="utf-8")

    summary = (
        full_df.groupby("pcap_file")
        .agg(total_packets=("pcap_file", "count"),
             malicious_packets=("is_malicious", "sum"),
             malicious_ratio=("is_malicious", "mean"))
        .reset_index()
        .sort_values("malicious_ratio", ascending=False)
    )
    summary_csv = os.path.join(results_dir, "summary_by_file.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    # 7) 保存模型与 scaler
    model_path  = os.path.join(models_dir, "isoforest.joblib")
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
        "packets": len(full_df),
        "malicious": int(full_df["is_malicious"].sum()),
        "contamination": contamination
    }



