# src/functions/unsupervised_train.py
# -*- coding: utf-8 -*-
import os
import math
from typing import List, Dict, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from joblib import dump as joblib_dump

from .feature_extractor import extract_features, extract_features_dir

# 与 feature_extractor 输出字段保持一致（若不存在则跳过）
_NUM_FEATURES_PREF = [
    "flow_duration", "pkt_count",
    "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max",
    "inter_arrival_mean", "inter_arrival_std",
    "tcp_flag_count",
    "pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd",
    "pps", "bps", "pps_fwd", "pps_bwd",
    "fwd_bwd_ratio",
    "iat_fwd_mean", "iat_fwd_std", "iat_bwd_mean", "iat_bwd_std",
]


def _collect_feature_csvs(input_path: str, tmp_out_dir: str, workers: int = 8, progress_cb=None) -> List[str]:
    """
    根据顶层选择的 input_path（文件或目录）产出特征 CSV 列表。
    - 目录：并发提取到 tmp_out_dir，下游统一加载
    - 文件：单文件提取一个 CSV
    """
    os.makedirs(tmp_out_dir, exist_ok=True)
    if os.path.isdir(input_path):
        return extract_features_dir(input_path, tmp_out_dir, workers=workers, progress_cb=progress_cb)
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        csv = os.path.join(tmp_out_dir, f"{base}_features.csv")
        extract_features(input_path, csv, progress_cb=progress_cb)
        return [csv]


def _load_all_features(csv_list: List[str]) -> pd.DataFrame:
    """
    读取所有特征 CSV 并合并；同时保留来源以便统计
    """
    parts = []
    for p in csv_list:
        try:
            df = pd.read_csv(p, encoding="utf-8")
            # 保留来源：用于回填 file/pcap_file
            df["__src_csv__"] = os.path.basename(p)
            parts.append(df)
        except Exception:
            pass
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _ensure_file_alias_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一提供 file 与 pcap_file 两个等价列，避免下游只认其一时报错。
    优先级：若已有 file 列，pcap_file=file；若只有 __src_csv__ 列，也填充到两列。
    """
    df = df.copy()
    # 先推导基础 file 列
    if "file" not in df.columns:
        if "__src_csv__" in df.columns:
            df["file"] = df["__src_csv__"]
        else:
            df["file"] = "unknown"

    # 再补 pcap_file
    if "pcap_file" not in df.columns:
        df["pcap_file"] = df["file"]

    return df


def train_unsupervised_on_split(input_path: str,
                                results_dir: str,
                                models_dir: str,
                                progress_cb=None) -> Dict[str, str]:
    """
    input_path: 顶部路径（目录 或 单个 pcap 文件）
    返回：
      {
        results_csv, summary_csv, model_path, scaler_path,
        flows, malicious
      }
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if progress_cb: progress_cb(3)

    # 1) 提取特征（写在 results/_train_features）
    tmp_feat_dir = os.path.join(results_dir, "_train_features")
    csv_list = _collect_feature_csvs(input_path, tmp_feat_dir, workers=8, progress_cb=progress_cb)

    if progress_cb: progress_cb(25)

    # 2) 读入并拼接
    df = _load_all_features(csv_list)
    if df.empty:
        raise RuntimeError("训练数据为空，无法训练。")

    # 统一补齐 file/pcap_file
    df = _ensure_file_alias_cols(df)

    # 3) 选择数值特征
    cols = [c for c in _NUM_FEATURES_PREF if c in df.columns]
    if not cols:
        raise RuntimeError("未找到可用的数值特征列。")

    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    # 4) 归一化 + 训练
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if progress_cb: progress_cb(50)

    iforest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    iforest.fit(Xs)

    if progress_cb: progress_cb(70)

    pred = iforest.predict(Xs)                 # -1 异常 / 1 正常
    score = -iforest.decision_function(Xs)     # 越大越异常（取负号统一“越大越危险”）

    out_df = df.copy()
    out_df["prediction"] = pred
    out_df["anomaly_score"] = score

    # 5) 写结果（含 file / pcap_file）
    results_csv = os.path.join(results_dir, "iforest_results.csv")
    out_df.to_csv(results_csv, index=False, encoding="utf-8")

    # 6) 摘要（按文件聚合）
    try:
        summary = (
            out_df
            .assign(pcap_file=lambda d: d.get("pcap_file", d.get("file", d.get("__src_csv__", "unknown"))))
            .groupby("pcap_file", as_index=False)
            .agg(
                flows=("prediction", "size"),
                malicious=("prediction", lambda s: int((s == -1).sum())),
                avg_score=("anomaly_score", "mean")
            )
            .sort_values(["malicious", "avg_score"], ascending=[False, False])
        )
        summary_csv = os.path.join(results_dir, "summary_by_file.csv")
        summary.to_csv(summary_csv, index=False, encoding="utf-8")
        flows_n = int(summary["flows"].sum()) if "flows" in summary else len(out_df)
        mal_n = int(summary["malicious"].sum()) if "malicious" in summary else int((out_df["prediction"] == -1).sum())
    except Exception:
        summary_csv = os.path.join(results_dir, "summary_by_file.csv")
        out_df[["pcap_file", "prediction", "anomaly_score"]].to_csv(summary_csv, index=False, encoding="utf-8")
        flows_n = len(out_df)
        mal_n = int((out_df["prediction"] == -1).sum())

    if progress_cb: progress_cb(85)

    # 7) 保存模型与标准化器
    model_path = os.path.join(models_dir, "isoforest.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    joblib_dump(iforest, model_path)
    joblib_dump(scaler, scaler_path)

    if progress_cb: progress_cb(100)

    return {
        "results_csv": results_csv,
        "summary_csv": summary_csv,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "flows": flows_n,
        "malicious": mal_n,
    }
