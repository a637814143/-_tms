# src/functions/predict_unsupervised.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
from joblib import load as joblib_load

from .feature_extractor import extract_features, extract_features_dir

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

def predict_with_model(input_path: str,
                       results_dir: str,
                       models_dir: str,
                       progress_cb=None) -> str:
    os.makedirs(results_dir, exist_ok=True)

    # 1) 加载模型和scaler
    model_path = os.path.join(models_dir, "isoforest.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise RuntimeError("未找到已训练的模型，请先训练！")
    iforest = joblib_load(model_path)
    scaler = joblib_load(scaler_path)

    # 2) 提取特征（临时目录）
    tmp_feat_dir = os.path.join(results_dir, "_predict_features")
    os.makedirs(tmp_feat_dir, exist_ok=True)
    if os.path.isdir(input_path):
        csv_list = extract_features_dir(input_path, tmp_feat_dir, workers=8, progress_cb=progress_cb)
    else:
        base = os.path.splitext(os.path.basename(input_path))[0]
        csv = os.path.join(tmp_feat_dir, f"{base}_features.csv")
        extract_features(input_path, csv, progress_cb=progress_cb)
        csv_list = [csv]

    # 3) 合并
    dfs = []
    for c in csv_list:
        try:
            df = pd.read_csv(c, encoding="utf-8")
            dfs.append(df)
        except Exception:
            pass
    if not dfs:
        raise RuntimeError("预测输入为空")
    df = pd.concat(dfs, ignore_index=True)

    # 4) 选择特征并预测
    cols = [c for c in _NUM_FEATURES_PREF if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    Xs = scaler.transform(X)

    pred = iforest.predict(Xs)             # -1 异常 / 1 正常
    score = -iforest.decision_function(Xs) # 越大越异常

    out_df = df.copy()
    out_df["prediction"] = pred
    out_df["anomaly_score"] = score

    # 5) 保存
    results_csv = os.path.join(results_dir, "prediction_results.csv")
    out_df.to_csv(results_csv, index=False, encoding="utf-8")

    return results_csv
