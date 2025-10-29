# -*- coding: utf-8 -*-
"""
分析结果并可视化（修正版，避免 Series 报错）
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _pick_col(df: pd.DataFrame, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_file_column(df: pd.DataFrame) -> str:
    col = _pick_col(df, "pcap_file", "file", "source_file", "filename")
    if col is None:
        df["pcap_file"] = "single"
        return "pcap_file"
    if col != "pcap_file":
        df["pcap_file"] = df[col]
    return "pcap_file"


def _ensure_is_malicious(df: pd.DataFrame) -> str:
    if "is_malicious" in df.columns:
        return "is_malicious"
    pred = _pick_col(df, "prediction", "pred", "label")
    if pred is not None:
        df["is_malicious"] = (df[pred] == -1).astype(bool)
        return "is_malicious"
    risk = _pick_col(df, "risk_score", "risk", "score")
    anom = _pick_col(df, "anomaly_score", "iforest_score")
    if risk is not None:
        s = pd.to_numeric(df[risk], errors="coerce").fillna(0.0)
        df["is_malicious"] = (s > 0)
        return "is_malicious"
    if anom is not None:
        s = pd.to_numeric(df[anom], errors="coerce").fillna(0.0)
        thr = float(np.nanmedian(s.values))
        df["is_malicious"] = (s > thr)
        return "is_malicious"
    df["is_malicious"] = False
    return "is_malicious"


def analyze_results(results_csv: str, out_dir: str, progress_cb=None):
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"未找到结果文件: {results_csv}")
    _ensure_dir(out_dir)

    if progress_cb: progress_cb(10)
    df = pd.read_csv(results_csv, encoding="utf-8")
    if df.empty:
        raise RuntimeError("结果 CSV 为空")

    file_col = _ensure_file_column(df)
    mal_col = _ensure_is_malicious(df)
    risk_col = _pick_col(df, "risk_score", "risk", "score")
    anom_col = _pick_col(df, "anomaly_score", "iforest_score")

    # —— 文件级 Top10 恶意比例 —— #
    if progress_cb: progress_cb(30)
    if "flow_id" in df.columns:
        cnt = df.groupby(file_col)[mal_col].agg(["mean", "count"]).reset_index()
    else:
        cnt = df.groupby(file_col)[mal_col].agg(["mean"]).reset_index()
        cnt["count"] = df.groupby(file_col)[mal_col].size().values

    cnt = cnt.rename(columns={"mean": "malicious_ratio", "count": "total_flows"})

    if risk_col is not None:
        p95 = (
            df.groupby(file_col)[risk_col]
            .apply(lambda s: float(np.percentile(pd.to_numeric(s, errors="coerce").fillna(0.0).values, 95)))
        )
        cnt = cnt.merge(p95.rename("risk_p95"), on=file_col, how="left")
    else:
        cnt["risk_p95"] = np.nan

    top10 = cnt.sort_values("malicious_ratio", ascending=False).head(10)
    out_path1 = os.path.join(out_dir, "top10_malicious_ratio.png")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=top10, x=file_col, y="malicious_ratio")
    plt.xticks(rotation=45, ha="right")
    plt.title("恶意比例 Top10（按文件）")
    plt.tight_layout()
    plt.savefig(out_path1, dpi=180)
    plt.close()

    # —— 分数分布 —— #
    if progress_cb: progress_cb(60)
    out_path2 = os.path.join(out_dir, "anomaly_score_distribution.png")
    use_col = risk_col if risk_col is not None else anom_col
    if use_col is not None:
        plt.figure(figsize=(8, 4.5))
        sns.histplot(pd.to_numeric(df[use_col], errors="coerce").fillna(0.0).values, bins=50, kde=True)
        plt.title("风险分布（分数越高越危险）" if use_col == risk_col else "异常分数分布（越小越异常）")
        plt.xlabel(use_col)
        plt.tight_layout()
        plt.savefig(out_path2, dpi=180)
        plt.close()
    else:
        out_path2 = None

    # —— 规则理由统计 —— #
    if progress_cb: progress_cb(75)
    out_path3 = os.path.join(out_dir, "rules_reason_counts.png")
    if "rules_reasons" in df.columns:
        reasons = []
        for x in df["rules_reasons"].astype(str):
            if not x:
                continue
            for r in [i.strip() for i in x.split(";") if i.strip()]:
                reasons.append(r)
        if reasons:
            s = pd.Series(reasons).value_counts().head(10)
            plt.figure(figsize=(8, 4.5))
            sns.barplot(x=s.values, y=s.index)
            plt.title("规则触发 Top10")
            plt.xlabel("次数")
            plt.tight_layout()
            plt.savefig(out_path3, dpi=180)
            plt.close()
    if not os.path.exists(out_path3):
        out_path3 = None

    # —— Top20 可疑流 —— #
    if progress_cb: progress_cb(90)
    if risk_col is not None:
        top20 = df.sort_values(by=risk_col, ascending=False).head(20)
    elif anom_col is not None:
        top20 = df.sort_values(by=anom_col, ascending=True).head(20)
    else:
        sort_cols = [mal_col]
        ascending = [False]
        if "bytes" in df.columns:
            sort_cols.append("bytes"); ascending.append(False)
        if "pkt_count" in df.columns:
            sort_cols.append("pkt_count"); ascending.append(False)
        top20 = df.sort_values(by=sort_cols, ascending=ascending).head(20)

    out_path4 = os.path.join(out_dir, "top20_flows.csv")
    top20.to_csv(out_path4, index=False, encoding="utf-8")

    summary_csv = os.path.join(out_dir, "summary_by_file.csv")
    cnt.to_csv(summary_csv, index=False, encoding="utf-8")

    if progress_cb: progress_cb(100)

    artifacts = [p for p in [out_path1, out_path2, out_path3, out_path4] if p]
    return {"artifacts": artifacts, "summary": summary_csv}
