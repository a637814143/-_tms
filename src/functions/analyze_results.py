# src/functions/analyze_results.py
from __future__ import annotations

import json
import math
import os

import matplotlib

# 使用无界面的后端，避免在线程中绘图触发 GUI 报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

def analyze_results(results_csv: str, out_dir: str, progress_cb=None) -> dict:
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

    # 3) Top20 异常包（分数越小越异常）
    top20 = df.sort_values("anomaly_score").head(20)
    out3 = os.path.join(out_dir, "top20_packets.csv")
    top20.to_csv(out3, index=False, encoding="utf-8")

    # -------- 补充信息 --------
    raw_quantiles = score_series.quantile([0.01, 0.05, 0.1, 0.5, 0.9]).to_dict()
    anomaly_score_quantiles = {str(k): _clean_number(v) for k, v in raw_quantiles.items()}
    score_threshold = anomaly_score_quantiles.get("0.05")
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

    summary_text = "\n".join(summary_lines)

    details_payload = {
        "total_rows": total_rows,
        "unique_files": unique_files,
        "malicious_total": malicious_total,
        "malicious_ratio": malicious_ratio_global,
        "score_threshold": score_threshold,
        "ratio_threshold": ratio_threshold,
        "anomalous_files": anomalous_files,
        "anomaly_score_quantiles": anomaly_score_quantiles,
        "single_file_status": single_file_status,
        "summary_csv": summary_path,
    }

    export_payload = {
        "anomaly_count": malicious_total,
        "anomaly_files": [str(item.get("pcap_file")) for item in anomalous_files if item.get("pcap_file")],
        "single_file_status": single_file_status,
    }

    summary_json_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary_text, "details": details_payload}, fh, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb(100)

    payload = {
        "plots": [p for p in (out1, out2) if p],
        "top20_csv": out3,
        "out_dir": out_dir,
        "summary_csv": summary_path,
        "summary_json": summary_json_path,
        "summary_text": summary_text,
        "metrics": details_payload,
        "export_payload": export_payload,
    }

    return payload