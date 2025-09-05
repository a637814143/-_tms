# src/functions/analyze_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

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

    summary = (
        df.groupby("pcap_file")
        .agg(malicious_ratio=("is_malicious", "mean"),
             total=("is_malicious", "count"))
        .reset_index()
        .sort_values("malicious_ratio", ascending=False)
    )
    top10 = summary.head(10)

    plt.figure(figsize=(10,6))
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

    plt.figure(figsize=(8,5))
    plt.hist(df["anomaly_score"], bins=50, alpha=0.8)
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

    if progress_cb: progress_cb(100)
    return {"plots": [out1, out2], "top20_csv": out3}

