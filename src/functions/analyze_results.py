# src/functions/analyze_results.py

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

# 使用无界面的后端，避免在线程中绘图触发 GUI 报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.functions.logging_utils import get_logger

logger = get_logger(__name__)


def _resolve_data_base() -> str:
    env = os.getenv("MALDET_DATA_DIR")
    if env and env.strip():
        try:
            Path(env).expanduser().resolve().mkdir(parents=True, exist_ok=True)
            return str(Path(env).expanduser().resolve())
        except Exception:
            pass
    try:
        proj_root = Path(__file__).resolve().parents[2]
        local_data = proj_root / "data"
        local_data.mkdir(parents=True, exist_ok=True)
        return str(local_data)
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

    # 3) Top20 异常包（分数越小越异常）
    top20 = df.sort_values("anomaly_score").head(20)
    out3 = os.path.join(out_dir, "top20_packets.csv")
    top20.to_csv(out3, index=False, encoding="utf-8")

    # -------- 补充信息 --------
    raw_quantiles = score_series.quantile([0.01, 0.05, 0.1, 0.5, 0.9]).to_dict()
    anomaly_score_quantiles = {str(k): _clean_number(v) for k, v in raw_quantiles.items()}
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

    metrics_csv_path = None
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = os.path.join(out_dir, "threshold_evaluation.csv")
        metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")

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
        "plots": [p for p in (out1, out2) if p],
        "top20_csv": out3,
        "out_dir": out_dir,
        "summary_csv": summary_path,
        "summary_json": summary_json_path,
        "summary_text": summary_text,
        "metrics": details_payload,
        "export_payload": export_payload,
        "metrics_csv": metrics_csv_path,
    }

    return payload