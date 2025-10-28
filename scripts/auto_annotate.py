"""Utility helpers for automatically annotating high-confidence predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.functions.annotations import annotation_summary, upsert_annotation
from src.functions.csv_utils import read_csv_flexible


def _pick_numeric(df: pd.DataFrame, *names: str) -> Optional[pd.Series]:
    for name in names:
        if name in df.columns:
            series = pd.to_numeric(df[name], errors="coerce")
            return series
    return None


def auto_annotate(
    out_csv: str,
    mode: str = "conservative",
    write_benign: bool = False,
    top_k: Optional[int] = None,
) -> dict:
    """Annotate high-confidence predictions into the global labels store."""

    path = Path(out_csv)
    if not path.exists():
        raise FileNotFoundError(f"未找到预测结果文件：{out_csv}")

    df = read_csv_flexible(path)
    if df.empty:
        stats = annotation_summary() or {}
        stats.update({"added_anomalies": 0, "added_normals": 0})
        return stats

    vote = _pick_numeric(df, "vote_ratio", "vote", "vote_score")

    risk_score = _pick_numeric(df, "risk_score")
    anomaly_score = _pick_numeric(df, "anomaly_score")

    if risk_score is not None and risk_score.notna().any():
        score = risk_score
        higher_is_more_anom = True
    elif anomaly_score is not None and anomaly_score.notna().any():
        score = anomaly_score
        higher_is_more_anom = False
    else:
        raise ValueError("预测结果缺少 anomaly_score 或 risk_score 列，无法自动打标签。")

    size = len(df)
    if mode not in {"conservative", "balanced"}:
        mode = "conservative"

    if mode == "conservative":
        default_k = max(100, int(size * 0.01))
        vote_threshold = 0.7
        benign_quantile = 0.97
    else:
        default_k = max(300, int(size * 0.03))
        vote_threshold = 0.6
        benign_quantile = 0.95

    if top_k is None:
        top_k = default_k
    top_k = max(0, int(top_k))

    score_valid = score.dropna()
    effective_k = min(len(score_valid), top_k)

    if effective_k > 0:
        if higher_is_more_anom:
            mal_idx = score_valid.nlargest(effective_k).index
        else:
            mal_idx = score_valid.nsmallest(effective_k).index
    else:
        mal_idx = pd.Index([])

    mal_mask = df.index.isin(mal_idx)
    if vote is not None:
        vote_mask = vote.fillna(1.0) >= vote_threshold
        mal_mask &= vote_mask

    ben_mask = None
    if write_benign:
        base_series = anomaly_score if anomaly_score is not None else score
        base_series = base_series.dropna()
        if not base_series.empty:
            if higher_is_more_anom:
                cutoff = base_series.quantile(1 - benign_quantile)
                if pd.isna(cutoff):
                    ben_mask = pd.Series(False, index=df.index)
                else:
                    ben_mask = (base_series <= cutoff).reindex(df.index, fill_value=False)
            else:
                cutoff = base_series.quantile(benign_quantile)
                if pd.isna(cutoff):
                    ben_mask = pd.Series(False, index=df.index)
                else:
                    ben_mask = (base_series >= cutoff).reindex(df.index, fill_value=False)
        else:
            ben_mask = pd.Series(False, index=df.index)

    added_pos = 0
    added_neg = 0
    if mal_mask.any():
        for _, row in df[mal_mask].iterrows():
            upsert_annotation(row.to_dict(), label=1.0, annotator="auto", notes=f"auto:{mode}")
            added_pos += 1

    if isinstance(ben_mask, pd.Series):
        aligned_mask = ben_mask.reindex(df.index, fill_value=False)
        for _, row in df[aligned_mask].iterrows():
            upsert_annotation(row.to_dict(), label=0.0, annotator="auto", notes=f"auto:{mode}")
            added_neg += 1

    stats = annotation_summary() or {}
    stats.update({"added_anomalies": added_pos, "added_normals": added_neg})
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto annotate prediction results.")
    parser.add_argument("out_csv", help="Path to prediction output CSV")
    parser.add_argument("--mode", default="conservative", choices=["conservative", "balanced"], help="Selection mode")
    parser.add_argument("--write-benign", action="store_true", help="Also write high-confidence benign samples")
    parser.add_argument("--top_k", type=int, help="Override number of anomaly samples to annotate")
    args = parser.parse_args()

    result = auto_annotate(args.out_csv, args.mode, args.write_benign, args.top_k)
    print(result)
