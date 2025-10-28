# -*- coding: utf-8 -*-
import argparse, json, os
import pandas as pd

# 用你项目里的标注工具
from src.functions.annotations import upsert_annotation
# 读取 analysis 的候选列名时保持一致
SCORE_CANDIDATES = ("anomaly_score", "score", "iforest_score")
VOTE_COL = "vote_ratio"

def _pick_score_col(df: pd.DataFrame) -> str:
    for c in SCORE_CANDIDATES:
        if c in df.columns:
            return c
    # 兜底：找名字里带 score 的
    for c in df.columns:
        if "score" in c.lower():
            return c
    raise RuntimeError("未找到分数字段，请确认预测CSV包含 anomaly_score/score 列")

def _load_metadata(path: str | None) -> dict:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_csv", help="预测结果CSV（pipeline_service predict 的输出）")
    ap.add_argument("--metadata", help="模型元数据JSON（训练产物）")
    ap.add_argument("--mode", choices=["conservative","balanced","aggressive"],
                    default="conservative", help="自动标注策略强度")
    ap.add_argument("--write-benign", action="store_true",
                    help="同时自动标注高置信度正常样本(label=0)")
    args = ap.parse_args()

    df = pd.read_csv(args.results_csv, encoding="utf-8")
    score_col = _pick_score_col(df)
    vote_col = VOTE_COL if VOTE_COL in df.columns else None

    meta = _load_metadata(args.metadata)
    thr = float(meta.get("threshold", float(df[score_col].quantile(0.05))))
    vote_thr = float(meta.get("vote_threshold", 0.5))
    # 分数越小越异常：你项目的最终判定是 score<=threshold 且 vote>=vote_threshold 才算异常
    # （与训练/推理一致）

    # 策略参数
    if args.mode == "conservative":
        p_anom = 0.01   # 仅拿最异常的1%
        p_norm = 0.99   # 仅拿最正常的1%（可选）
        vote_hi = max(vote_thr, 0.7)
        vote_lo = 0.3
    elif args.mode == "balanced":
        p_anom = 0.03
        p_norm = 0.97
        vote_hi = max(vote_thr, 0.6)
        vote_lo = 0.4
    else:  # aggressive
        p_anom = 0.05
        p_norm = 0.95
        vote_hi = max(vote_thr, 0.5)
        vote_lo = 0.5

    # 计算分位点，低分=更异常
    q_anom = float(df[score_col].quantile(p_anom))
    q_norm = float(df[score_col].quantile(p_norm))

    # 高置信度恶意：明显低于训练阈值/分位点，且投票高
    if vote_col:
        mask_anom = (df[score_col] <= min(thr, q_anom)) & (df[vote_col] >= vote_hi)
    else:
        mask_anom = (df[score_col] <= min(thr, q_anom))

    # 高置信度正常（可选）：明显高分且投票低
    if args.write_benign:
        if vote_col:
            mask_norm = (df[score_col] >= q_norm) & (df[vote_col] <= vote_lo)
        else:
            mask_norm = (df[score_col] >= q_norm)
    else:
        mask_norm = pd.Series([False]*len(df))

    n_pos = int(mask_anom.sum())
    n_neg = int(mask_norm.sum())

    for _, row in df[mask_anom].iterrows():
        upsert_annotation(row.to_dict(), label=1.0, annotator="auto", notes=f"auto:{args.mode}")

    for _, row in df[mask_norm].iterrows():
        upsert_annotation(row.to_dict(), label=0.0, annotator="auto", notes=f"auto:{args.mode}")

    print(f"[auto-annotate] positive={n_pos}, benign={n_neg}, total={n_pos+n_neg}")

if __name__ == "__main__":
    main()
