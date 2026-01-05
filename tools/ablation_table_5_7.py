# table_5_7_run_direct.py
# 直接运行：计算表5-7（纯模型 / 纯规则 / 融合-常规 / 融合-高敏）
# 输出：table_5_7_for_thesis.csv（可直接抄表）

import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ===================== 你只需要改这里（固定地址） =====================
BASELINE_CSV = r"D:\pythonProject8\data\results\modelprediction\biao5-8\prediction_balanced_20000_Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX_20251229_200341_small_20260103_162411.csv"
AGGRESSIVE_CSV = r"D:\pythonProject8\data\results\modelprediction\biao5-8\prediction_balanced_20000_Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX_20251229_200341_small_20260103_162507.csv"

# 列名（按你现在导出的CSV）
LABEL_COL = "LabelBinary"          # 如果你的列是 label，就改成 "label"
MODEL_SCORE_COL = "malicious_score"
RULES_SCORE_COL = "rules_score"    # 0-100
FUSION_SCORE_COL = "fusion_score"  # 0-1
FUSION_DECISION_COL = "fusion_decision"  # 0/1（最终融合判定）

# 阈值设置
MODEL_THR = 0.5
RULE_THR_BASELINE = 65.0           # 0-100（常规档规则触发阈值）
RULE_THR_AGGRESSIVE = 25.0         # 0-100（高敏档规则触发阈值）——可选（脚本默认不输出第二条规则行）
TOP_PERCENT = 0.01                 # Top-1%
OUT_CSV = "table_5_7_for_thesis.csv"
# =====================================================================


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn + 1e-12))


def top_hit_rate(y_true, score, top_percent=0.01):
    """
    Top-1% 命中率：当分数大量并列（比如很多100）时，加入极小抖动让排序可复现。
    """
    n = len(score)
    k = max(1, int(math.ceil(n * top_percent)))
    jitter = (np.arange(n) * 1e-12).astype(float)
    idx = np.argsort(-(score + jitter))[:k]
    return float(np.mean(y_true[idx] == 1)), k


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def eval_from_scores(name, y_true, score, thr):
    y_pred = (score >= thr).astype(int)
    hit, _ = top_hit_rate(y_true, score, TOP_PERCENT)
    return {
        "组别": name,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, score),
        "误报率(FPR)": fpr(y_true, y_pred),
        f"Top-{int(TOP_PERCENT*100)}%命中率": hit,
    }


def eval_from_decision(name, y_true, score, decision):
    hit, _ = top_hit_rate(y_true, score, TOP_PERCENT)
    return {
        "组别": name,
        "F1": float(f1_score(y_true, decision, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, score),
        "误报率(FPR)": fpr(y_true, decision),
        f"Top-{int(TOP_PERCENT*100)}%命中率": hit,
    }


def main():
    base = pd.read_csv(BASELINE_CSV)
    aggr = pd.read_csv(AGGRESSIVE_CSV)

    # 检查列
    need_cols = [LABEL_COL, MODEL_SCORE_COL, RULES_SCORE_COL, FUSION_SCORE_COL, FUSION_DECISION_COL]
    for c in need_cols:
        if c not in base.columns:
            raise ValueError(f"[Baseline] 缺少列：{c}，当前列：{base.columns.tolist()}")
        if c not in aggr.columns:
            raise ValueError(f"[Aggressive] 缺少列：{c}，当前列：{aggr.columns.tolist()}")

    y = base[LABEL_COL].astype(int).to_numpy()
    y2 = aggr[LABEL_COL].astype(int).to_numpy()
    if len(y) != len(y2) or not np.all(y == y2):
        print("[WARN] 两份CSV的label不一致/长度不一致：请确认是同一测试集导出。")

    model_score = base[MODEL_SCORE_COL].astype(float).clip(0, 1).to_numpy()

    rules_base = (base[RULES_SCORE_COL].astype(float) / 100.0).clip(0, 1).to_numpy()
    fusion_base = base[FUSION_SCORE_COL].astype(float).clip(0, 1).to_numpy()
    decision_base = base[FUSION_DECISION_COL].astype(int).to_numpy()

    rules_aggr = (aggr[RULES_SCORE_COL].astype(float) / 100.0).clip(0, 1).to_numpy()
    fusion_aggr = aggr[FUSION_SCORE_COL].astype(float).clip(0, 1).to_numpy()
    decision_aggr = aggr[FUSION_DECISION_COL].astype(int).to_numpy()

    rows = []

    # E：纯模型（离线消融）
    rows.append(eval_from_scores("E 纯模型", y, model_score, MODEL_THR))

    # H：纯规则（离线消融：按常规阈值）
    rows.append(eval_from_scores("H 纯规则(常规阈值)", y, rules_base, RULE_THR_BASELINE / 100.0))

    # F：融合-常规（使用系统最终融合判定）
    rows.append(eval_from_decision("F 融合Baseline(常规)", y, fusion_base, decision_base))

    # G：融合-高敏（使用系统最终融合判定）
    rows.append(eval_from_decision("G 融合Aggressive(高敏)", y, fusion_aggr, decision_aggr))

    out = pd.DataFrame(rows)
    cols = ["组别", "F1", "ROC-AUC", "误报率(FPR)", f"Top-{int(TOP_PERCENT*100)}%命中率"]

    print("\n===== 表5-7 可直接抄写 =====")
    print(out[cols].to_string(index=False))

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{OUT_CSV}")

    # 附加：打印规则触发情况（方便你写论文解释）
    trig_rate_base = float((base[RULES_SCORE_COL].astype(float) > 0).mean())
    trig_rate_aggr = float((aggr[RULES_SCORE_COL].astype(float) > 0).mean())
    print(f"\n[INFO] rules_score>0 触发比例：baseline={trig_rate_base:.4%}, aggressive={trig_rate_aggr:.4%}")


if __name__ == "__main__":
    main()
