# ablation_table_5_7.py
import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# ========= 你要改的输入 =========
PRED_CSV = r"D:\pythonProject8\data\CSV\results\unsw_test_predictions.csv"  # 改成你的预测输出CSV
LABEL_COL = "LabelBinary"
MODEL_SCORE_COL = "model_score"     # 有的文件叫 malicious_score，按你的实际改
RULES_SCORE_COL = "rules_score"     # 0-100
TOP_PERCENT = 0.01                  # Top-1%

# 纯模型固定阈值（用于“固定阈值口径”的F1/FPR）
MODEL_THRESHOLD = 0.5

# 档位参数（与你论文表3-1一致）
BASELINE = dict(name="Baseline", w_m=0.85, w_r=0.15, t=0.75, trigger_threshold=65)
AGGRESSIVE = dict(name="Aggressive", w_m=0.25, w_r=0.75, t=0.30, trigger_threshold=25)

# 纯规则固定阈值（建议=Baseline 触发阈值/100）
RULES_THRESHOLD = BASELINE["trigger_threshold"] / 100.0
# ==============================


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn + 1e-12))


def top_hit_rate(y_true, score, top_percent=0.01):
    n = len(score)
    k = max(1, int(math.ceil(n * top_percent)))
    idx = np.argsort(-score)[:k]
    return float(np.mean(y_true[idx] == 1)), k


def decision_fixed(score, thr):
    return (score >= thr).astype(int)


def decision_top_percent(score, top_percent=0.01):
    n = len(score)
    k = max(1, int(math.ceil(n * top_percent)))
    cutoff = np.sort(score)[-k]   # 第k大的分数作为阈值（包含并列）
    return (score >= cutoff).astype(int)


def fuse(model_score, rules_score_0_100, profile):
    r = np.clip(rules_score_0_100 / 100.0, 0.0, 1.0)
    m = np.clip(model_score, 0.0, 1.0)
    return np.clip(profile["w_m"] * m + profile["w_r"] * r, 0.0, 1.0)


def eval_variant(name, y_true, score, fixed_thr):
    auc = safe_auc(y_true, score)

    # 固定阈值口径
    y_pred_fixed = decision_fixed(score, fixed_thr)
    f1_fixed = float(f1_score(y_true, y_pred_fixed, zero_division=0))
    fpr_fixed = fpr(y_true, y_pred_fixed)

    # Top-1%告警名额口径
    hit, k = top_hit_rate(y_true, score, TOP_PERCENT)
    y_pred_top = decision_top_percent(score, TOP_PERCENT)
    f1_top = float(f1_score(y_true, y_pred_top, zero_division=0))
    fpr_top = fpr(y_true, y_pred_top)

    return {
        "组别": name,
        "连续分数": "见脚本",
        "阈值判定口径": f"固定阈值({fixed_thr}) / Top-{int(TOP_PERCENT*100)}%",
        "F1": f1_fixed,
        "ROC-AUC": auc,
        "误报率(FPR)": fpr_fixed,
        f"Top-{int(TOP_PERCENT*100)}%命中率": hit,
        "F1@Top": f1_top,          # 论文表格不一定要填，可留作自己参考
        "FPR@Top": fpr_top,        # 论文表格不一定要填，可留作自己参考
        "TopK": int(k),
        "固定阈值": fixed_thr,
    }


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = pd.read_csv(PRED_CSV)

    for col in [LABEL_COL, MODEL_SCORE_COL, RULES_SCORE_COL]:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}，请检查你的CSV列名。")

    y_true = df[LABEL_COL].astype(int).to_numpy()
    model_score = df[MODEL_SCORE_COL].astype(float).to_numpy()
    rules_score = df[RULES_SCORE_COL].astype(float).to_numpy()

    # E：纯模型（w_r=0）
    score_E = model_score
    res_E = eval_variant("E 纯模型(w_r=0)", y_true, score_E, MODEL_THRESHOLD)

    # F：Baseline 融合（模型主导/轻规则）
    score_F = fuse(model_score, rules_score, BASELINE)
    res_F = eval_variant("F Baseline(w_m=0.85,w_r=0.15,t=0.75)", y_true, score_F, BASELINE["t"])

    # G：Aggressive 融合（规则增强/高敏）
    score_G = fuse(model_score, rules_score, AGGRESSIVE)
    res_G = eval_variant("G Aggressive(w_m=0.25,w_r=0.75,t=0.30)", y_true, score_G, AGGRESSIVE["t"])

    # Aggressive 的真实最终判定（fusion>=t OR rules>=trigger）
    y_pred_aggr_final = ((score_G >= AGGRESSIVE["t"]) | (rules_score >= AGGRESSIVE["trigger_threshold"])).astype(int)
    f1_final = float(f1_score(y_true, y_pred_aggr_final, zero_division=0))
    fpr_final = fpr(y_true, y_pred_aggr_final)

    # H：纯规则（w_m=0）
    score_H = np.clip(rules_score / 100.0, 0.0, 1.0)
    res_H = eval_variant("H 纯规则(w_m=0)", y_true, score_H, RULES_THRESHOLD)

    out = pd.DataFrame([res_E, res_F, res_G, res_H])

    # 论文表5-7要用的列（你可以直接抄）
    table_cols = ["组别", "F1", "ROC-AUC", "误报率(FPR)", f"Top-{int(TOP_PERCENT*100)}%命中率"]
    print("\n===== 表5-7 可直接抄写的结果 =====")
    print(out[table_cols].to_string(index=False))

    out.to_csv("table_5_7_ablation_full.csv", index=False)
    out[table_cols].to_csv("table_5_7_ablation_for_thesis.csv", index=False)

    print("\n已保存：table_5_7_ablation_for_thesis.csv（论文用）")
    print("已保存：table_5_7_ablation_full.csv（含Top口径的F1/FPR等扩展信息）")

    print("\n[说明] Aggressive 真实最终判定包含 OR 规则强触发：")
    print(f"  F1@final_aggressive = {f1_final:.6f}")
    print(f"  FPR@final_aggressive = {fpr_final:.6f}")
    print("建议你在论文表5-7的注释写明：Aggressive 的最终判定口径比仅 fusion>=t 更偏向高召回。")


if __name__ == "__main__":
    main()
