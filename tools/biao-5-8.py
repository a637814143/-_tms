# biao-5-8.py
# 表5-8：PortScan 平衡子集的“阈值校准/约束阈值”对比
# 直接运行，固定路径。输出：table_5_8_threshold_calibration.csv

import math
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# ===================== 固定地址（已按你提供的路径写入） =====================
BASELINE_CSV = r"D:\pythonProject8\data\results\modelprediction\biao5-8\prediction_balanced_20000_Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX_20251229_200341_small_20260103_162411.csv"
AGGRESSIVE_CSV = r"D:\pythonProject8\data\results\modelprediction\biao5-8\prediction_balanced_20000_Friday-WorkingHours-Afternoon-PortScan_pcap_ISCX_20251229_200341_small_20260103_162507.csv"
# ==========================================================================

LABEL_COL = "Label"  # CICIDS: 'BENIGN' / 'PortScan'
MODEL_SCORE_COL = "malicious_score"
FUSION_SCORE_COL = "fusion_score"
FUSION_DECISION_COL = "fusion_decision"

TOP_PERCENT = 0.01
OUT_CSV = "table_5_8_threshold_calibration.csv"

# 约束：控制误报率（你可改成 0.1 或 0.3 等）
FPR_CONSTRAINT = 0.2


def to_binary_label(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.lower()
    return np.where(s == "benign", 0, 1).astype(int)


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn + 1e-12))


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def top_hit_rate(y_true, score, top_percent=0.01):
    n = len(score)
    k = max(1, int(math.ceil(n * top_percent)))
    jitter = (np.arange(n) * 1e-12).astype(float)  # 打破并列，保证可复现
    idx = np.argsort(-(score + jitter))[:k]
    return float(np.mean(y_true[idx] == 1)), k


def best_thr_for_f1(y_true, scores):
    thrs = np.linspace(0, 1, 1001)
    best_f1, best_t = -1.0, 0.5
    for t in thrs:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_f1), float(best_t)


def best_thr_for_f1_with_fpr_cap(y_true, scores, fpr_cap=0.2):
    """
    在 FPR <= fpr_cap 约束下，选 F1 最大的阈值。
    若无可行阈值，则返回 (nan, nan)。
    """
    thrs = np.linspace(0, 1, 1001)
    best_f1, best_t = -1.0, None
    for t in thrs:
        pred = (scores >= t).astype(int)
        cur_fpr = fpr(y_true, pred)
        if cur_fpr <= fpr_cap:
            cur_f1 = f1_score(y_true, pred, zero_division=0)
            if cur_f1 > best_f1:
                best_f1, best_t = cur_f1, t
    if best_t is None:
        return float("nan"), float("nan")
    return float(best_f1), float(best_t)


def row_from_score(name, y_true, score, thr):
    pred = (score >= thr).astype(int)
    hit, _ = top_hit_rate(y_true, score, TOP_PERCENT)
    return {
        "方案": name,
        "阈值": thr,
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, score),
        "误报率(FPR)": fpr(y_true, pred),
        f"Top-{int(TOP_PERCENT*100)}%命中率": hit,
    }


def row_from_decision(name, y_true, score, decision):
    hit, _ = top_hit_rate(y_true, score, TOP_PERCENT)
    return {
        "方案": name,
        "阈值": "系统决策列",
        "F1": float(f1_score(y_true, decision, zero_division=0)),
        "ROC-AUC": safe_auc(y_true, score),
        "误报率(FPR)": fpr(y_true, decision),
        f"Top-{int(TOP_PERCENT*100)}%命中率": hit,
    }


def main():
    base = pd.read_csv(BASELINE_CSV)
    aggr = pd.read_csv(AGGRESSIVE_CSV)

    y = to_binary_label(base[LABEL_COL])
    y2 = to_binary_label(aggr[LABEL_COL])
    if len(y) != len(y2) or not np.all(y == y2):
        print("[WARN] 两份CSV标签不一致/长度不一致：请确认同一测试子集导出。")

    m_base = base[MODEL_SCORE_COL].astype(float).clip(0, 1).to_numpy()
    f_base = base[FUSION_SCORE_COL].astype(float).clip(0, 1).to_numpy()
    d_base = base[FUSION_DECISION_COL].astype(int).to_numpy()

    f_aggr = aggr[FUSION_SCORE_COL].astype(float).clip(0, 1).to_numpy()
    d_aggr = aggr[FUSION_DECISION_COL].astype(int).to_numpy()

    rows = []

    # 1) 纯模型：默认阈值 0.5
    rows.append(row_from_score("E 纯模型(阈值0.5)", y, m_base, 0.5))

    # 2) 纯模型：最佳阈值（无约束）
    _, best_t_m = best_thr_for_f1(y, m_base)
    rows.append(row_from_score(f"E* 纯模型(最佳阈值 t*={best_t_m:.3f})", y, m_base, best_t_m))

    # 3) 纯模型：在FPR约束下的最佳阈值（更符合工程）
    _, best_t_m_cap = best_thr_for_f1_with_fpr_cap(y, m_base, fpr_cap=FPR_CONSTRAINT)
    if best_t_m_cap == best_t_m_cap:  # not nan
        rows.append(row_from_score(f"E** 纯模型(FPR≤{FPR_CONSTRAINT:.1f} 最佳阈值 t*={best_t_m_cap:.3f})", y, m_base, best_t_m_cap))

    # 4) 融合Baseline：系统决策列
    rows.append(row_from_decision("F 融合Baseline(系统决策)", y, f_base, d_base))

    # 5) 融合Baseline：最佳阈值（无约束）
    _, best_t_f = best_thr_for_f1(y, f_base)
    rows.append(row_from_score(f"F* 融合Baseline(最佳阈值 t*={best_t_f:.3f})", y, f_base, best_t_f))

    # 6) 融合Baseline：FPR约束下最佳阈值
    _, best_t_f_cap = best_thr_for_f1_with_fpr_cap(y, f_base, fpr_cap=FPR_CONSTRAINT)
    if best_t_f_cap == best_t_f_cap:
        rows.append(row_from_score(f"F** 融合Baseline(FPR≤{FPR_CONSTRAINT:.1f} 最佳阈值 t*={best_t_f_cap:.3f})", y, f_base, best_t_f_cap))

    # 7) 融合Aggressive：系统决策列（高敏）
    rows.append(row_from_decision("G 融合Aggressive(系统决策)", y, f_aggr, d_aggr))

    out = pd.DataFrame(rows)

    show_cols = ["方案", "阈值", "F1", "ROC-AUC", "误报率(FPR)", f"Top-{int(TOP_PERCENT*100)}%命中率"]
    print("\n===== 表5-8（阈值校准/约束阈值对比）可直接抄写 =====")
    print(out[show_cols].to_string(index=False))

    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{OUT_CSV}")

    print(f"\n[INFO] 正例比例={y.mean():.6f}（该子集为平衡抽样时应接近0.5）")
    print(f"[INFO] 误报约束：FPR ≤ {FPR_CONSTRAINT:.1f}")


if __name__ == "__main__":
    main()
