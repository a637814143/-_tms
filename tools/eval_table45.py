# tools/eval_table45.py
import argparse
import os
import pandas as pd

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    # 兼容一些“带空格的列名”
    for c in cols:
        if c.strip() in candidates:
            return c
    return None

def binarize_truth(df):
    cols = df.columns.tolist()
    truth_col = pick_col(cols, ["LabelBinary", "labelbinary", "Label", "label", "class", "ground_truth", " Ground Truth", " Label"])
    if truth_col is None:
        raise ValueError("找不到真值列：需要 LabelBinary 或 Label（BENIGN/攻击名）等")

    s = df[truth_col]

    # 1) 如果能转成数值（0/1），直接用
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        y_true = (num.fillna(0.0) > 0).astype(int)
        return y_true, truth_col

    # 2) 否则按字符串：BENIGN/normal 为0，其它为1
    t = s.fillna("").astype(str).str.strip().str.lower()
    benign = {"benign", "normal", "0"}
    y_true = ((t != "") & (~t.isin(benign))).astype(int)
    return y_true, truth_col

def load_preds(path):
    df = pd.read_csv(path)
    cols = df.columns.tolist()

    pred_col = pick_col(cols, ["prediction_status", "pred", "y_pred", "Prediction", "predict", "label_pred"])
    if pred_col is None:
        raise ValueError(f"{os.path.basename(path)} 找不到预测标签列 prediction_status")

    score_col = pick_col(cols, ["fusion_score", "malicious_score", "score", "anomaly_score", "prob", "proba"])
    if score_col is None:
        # 没有分数也能算 Accuracy/PRF，只是算不了 AUC/AP
        score_col = None

    y_true, truth_col = binarize_truth(df)
    y_pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0).astype(int)

    y_score = None
    if score_col is not None:
        y_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    return df, y_true, y_pred, y_score, truth_col, pred_col, score_col

def safe_metrics(y_true, y_pred, y_score):
    # 基础指标（不依赖 sklearn）
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * prec * rec) / max(prec + rec, 1e-12)

    auc = None
    ap = None

    # AUC/AP 需要两类都有 + 需要分数列
    if y_score is not None and len(set(y_true.tolist())) == 2:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = float(roc_auc_score(y_true, y_score))
            ap = float(average_precision_score(y_true, y_score))
        except Exception:
            auc = None
            ap = None

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
        "AP": ap,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", required=True, help="A组预测结果csv（UNSW+Baseline）")
    parser.add_argument("--B", required=True, help="B组预测结果csv（UNSW+Aggressive）")
    parser.add_argument("--C", required=True, help="C组预测结果csv（CICIDS+Baseline）")
    parser.add_argument("--D", required=True, help="D组预测结果csv（CICIDS+Aggressive）")
    parser.add_argument("--out", default="table4_5_summary.csv", help="输出汇总表csv")
    args = parser.parse_args()

    groups = {
        "A": ("UNSW CSV", "Baseline", args.A),
        "B": ("UNSW CSV", "Aggressive", args.B),
        "C": ("CICIDS/PCAP", "Baseline", args.C),
        "D": ("CICIDS/PCAP", "Aggressive", args.D),
    }

    rows = []
    for g, (model_type, profile, path) in groups.items():
        _, y_true, y_pred, y_score, truth_col, pred_col, score_col = load_preds(path)
        m = safe_metrics(y_true, y_pred, y_score)
        row = {
            "组": g,
            "模型类型": model_type,
            "规则档位": profile,
            "真值列": truth_col,
            "预测列": pred_col,
            "分数列": score_col if score_col else "",
            **m
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    # 常用四舍五入显示
    for c in ["Accuracy", "Precision", "Recall", "F1", "AUC", "AP"]:
        if c in out_df.columns:
            out_df[c] = out_df[c].map(lambda x: "" if x is None else round(x, 4))

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("已生成：", args.out)
    print(out_df[["组","模型类型","规则档位","Accuracy","Precision","Recall","F1","AUC","AP","TN","FP","FN","TP"]])

if __name__ == "__main__":
    main()
