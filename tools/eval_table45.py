# tools/eval_table45_interactive.py
import os
import pandas as pd


# ----------------------------
# Column picking (robust)
# ----------------------------
def _norm(s: str) -> str:
    """Normalize column names for matching."""
    if s is None:
        return ""
    # remove BOM, trim, lowercase, remove all whitespace
    return "".join(str(s).replace("\ufeff", "").strip().lower().split())


def pick_col(cols, candidates):
    """
    Find a column from cols that matches any candidate.
    - Ignores casing and whitespace (including internal spaces/newlines).
    - Keeps original column name for accessing dataframe.
    """
    if not cols:
        return None

    col_map = {_norm(c): c for c in cols}
    cand_norm = [_norm(c) for c in candidates]

    for cn in cand_norm:
        if cn in col_map:
            return col_map[cn]

    # fallback: try strip match
    for c in cols:
        if str(c).strip() in candidates:
            return c
    return None


# ----------------------------
# Label / prediction parsing
# ----------------------------
_BENIGN_TOKENS = {
    "benign", "normal", "0", "false", "neg", "negative", "no", "none", "clean"
}
_MALICIOUS_TOKENS = {
    "1", "true", "pos", "positive", "yes", "attack", "malicious", "anomaly", "abnormal", "intrusion"
}


def binarize_truth(df: pd.DataFrame):
    cols = df.columns.tolist()
    truth_col = pick_col(
        cols,
        [
            "LabelBinary", "labelbinary",
            "Label", "label",
            "class", "Class",
            "ground_truth", "GroundTruth", "Ground Truth",
            "y_true", "truth",
        ],
    )
    if truth_col is None:
        raise ValueError("找不到真值列：需要 LabelBinary 或 Label（BENIGN/攻击名）等")

    s = df[truth_col]

    # 1) numeric-like -> >0 means positive
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        y_true = (num.fillna(0.0) > 0).astype(int)
        return y_true, truth_col

    # 2) string-like: BENIGN/normal -> 0, others -> 1
    t = s.fillna("").astype(str).str.strip().str.lower()
    y_true = ((t != "") & (~t.isin(_BENIGN_TOKENS))).astype(int)
    return y_true, truth_col


def binarize_pred(series: pd.Series) -> pd.Series:
    """
    Robustly convert prediction column to 0/1.
    Supports:
    - numeric 0/1, -1/1, probability score
    - boolean True/False
    - strings: benign/malicious/attack/...
    """
    s = series

    # boolean dtype
    if s.dtype == bool:
        return s.astype(int)

    # numeric-like
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return (num.fillna(0.0) > 0).astype(int)

    # string-like
    t = s.fillna("").astype(str).str.strip().str.lower()
    is_benign = t.isin(_BENIGN_TOKENS) | t.str.contains("benign") | t.str.contains("normal")
    is_mal = (
        t.isin(_MALICIOUS_TOKENS)
        | t.str.contains("attack")
        | t.str.contains("malicious")
        | t.str.contains("anomal")
        | t.str.contains("intrus")
    )
    y = pd.Series(0, index=t.index, dtype=int)
    y[is_mal] = 1
    y[(~is_benign) & (t != "")] = 1
    return y


def load_preds(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")

    df = pd.read_csv(path)
    cols = df.columns.tolist()

    pred_col = pick_col(
        cols,
        ["prediction_status", "pred", "y_pred", "prediction", "predict", "label_pred", "yhat"]
    )
    if pred_col is None:
        raise ValueError(f"{os.path.basename(path)} 找不到预测列（例如 prediction_status / pred / y_pred）")

    score_col = pick_col(
        cols,
        ["fusion_score", "malicious_score", "score", "anomaly_score", "prob", "proba", "probability"]
    )
    # score_col can be None

    y_true, truth_col = binarize_truth(df)
    y_pred = binarize_pred(df[pred_col])

    y_score = None
    if score_col is not None:
        y_score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)

    return df, y_true, y_pred, y_score, truth_col, pred_col, score_col


# ----------------------------
# Metrics
# ----------------------------
def safe_metrics(y_true, y_pred, y_score):
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
    if y_score is not None and len(set(pd.Series(y_true).tolist())) == 2:
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


def _round_or_blank(x, nd=4):
    if x is None:
        return ""
    try:
        return round(float(x), nd)
    except Exception:
        return ""


def eval_one(tag: str, model_type: str, profile: str, path: str):
    _, y_true, y_pred, y_score, truth_col, pred_col, score_col = load_preds(path)
    m = safe_metrics(y_true, y_pred, y_score)

    row = {
        "组": tag,
        "模型类型": model_type,
        "规则档位": profile,
        "文件": os.path.basename(path),
        "真值列": truth_col,
        "预测列": pred_col,
        "分数列": score_col if score_col else "",
        **m
    }
    # round display
    for k in ["Accuracy", "Precision", "Recall", "F1", "AUC", "AP"]:
        row[k] = _round_or_blank(row.get(k), 4)
    return row


def print_rows(rows):
    # terminal-friendly print
    cols = ["组", "模型类型", "规则档位", "文件", "Accuracy", "Precision", "Recall", "F1", "AUC", "AP", "TN", "FP", "FN", "TP"]
    df = pd.DataFrame(rows)
    df = df.loc[:, [c for c in cols if c in df.columns]]
    print("\n========== 表 4-5 指标汇总 ==========")
    print(df.to_string(index=False))
    print("====================================\n")


def main():
    print("请输入四个预测结果 CSV 的完整路径（可直接复制粘贴）：")
    a_path = input("A（UNSW CSV + Baseline）: ").strip().strip('"')
    b_path = input("B（UNSW CSV + Aggressive）: ").strip().strip('"')
    c_path = input("C（CICIDS/PCAP + Baseline）: ").strip().strip('"')
    d_path = input("D（CICIDS/PCAP + Aggressive）: ").strip().strip('"')

    groups = [
        ("A", "UNSW CSV", "Baseline", a_path),
        ("B", "UNSW CSV", "Aggressive", b_path),
        ("C", "CICIDS/PCAP", "Baseline", c_path),
        ("D", "CICIDS/PCAP", "Aggressive", d_path),
    ]

    rows = []
    for tag, model_type, profile, path in groups:
        try:
            rows.append(eval_one(tag, model_type, profile, path))
        except Exception as e:
            print(f"\n[{tag}] 计算失败：{e}\n路径：{path}\n")
            raise

    print_rows(rows)

    # Optional: Save summary
    out = input("是否保存汇总到 CSV？（直接回车=不保存；输入路径=保存）: ").strip().strip('"')
    if out:
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        print("已保存：", out)


if __name__ == "__main__":
    main()
