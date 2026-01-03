import os, glob
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# 你的 prediction 结果目录（改成你自己的）
PRED_DIR = r"D:\pythonProject8\data\results\modelprediction\siwen"
OUT_CSV  = os.path.join(PRED_DIR, "table_5_4_summary.csv")

def label_to_binary(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str).str.strip().str.lower()
    # BENIGN/Normal/0 -> 0, 其它 -> 1
    benign = s2.isin(["benign", "normal", "0", "false", "negative", "neg", "none", "no", "clean"])
    return (~benign).astype(int)

def calc_one(path: str) -> dict:
    df = pd.read_csv(path)

    # score
    if "fusion_score" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} 缺少 fusion_score")
    score = pd.to_numeric(df["fusion_score"], errors="coerce").fillna(0.0)

    # truth
    truth_col = None
    for c in ["LabelBinary", "labelbinary", "Label", "label", "Class", "class"]:
        if c in df.columns:
            truth_col = c
            break
    if truth_col is None:
        raise ValueError(f"{os.path.basename(path)} 找不到真值列(LabelBinary/Label/...)")

    y = label_to_binary(df[truth_col])

    # AUC/AP（需要两类都存在）
    roc = roc_auc_score(y, score) if y.nunique() == 2 else None
    ap  = average_precision_score(y, score) if y.nunique() == 2 else None

    # Top1%命中率
    n = max(1, int(len(df) * 0.01))
    top = df.assign(_y=y, _s=score).nlargest(n, "_s")
    top1_hit = float(top["_y"].mean())

    return {
        "文件": os.path.basename(path),
        "样本数": len(df),
        "正类数(异常)": int(y.sum()),
        "正类占比": round(float(y.mean()), 6),
        "ROC-AUC": None if roc is None else round(float(roc), 4),
        "PR-AUC": None if ap is None else round(float(ap), 4),
        "Top1%命中率": round(top1_hit, 4),
        "Top1%样本数": n,
    }

def main():
    files = sorted(glob.glob(os.path.join(PRED_DIR, "prediction_*.csv")))
    if not files:
        raise SystemExit(f"目录下找不到 prediction_*.csv: {PRED_DIR}")

    rows = []
    for f in files:
        try:
            rows.append(calc_one(f))
        except Exception as e:
            rows.append({"文件": os.path.basename(f), "错误": str(e)})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n已保存汇总表：", OUT_CSV)

if __name__ == "__main__":
    main()
