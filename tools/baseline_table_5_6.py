# baseline_table_5_6_safe.py
import os
# ===== 关键：避免 OpenMP/MKL 冲突导致的 -1 崩溃（必须放在 sklearn 之前）=====
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ==========================================================================

import math
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ========= 你要填的输入 =========
TRAIN_CSV = r"D:\pythonProject8\data\CSV\feature\UNSW_NB15_train_with_label.csv"
TEST_CSV  = r"D:\pythonProject8\data\CSV\feature\UNSW_NB15_test_with_label.csv"
LABEL_COL = "LabelBinary"          # 真实标签列（0/1）
TOP_PERCENT = 0.01                 # Top-1%
THRESHOLD = 0.5                    # 固定阈值（算F1/FPR）
THIRD_BASELINE = "lr"              # "lr" 或 "knn"
DROP_COLS = []                     # 非数值列、id列可加这里，如 ["id","Flow ID"]
# ==============================

def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return fp / (fp + tn + 1e-12)

def top_hit_rate(y_true, score, top_percent=0.01):
    n = len(score)
    k = max(1, int(math.ceil(n * top_percent)))
    idx = np.argsort(-score)[:k]
    return float(np.mean(y_true[idx] == 1)), k

def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    # 让 train/test 列完全一致（缺的补0，多的删）
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    for c in X_test.columns:
        if c not in X_train.columns:
            X_train[c] = 0
    X_test = X_test[X_train.columns]
    return X_train, X_test

def get_score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:,1]
    # decision_function -> minmax 到 [0,1]
    s = model.decision_function(X_test)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return s

def eval_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n[RUN] training {name} ...")
    model.fit(X_train, y_train)
    score = get_score(model, X_test)

    auc = roc_auc_score(y_test, score) if len(np.unique(y_test)) > 1 else float("nan")
    y_pred = (score >= THRESHOLD).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fpr_v = fpr(y_test, y_pred)
    hit, k = top_hit_rate(y_test, score, TOP_PERCENT)

    print(f"[OK] {name}: F1={f1:.6f}, AUC={auc:.6f}, FPR={fpr_v:.6f}, Top-1% hit={hit:.6f} (K={k})")
    return {
        "方法": name,
        "训练数据": "UNSW Train",
        "测试数据": "UNSW Test",
        "F1": float(f1),
        "ROC-AUC": float(auc),
        "误报率(FPR)": float(fpr_v),
        f"Top-{int(TOP_PERCENT*100)}%命中率": float(hit),
        "TopK": int(k),
    }

def main():
    print("[INFO] loading CSV ...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print("[INFO] train shape:", train_df.shape, " test shape:", test_df.shape)

    if LABEL_COL not in train_df.columns:
        raise ValueError(f"找不到标签列 {LABEL_COL}，请检查列名。")

    y_train = train_df[LABEL_COL].astype(int).to_numpy()
    y_test  = test_df[LABEL_COL].astype(int).to_numpy()

    X_train = train_df.drop(columns=[LABEL_COL] + DROP_COLS, errors="ignore")
    X_test  = test_df.drop(columns=[LABEL_COL] + DROP_COLS, errors="ignore")

    # 只保留数值列，避免字符串列导致崩/报错
    X_train = X_train.select_dtypes(include=[np.number])
    X_test  = X_test.select_dtypes(include=[np.number])

    # NaN/Inf 处理（不然某些模型会直接报错或不稳定）
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test = align_columns(X_train, X_test)

    results = []

    # 1) RF：单线程，避免 -1
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    results.append(eval_model("RandomForest (RF)", rf, X_train, y_train, X_test, y_test))

    # 2) SVM(RBF)：概率关闭，改用 decision_function（更稳定，照样能算AUC/Top-N）
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=False, class_weight="balanced", random_state=42)),
    ])
    results.append(eval_model("SVM (RBF, decision_function)", svm, X_train, y_train, X_test, y_test))

    # 3) LR 或 kNN
    if THIRD_BASELINE == "lr":
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, n_jobs=1, class_weight="balanced")),
        ])
        results.append(eval_model("LogisticRegression (LR)", lr, X_train, y_train, X_test, y_test))
    else:
        knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=15)),
        ])
        results.append(eval_model("kNN (k=15)", knn, X_train, y_train, X_test, y_test))

    out = pd.DataFrame(results)
    cols = ["方法","训练数据","测试数据","F1","ROC-AUC","误报率(FPR)",f"Top-{int(TOP_PERCENT*100)}%命中率"]
    print("\n===== 表5-6可直接抄写的结果 =====")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    cols = ["方法", "训练数据", "测试数据", "F1", "ROC-AUC", "误报率(FPR)", f"Top-{int(TOP_PERCENT * 100)}%命中率"]
    print(out[cols].to_string(index=False))

    out.to_csv("table_5_6_baselines.csv", index=False)
    print("\n已保存：table_5_6_baselines.csv")

if __name__ == "__main__":
    main()
