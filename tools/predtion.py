import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

path = "D:\\pythonProject8\\data\\results\\modelprediction\\prediction_balanced_20000_Friday-WorkingHours-Afternoon-DDos_pcap_ISCX_20251218_150844_small_20251229_192935.csv"
df = pd.read_csv(path)

y_true = (df["Label"].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
y_score = pd.to_numeric(df["fusion_score"], errors="coerce").fillna(0)

roc = roc_auc_score(y_true, y_score) if y_true.nunique()==2 else None
ap  = average_precision_score(y_true, y_score) if y_true.nunique()==2 else None

n = max(1, int(len(df) * 0.01))
top = df.nlargest(n, "fusion_score")
top_hit = (top["Label"].astype(str).str.strip().str.upper() != "BENIGN").mean()

print("ROC-AUC:", roc, "PR-AUC:", ap, "Top1%命中率:", top_hit)
