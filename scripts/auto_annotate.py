"""Lightweight auto-annotation utility for the GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.functions.csv_utils import read_csv_flexible


def _score_series(df: pd.DataFrame) -> Optional[pd.Series]:
    for column in ("malicious_score", "anomaly_score", "score"):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    if "prediction" in df.columns:
        series = df["prediction"].astype(str).str.lower()
        return series.isin({"-1", "anomaly", "malicious", "attack"}).astype(float)
    return None


def auto_annotate(
    csv_path: str,
    *,
    mode: str = "conservative",
    write_benign: bool = True,
    top_k: int = 300,
) -> Dict[str, int]:
    df = read_csv_flexible(csv_path)
    score = _score_series(df)
    if score is None:
        raise ValueError("无法从CSV中推断异常分数或预测列")

    df = df.copy()
    df["__score__"] = score

    top = df.nlargest(top_k, "__score__") if top_k > 0 else df[df["__score__"] > 0]
    positive_indices = set(top.index)

    labels = []
    for idx in df.index:
        if idx in positive_indices:
            labels.append(1.0)
        elif write_benign:
            labels.append(0.0)
        else:
            labels.append(None)

    df["label"] = labels

    label_path = Path(csv_path).with_name("labels.csv")
    output_columns = [col for col in ("flow_id", "Flow ID", "__source_file__", "pcap_file") if col in df.columns]
    if not output_columns:
        df.insert(0, "index", range(len(df)))
        output_columns = ["index"]
    export_cols = output_columns + ["label", "__score__"]
    df[export_cols].to_csv(label_path, index=False)

    total = int(df["label"].notna().sum())
    anomalies = int((df["label"] == 1.0).sum())
    normals = int((df["label"] == 0.0).sum())

    return {
        "total": total,
        "anomalies": anomalies,
        "normals": normals,
        "added_anomalies": anomalies,
        "added_normals": normals,
    }
