"""Result analysis helpers for the legacy GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from .csv_utils import read_csv_flexible

ProgressCallback = Optional[Callable[[int], None]]


def analyze_results(
    csv_path: str,
    out_dir: str,
    *,
    metadata_path: Optional[str] = None,
    progress_cb: ProgressCallback = None,
) -> Dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = read_csv_flexible(csv_path)

    metrics: Dict[str, object] = {"total": int(len(df))}
    if "prediction" in df.columns:
        preds = df["prediction"].astype(str).str.lower()
        anomaly_mask = preds.isin({"-1", "anomaly", "malicious", "attack"}) | preds.str.contains(
            "attack|malicious|anomaly|恶意|异常", case=False, na=False
        )
        metrics["anomalies"] = int(anomaly_mask.sum())
    if "malicious_score" in df.columns:
        metrics["avg_score"] = float(pd.to_numeric(df["malicious_score"], errors="coerce").mean(skipna=True) or 0.0)

    summary_text = f"总记录 {metrics.get('total', 0)} 条，疑似异常 {metrics.get('anomalies', 0)} 条。"

    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps({"metrics": metrics}, ensure_ascii=False, indent=2))

    summary_csv = out / "summary.csv"
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)

    if progress_cb:
        progress_cb(100)

    return {
        "out_dir": str(out),
        "summary_text": summary_text,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "metrics": metrics,
        "plots": [],
    }
