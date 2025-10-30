"""Simple JSON-backed annotation helpers for the GUI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.configuration import project_root

ANNOTATION_FILE = project_root() / "artifacts" / "annotations.json"
ANNOTATION_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_annotations() -> Dict[str, Dict[str, object]]:
    if ANNOTATION_FILE.exists():
        try:
            return json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_annotations(data: Dict[str, Dict[str, object]]) -> None:
    ANNOTATION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_key(record: Dict[str, object]) -> Optional[str]:
    for key in ("flow_id", "Flow ID", "__source_file__", "pcap_file"):
        value = record.get(key)
        if value:
            return str(value)
    if record.get("Source IP") and record.get("Destination IP"):
        src = record.get("Source IP")
        dst = record.get("Destination IP")
        return f"{src}->{dst}:{record.get('Source Port')}->{record.get('Destination Port')}"
    return None


def upsert_annotation(record: Dict[str, object], *, label: float, notes: Optional[str] = None) -> None:
    key = _resolve_key(record)
    if not key:
        raise ValueError("无法确定标注键")
    data = _load_annotations()
    data[key] = {
        "label": float(label),
        "notes": notes,
        "timestamp": time.time(),
    }
    _save_annotations(data)


def annotation_summary() -> Dict[str, object]:
    data = _load_annotations()
    total = len(data)
    anomalies = sum(1 for value in data.values() if float(value.get("label", 0.0)) > 0.5)
    normals = total - anomalies
    return {"total": total, "anomalies": anomalies, "normals": normals}


def apply_annotations_to_frame(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    data = _load_annotations()
    if not data:
        return pd.Series([None] * len(df), index=df.index)

    def _row_key(row):
        for key in ("flow_id", "Flow ID", "__source_file__", "pcap_file"):
            value = row.get(key)
            if value:
                return str(value)
        if row.get("Source IP") and row.get("Destination IP"):
            return f"{row.get('Source IP')}->{row.get('Destination IP')}:{row.get('Source Port')}->{row.get('Destination Port')}"
        return None

    labels = []
    for _, row in df.iterrows():
        key = _row_key(row)
        entry = data.get(key) if key else None
        labels.append(entry.get("label") if entry else None)
    return pd.Series(labels, index=df.index)
