"""Utilities for managing manual annotations / labels for flows."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from src.configuration import get_path, load_config, project_root
from src.functions.logging_utils import get_logger

LOGGER = get_logger(__name__)

ANNOTATION_DIR_NAME = "annotations"
ANNOTATION_FILE_NAME = "labels.csv"

KEY_COLUMNS = (
    "flow_id",
    "pcap_file",
    "__source_file__",
    "__source_path__",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
)


def _data_root() -> Path:
    try:
        base = get_path("data_dir")
    except Exception:
        base = Path.home() / "maldet_data"
        base.mkdir(parents=True, exist_ok=True)
    annotations_dir = base / ANNOTATION_DIR_NAME
    annotations_dir.mkdir(parents=True, exist_ok=True)
    return annotations_dir


def annotation_store_path() -> Path:
    """Return the CSV path that stores manual annotations."""

    return _data_root() / ANNOTATION_FILE_NAME


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "flow_key",
        "label",
        "annotator",
        "notes",
        "timestamp",
    ]
    for column in expected:
        if column not in df.columns:
            df[column] = "" if column != "label" else pd.Series(dtype=float)
    return df


def load_annotations() -> pd.DataFrame:
    path = annotation_store_path()
    if not path.exists():
        return pd.DataFrame(columns=["flow_key", "label", "annotator", "notes", "timestamp"])
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if "label" in df.columns:
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = _ensure_columns(df)
        df = df.dropna(subset=["flow_key"])
        df = df.drop_duplicates(subset=["flow_key"], keep="last")
        return df
    except Exception as exc:
        LOGGER.warning("Failed to read annotations from %s: %s", path, exc)
        return pd.DataFrame(columns=["flow_key", "label", "annotator", "notes", "timestamp"])


def save_annotations(df: pd.DataFrame) -> None:
    path = annotation_store_path()
    df = _ensure_columns(df.copy())
    df.to_csv(path, index=False, encoding="utf-8")


def build_flow_key(record: Mapping[str, object]) -> str:
    parts: list[str] = []
    for column in KEY_COLUMNS:
        value = record.get(column) if isinstance(record, Mapping) else None
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    if not parts and isinstance(record, Mapping):
        try:
            payload = json.dumps(dict(record), sort_keys=True, ensure_ascii=False)
            parts.append(payload)
        except Exception:
            pass
    digest_input = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(digest_input).hexdigest()


def upsert_annotation(
    record: Mapping[str, object],
    *,
    label: float,
    annotator: Optional[str] = None,
    notes: str | None = None,
) -> None:
    key = build_flow_key(record)
    df = load_annotations()
    payload = {
        "flow_key": key,
        "label": float(label),
        "annotator": annotator or "ui",
        "notes": notes or "",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    mask = df["flow_key"] == key if not df.empty else pd.Series(dtype=bool)
    if not df.empty and mask.any():
        df.loc[mask, payload.keys()] = list(payload.values())
    else:
        df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    save_annotations(df)
    LOGGER.info("Stored manual label for flow=%s value=%s", key, label)


def apply_annotations_to_frame(df: pd.DataFrame) -> Optional[pd.Series]:
    annotations = load_annotations()
    if df.empty or annotations.empty:
        return None
    keys = df.apply(lambda row: build_flow_key(row), axis=1)
    keys.name = "flow_key"
    merged = pd.concat([df.reset_index(drop=True), keys], axis=1)
    merged = merged.merge(
        annotations[["flow_key", "label"]],
        on="flow_key",
        how="left",
    )
    if "label" not in merged.columns:
        return None
    labels = merged["label"].to_numpy(dtype=float)
    if not pd.isfinite(labels).any():
        return None
    return pd.Series(labels, index=df.index, dtype=float)


def annotation_summary() -> dict:
    df = load_annotations()
    if df.empty:
        return {"total": 0, "anomalies": 0, "normals": 0}
    labels = pd.to_numeric(df["label"], errors="coerce").fillna(-1)
    anomalies = int((labels > 0.5).sum())
    normals = int((labels == 0).sum())
    return {
        "total": int(len(df)),
        "anomalies": anomalies,
        "normals": normals,
    }


def configured_plugin_dirs() -> Iterable[Path]:
    config = load_config()
    plugins_cfg = config.get("plugins") if isinstance(config, dict) else None
    if not isinstance(plugins_cfg, Mapping):
        return []
    dirs = plugins_cfg.get("feature_dirs") if isinstance(plugins_cfg, Mapping) else None
    if not isinstance(dirs, (list, tuple)):
        return []
    for item in dirs:
        if not item:
            continue
        path = Path(item)
        if not path.is_absolute():
            base = project_root()
            path = base.joinpath(item)
        yield path.expanduser().resolve()


__all__ = [
    "annotation_store_path",
    "load_annotations",
    "save_annotations",
    "build_flow_key",
    "upsert_annotation",
    "apply_annotations_to_frame",
    "annotation_summary",
    "configured_plugin_dirs",
]