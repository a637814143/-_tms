"""Training wrappers mapping the legacy GUI to the new PCAP pipeline."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from joblib import load

from PCAP import TrainingSummary, train_hist_gradient_boosting

ProgressCallback = Optional[Callable[[int], None]]

META_COLUMNS = {
    "pcap_file",
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "__source_file__",
    "__source_path__",
}


def _resolve_dataset_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        csv_files = sorted(candidate.glob("*.csv"))
        if csv_files:
            return csv_files[-1]
    raise FileNotFoundError(f"Unable to locate dataset under {path}")


def _write_metadata(models_dir: Path, summary: TrainingSummary, dataset_path: Path, model_artifact: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    latest_meta = models_dir / "latest_iforest_metadata.json"
    meta_path = models_dir / f"iforest_metadata_{timestamp}.json"

    metadata = {
        "timestamp": timestamp,
        "classes": summary.classes,
        "feature_names": summary.feature_names,
        "flow_count": summary.flow_count,
        "label_mapping": summary.label_mapping,
        "dropped_flows": summary.dropped_flows,
        "dataset": str(dataset_path),
        "pipeline_latest": str(model_artifact),
        "metadata_latest": str(latest_meta),
    }

    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return latest_meta


def train_unsupervised_on_split(
    dataset_source: Union[str, Path],
    results_dir: Union[str, Path],
    models_dir: Union[str, Path],
    *,
    progress_cb: ProgressCallback = None,
    iterations: Optional[int] = None,
    **_: object,
) -> Dict[str, object]:
    """Train the gradient boosting model expected by the GUI."""

    dataset_path = _resolve_dataset_path(dataset_source)
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.txt"

    params = {}
    if iterations is not None:
        params["max_iter"] = int(iterations)
    if progress_cb:
        progress_cb(5)

    summary: TrainingSummary = train_hist_gradient_boosting(
        dataset_path,
        model_path,
        **params,
    )

    # Mirror artifacts using legacy naming so the GUI keeps working.
    latest_pipeline = models_dir / "latest_iforest_pipeline.joblib"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stamped_pipeline = models_dir / f"iforest_pipeline_{timestamp}.joblib"
    shutil.copy2(model_path, latest_pipeline)
    shutil.copy2(model_path, stamped_pipeline)

    metadata_path = _write_metadata(models_dir, summary, dataset_path, latest_pipeline)

    feature_importance = []
    try:
        artifact = load(model_path)
        model = artifact.get("model")
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            feature_importance = list(zip(summary.feature_names, importances))
    except Exception:
        feature_importance = []

    messages = [
        f"训练完成：{summary.flow_count} 条流，{len(summary.feature_names)} 个特征。",
    ]
    if summary.dropped_flows:
        messages.append(f"已丢弃 {summary.dropped_flows} 条未标注流。")

    if progress_cb:
        progress_cb(100)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metadata_latest": str(metadata_path),
        "pipeline_latest": str(latest_pipeline),
        "results_csv": str(dataset_path),
        "summary_csv": str(dataset_path),
        "messages": messages,
        "feature_importance": feature_importance,
        "threshold": None,
        "vote_threshold": None,
    }
