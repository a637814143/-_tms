"""Run a single supervised baseline training and export metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import pandas as pd

from src.configuration import load_config
from src.functions.modeling import train_supervised_on_split


def _load_config(path: Path) -> Dict[str, object]:
    cfg = load_config(str(path))
    return cfg if isinstance(cfg, dict) else {}


def _collect_metrics(metadata_path: Path) -> Dict[str, float]:
    if not metadata_path.is_file():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    metrics = data.get("model_metrics", {}) if isinstance(data, dict) else {}
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def main() -> int:
    parser = argparse.ArgumentParser(description="训练基线有监督模型并导出指标。")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config/default.yaml"),
        help="配置文件路径，默认使用 config/default.yaml",
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="带标签的训练 CSV 或目录路径。",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="训练结果与模型的输出目录。",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    train_path = Path(args.train_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(config_path)

    results = train_supervised_on_split(
        split_dir=train_path,
        results_dir=output_dir,
        models_dir=output_dir,
        model_tag="baseline",
        **(config.get("training") or {}),
    )

    metadata_path = Path(results["model_metadata_path"])
    metrics = _collect_metrics(metadata_path)

    metrics_path = output_dir / "baseline_metrics.csv"
    if metrics:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    else:
        metrics_path.write_text("", encoding="utf-8")

    print(f"训练完成，模型保存在 {results['model_path']}，指标写入 {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
