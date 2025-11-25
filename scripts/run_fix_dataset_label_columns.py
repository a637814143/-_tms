"""CLI helper to realign mislabelled CSV datasets.

This wraps :func:`src.functions.csv_utils.fix_dataset_label_columns` with the
expected default path so the corrected ``*_fixed.csv`` file can be generated
with a single invocation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.functions.csv_utils import fix_dataset_label_columns

DEFAULT_DATASET = Path(__file__).parent.parent / "data/CSV/csv_preprocess/dataset_20251125_203952.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复 dataset_20251125_203952.csv 这类 86 列表头、88 列数据的错位标签。",
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help=(
            "待修复的 CSV 路径，默认指向 data/CSV/csv_preprocess/"
            "dataset_20251125_203952.csv。"
        ),
    )
    args = parser.parse_args()
    csv_path = Path(args.csv_path)

    fixed_path = fix_dataset_label_columns(csv_path)
    print(f"修复后的数据集：{fixed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())