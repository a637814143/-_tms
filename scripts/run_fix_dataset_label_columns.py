"""CLI helper to realign mislabelled CSV datasets.

This wraps :func:`src.functions.csv_utils.fix_dataset_label_columns` and adds
batch support so misaligned datasets in a folder can be corrected in one
command.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

from src.functions.csv_utils import fix_dataset_label_columns

DEFAULT_DATASET = Path(__file__).parent.parent / "data/CSV/csv_preprocess/dataset_20251125_203952.csv"


def _discover_csvs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*.csv") if p.is_file()])


def _copy_to_output(fixed_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / fixed_path.name
    if fixed_path.resolve() == target.resolve():
        return fixed_path
    shutil.copy2(fixed_path, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="修复 dataset_20251125_203952.csv 这类 86 列表头、88 列数据的错位标签。",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_DATASET),
        help=(
            "待修复的 CSV 路径或包含多个 CSV 的目录，"
            "默认指向 data/CSV/csv_preprocess/dataset_20251125_203952.csv。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="修复后的文件输出目录；若不指定则写回输入目录。",
    )
    args = parser.parse_args()

    source = Path(args.input_path)
    if not source.exists():
        raise SystemExit(f"输入路径不存在: {source}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    csv_files = _discover_csvs(source)
    if not csv_files:
        raise SystemExit("未找到任何 CSV 文件可供修复。")

    written: List[Path] = []
    for csv_path in csv_files:
        fixed = fix_dataset_label_columns(csv_path)
        if output_dir is not None:
            fixed = _copy_to_output(fixed, output_dir)
        written.append(fixed)
        print(f"✅ 修复完成: {csv_path.name} -> {fixed}")

    print(f"共修复 {len(written)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
