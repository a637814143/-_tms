#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目统一入口（建议放在仓库根目录：run.py）

默认启动 GUI：
    python run.py

如果你更倾向于命令行（pipeline_service 已内置 build_parser/run_cli），可以：
    python run.py --cli extract <pcap_dir> <output_dir>
    python run.py --cli train   <split_dir> <results_dir> <models_dir>
    python run.py --cli predict <pipeline> <features.csv> [--metadata xxx.json] [--output out.csv]
    python run.py --cli analyze <results.csv> <output_dir> [--metadata xxx.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import runpy


def _ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parent
    # 让 `import src....` 可用
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run.py", add_help=True)
    parser.add_argument(
        "--cli",
        action="store_true",
        help="使用命令行模式（调用 src.services.pipeline_service.run_cli）",
    )
    args, rest = parser.parse_known_args(argv)

    root = _ensure_repo_root_on_path()

    if args.cli:
        # 直接调用 pipeline_service 的 CLI（更稳定，参数解析交给它）
        try:
            from src.services.pipeline_service import run_cli
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "无法导入 src.services.pipeline_service.run_cli。"
                "请确认你在仓库根目录运行，并已安装 requirements.txt 依赖。"
            )(exc)
        return int(run_cli(rest))

    # GUI：等价于 `python -m src.ui.ui_main` 或 `python src/ui/ui_main.py`
    ui_entry = root / "src" / "ui" / "ui_main.py"
    if not ui_entry.exists():
        raise FileNotFoundError(f"未找到 GUI 入口文件：{ui_entry}")

    # runpy 可以模拟脚本运行，从而触发 ui_main.py 里的 __main__ 逻辑
    runpy.run_path(str(ui_entry), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())