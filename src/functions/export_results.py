# src/functions/export_results.py
# -*- coding: utf-8 -*-
"""
导出结果（独立模块）
- 导出现有 DataFrame 的预览为 CSV / Excel
- 打包 UI “输出文件列表”中的文件/目录为 ZIP
- 复制“查看流量信息”生成的全量 CSV 到指定位置
"""

import os
import shutil
import zipfile
from datetime import datetime

try:
    import pandas as pd
except Exception:
    pd = None


def export_preview_to_csv(df, dst_csv: str) -> str:
    """
    将预览 DataFrame 导出为 CSV。
    :return: 导出后的绝对路径
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError("没有可导出的表格数据（预览为空）")
    os.makedirs(os.path.dirname(dst_csv) or ".", exist_ok=True)
    df.to_csv(dst_csv, index=False, encoding="utf-8")
    return os.path.abspath(dst_csv)


def export_preview_to_xlsx(df, dst_xlsx: str) -> str:
    """
    将预览 DataFrame 导出为 Excel（.xlsx）
    """
    if pd is None:
        raise RuntimeError("需要 pandas 才能导出 Excel")
    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError("没有可导出的表格数据（预览为空）")
    os.makedirs(os.path.dirname(dst_xlsx) or ".", exist_ok=True)
    # engine 由 pandas 自动选择（openpyxl/xlsxwriter）
    df.to_excel(dst_xlsx, index=False)
    return os.path.abspath(dst_xlsx)


def zip_outputs(paths, dst_zip: str, arc_base: str = "") -> str:
    """
    将 paths 中的文件或文件夹打包为一个 zip。
    - 不存在的条目会跳过
    - arc_base：Zip 内部的顶层目录名（可留空）
    """
    if not paths:
        raise ValueError("没有可打包的条目")
    os.makedirs(os.path.dirname(dst_zip) or ".", exist_ok=True)
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if not p or not os.path.exists(p):
                continue
            p = os.path.abspath(p)
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        ap = os.path.join(root, fn)
                        rel = os.path.relpath(ap, os.path.dirname(p))  # 目录上层作为相对根
                        arcname = os.path.join(arc_base, rel).replace("\\", "/")
                        zf.write(ap, arcname)
            else:
                name = os.path.basename(p)
                arcname = os.path.join(arc_base, name).replace("\\", "/")
                zf.write(p, arcname)
    return os.path.abspath(dst_zip)


def copy_full_csv(src_csv: str, dst_csv: str) -> str:
    """
    复制“查看流量信息”的全量 CSV（pcap_info_all.csv）到目标位置。
    """
    if not src_csv or not os.path.exists(src_csv):
        raise FileNotFoundError("找不到全量 CSV 文件")
    os.makedirs(os.path.dirname(dst_csv) or ".", exist_ok=True)
    shutil.copy2(src_csv, dst_csv)
    return os.path.abspath(dst_csv)


def default_export_name(prefix: str, ext: str) -> str:
    """
    生成一个带时间戳的默认导出文件名，如 export_preview_2025-09-10_143011.csv
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"
