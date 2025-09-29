# src/functions/feature_extractor.py
import os
import dpkt
import glob
import socket
from typing import Callable, List, Optional

import dpkt
import pandas as pd

def _ip_to_str(raw):
    try:
        return socket.inet_ntoa(raw)
    except Exception:
        # 不是 IPv4 时兜底
        try:
            return socket.inet_ntop(socket.AF_INET6, raw)
        except Exception:
            return ""

def extract_features(pcap_path: str, output_csv: str, progress_cb=None) -> str:
def extract_features(
    pcap_path: str,
    output_csv: str,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """
    从单个 pcap 文件提取“按包”的轻量特征，并保存为 CSV。
    字段：timestamp, src_ip, dst_ip, protocol, length, pcap_file
    说明：轻量 & 快速，便于无监督训练（IsolationForest）。
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"文件不存在: {pcap_path}")

    flows = []
    total = 0

    # 先统计包数，用于进度条
    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.NeedData):
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        for _ in reader:
            total += 1

    with open(pcap_path, "rb") as f:
        # 再次读取
        try:
            reader = dpkt.pcap.Reader(f)
@@ -63,25 +70,81 @@ def extract_features(pcap_path: str, output_csv: str, progress_cb=None) -> str:
                    "timestamp": ts,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto,
                    "length": length,
                    "pcap_file": os.path.basename(pcap_path)
                })
            except Exception:
                # 坏包/非IP 包等，直接跳过
                pass

            if progress_cb and total > 0 and (i % 200 == 0 or i == total):
                progress_cb(int(i * 100 / total))

    if not flows:
        raise ValueError(f"{pcap_path} 未提取到任何有效数据")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df = pd.DataFrame(flows)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    if progress_cb:
        progress_cb(100)
    return output_csv


def extract_features_dir(
    pcap_dir: str,
    output_dir: str,
    *,
    workers: int = 4,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> List[str]:
    """批量提取目录下所有 PCAP/PCAPNG 文件的轻量特征。

    为保持与 UI 的兼容性，``workers`` 参数目前用于接口对齐，实际处理按顺序执行，
    以便可以正确汇聚每个文件的进度到整体进度回调。
    """

    if not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"目录不存在: {pcap_dir}")

    patterns = [os.path.join(pcap_dir, "*.pcap"), os.path.join(pcap_dir, "*.pcapng")]
    pcap_files = sorted({p for pattern in patterns for p in glob.glob(pattern)})

    if not pcap_files:
        raise RuntimeError(f"目录中没有找到 pcap/pcapng 文件: {pcap_dir}")

    os.makedirs(output_dir, exist_ok=True)

    total = len(pcap_files)
    csv_paths: List[str] = []

    if progress_cb:
        progress_cb(0)

    for idx, pcap_path in enumerate(pcap_files, start=1):
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_features.csv")

        if progress_cb:
            base_progress = (idx - 1) / total

            def _per_file_cb(pct: int, base_progress=base_progress):
                overall = base_progress + max(0, min(100, pct)) / 100 / total
                progress_cb(min(99, int(overall * 100)))

            per_file_cb: Optional[Callable[[int], None]] = _per_file_cb
        else:
            per_file_cb = None

        csv_paths.append(extract_features(pcap_path, csv_path, progress_cb=per_file_cb))

        if progress_cb:
            progress_cb(int(idx * 100 / total))

    if progress_cb:
        progress_cb(100)

    return csv_paths