"""PCAP 特征提取模块。"""

from __future__ import annotations

import glob
import os
import socket
from typing import Callable, Iterable, List, Optional, Tuple

import dpkt
import pandas as pd

Packet = Tuple[float, bytes]


def _ip_to_str(raw: bytes) -> str:
    try:
        return socket.inet_ntoa(raw)
    except Exception:
        try:
            return socket.inet_ntop(socket.AF_INET6, raw)
        except Exception:
            return ""


def _open_reader(file_obj):
    try:
        return dpkt.pcap.Reader(file_obj)
    except (ValueError, dpkt.NeedData):
        file_obj.seek(0)
        return dpkt.pcapng.Reader(file_obj)


def _iterate_packets(reader) -> Iterable[Packet]:
    for packet in reader:
        if isinstance(packet, tuple) and len(packet) == 2:
            yield float(packet[0]), packet[1]
        else:
            timestamp = getattr(packet, "timestamp", None)
            data = getattr(packet, "packet_data", None)
            if timestamp is None or data is None:
                continue
            yield float(timestamp), data


def _count_packets(pcap_path: str) -> int:
    total = 0
    with open(pcap_path, "rb") as handle:
        reader = _open_reader(handle)
        for _ in _iterate_packets(reader):
            total += 1
    return total


def extract_features(
    pcap_path: str,
    output_csv: str,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> str:
    """从单个 PCAP 文件提取按包的轻量特征，并保存为 CSV。"""
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"文件不存在: {pcap_path}")

    total_packets = _count_packets(pcap_path)
    flows = []

    with open(pcap_path, "rb") as handle:
        reader = _open_reader(handle)
        for index, (ts, buf) in enumerate(_iterate_packets(reader), start=1):
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data

                if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                if isinstance(ip, dpkt.ip.IP):
                    proto = ip.p
                    src_ip = _ip_to_str(ip.src)
                    dst_ip = _ip_to_str(ip.dst)
                else:
                    proto = ip.nxt
                    src_ip = _ip_to_str(ip.src)
                    dst_ip = _ip_to_str(ip.dst)

                flows.append(
                    {
                        "timestamp": ts,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": proto,
                        "length": len(buf),
                        "pcap_file": os.path.basename(pcap_path),
                    }
                )
            except Exception:
                continue

            if progress_cb and total_packets > 0:
                if index % 200 == 0 or index == total_packets:
                    progress_cb(int(index * 100 / total_packets))

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
    """批量提取目录下 PCAP/PCAPNG 文件的轻量特征。"""
    if not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"目录不存在: {pcap_dir}")

    patterns = [os.path.join(pcap_dir, "*.pcap"), os.path.join(pcap_dir, "*.pcapng")]
    pcap_files = sorted({path for pattern in patterns for path in glob.glob(pattern)})

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