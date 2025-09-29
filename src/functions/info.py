# -*- coding: utf-8 -*-
"""从 PCAP 文件提取简化的流量特征。"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader

FlowKey = Tuple[str, str, int, int, str]


@dataclass
class FlowStats:
    """记录单条网络流的基础统计信息。"""

    forward_lengths: list[int] = field(default_factory=list)
    backward_lengths: list[int] = field(default_factory=list)
    forward_times: list[float] = field(default_factory=list)
    backward_times: list[float] = field(default_factory=list)
    forward_flags: Counter[str] = field(default_factory=Counter)
    backward_flags: Counter[str] = field(default_factory=Counter)

    def add_packet(
        self,
        *,
        length: int,
        timestamp: float,
        direction: str,
        flags: Optional[str] = None,
    ) -> None:
        if direction == "fwd":
            self.forward_lengths.append(length)
            self.forward_times.append(timestamp)
            if flags:
                self.forward_flags[flags] += 1
        else:
            self.backward_lengths.append(length)
            self.backward_times.append(timestamp)
            if flags:
                self.backward_flags[flags] += 1

    def all_times(self) -> list[float]:
        return self.forward_times + self.backward_times


def _iter_packets(path: str) -> Iterable:
    with PcapReader(path) as reader:
        for packet in reader:
            yield packet


def _flow_key(pkt) -> Optional[Tuple[FlowKey, FlowKey, str]]:
    if IP not in pkt:
        return None

    layer = None
    proto = ""
    src_port = 0
    dst_port = 0

    if TCP in pkt:
        layer = pkt[TCP]
        proto = "TCP"
    elif UDP in pkt:
        layer = pkt[UDP]
        proto = "UDP"

    if layer is None:
        return None

    src_port = int(layer.sport)
    dst_port = int(layer.dport)
    forward = (pkt[IP].src, pkt[IP].dst, src_port, dst_port, proto)
    backward = (pkt[IP].dst, pkt[IP].src, dst_port, src_port, proto)
    flags = str(layer.flags) if hasattr(layer, "flags") else ""
    return forward, backward, flags


def get_pcap_features(
    path: str,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """提取 PCAP 文件的流量特征，确保兼容 GUI 回调。"""

    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    flows: Dict[FlowKey, FlowStats] = defaultdict(FlowStats)
    total_packets = sum(1 for _ in _iter_packets(path))
    start_time = time.time()

    processed = 0

    for pkt in _iter_packets(path):
        if cancel_cb and cancel_cb():
            break

        keys = _flow_key(pkt)
        if keys is None:
            continue

        key_fwd, key_bwd, flags = keys
        timestamp = float(pkt.time)
        length = len(pkt)

        if key_fwd in flows:
            flows[key_fwd].add_packet(
                length=length, timestamp=timestamp, direction="fwd", flags=flags
            )
        elif key_bwd in flows:
            flows[key_bwd].add_packet(
                length=length, timestamp=timestamp, direction="bwd", flags=flags
            )
        else:
            flows[key_fwd].add_packet(
                length=length, timestamp=timestamp, direction="fwd", flags=flags
            )

        processed += 1
        if progress_cb and total_packets:
            if processed % 500 == 0 or processed == total_packets:
                progress_cb(min(99, int(processed * 100 / total_packets)))

    records = []
    for (src_ip, dst_ip, src_port, dst_port, proto), stats in flows.items():
        times = stats.all_times()
        if not times:
            continue

        duration = max(times) - min(times)

        def safe_mean(values: list[int]) -> float:
            return statistics.mean(values) if values else 0.0

        def safe_std(values: list[int]) -> float:
            return statistics.pstdev(values) if len(values) > 1 else 0.0

        def safe_max(values: list[int]) -> int:
            return max(values) if values else 0

        def safe_min(values: list[int]) -> int:
            return min(values) if values else 0

        total_fwd = sum(stats.forward_lengths)
        total_bwd = sum(stats.backward_lengths)
        total_packets_flow = len(stats.forward_lengths) + len(stats.backward_lengths)

        records.append(
            {
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "flow_duration": duration,
                "total_fwd_pkts": len(stats.forward_lengths),
                "total_bwd_pkts": len(stats.backward_lengths),
                "total_len_fwd_pkts": total_fwd,
                "total_len_bwd_pkts": total_bwd,
                "fwd_pkt_len_max": safe_max(stats.forward_lengths),
                "fwd_pkt_len_min": safe_min(stats.forward_lengths),
                "fwd_pkt_len_mean": safe_mean(stats.forward_lengths),
                "fwd_pkt_len_std": safe_std(stats.forward_lengths),
                "bwd_pkt_len_max": safe_max(stats.backward_lengths),
                "bwd_pkt_len_min": safe_min(stats.backward_lengths),
                "bwd_pkt_len_mean": safe_mean(stats.backward_lengths),
                "bwd_pkt_len_std": safe_std(stats.backward_lengths),
                "flow_byts_per_s": (total_fwd + total_bwd) / duration if duration > 0 else 0.0,
                "flow_pkts_per_s": total_packets_flow / duration if duration > 0 else 0.0,
            }
        )

    df = pd.DataFrame.from_records(records)

    if progress_cb:
        progress_cb(100)

    elapsed = time.time() - start_time
    print(f"[INFO] 提取完成: {len(df)} 条流, 耗时 {elapsed:.2f}s")

    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract simplified flow features from a PCAP/PCAPNG file.",
    )
    parser.add_argument("pcap", help="Path to the input PCAP/PCAPNG file")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional CSV output path (defaults to <pcap>.csv)",
        default=None,
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Preview the first N rows when no output file is specified",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.pcap):
        parser.error(f"文件不存在: {args.pcap}")

    df = get_pcap_features(args.pcap)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False, encoding="utf-8")
        print(f"[+] 已写入 CSV: {args.output}")
    else:
        preview_rows = df.head(args.head)
        if not preview_rows.empty:
            print(preview_rows.to_string(index=False))
        print(f"[INFO] 预览 {min(args.head, len(df))} / {len(df)} 条记录")


if __name__ == "__main__":
    main()
