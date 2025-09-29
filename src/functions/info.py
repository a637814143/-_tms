"""PCAP 信息提取工具。"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections import defaultdict
from typing import Callable, Optional

import pandas as pd
from scapy.all import IP, TCP, UDP, PcapReader


def get_pcap_features(
    path: str,
    *,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> pd.DataFrame:
    """提取 PCAP 文件的流量特征（简化版 CICFlowMeter 特征集）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    flows: defaultdict = defaultdict(
        lambda: {
            "fwd_pkts": [],
            "bwd_pkts": [],
            "fwd_times": [],
            "bwd_times": [],
            "fwd_flags": defaultdict(int),
            "bwd_flags": defaultdict(int),
            "fwd_bytes": 0,
            "bwd_bytes": 0,
        }
    )

    start_time = time.time()

    total_packets = 0
    with PcapReader(path) as reader:
        for _ in reader:
            total_packets += 1
            if cancel_cb and cancel_cb():
                break

    processed = 0
    with PcapReader(path) as reader:
        for pkt in reader:
            if cancel_cb and cancel_cb():
                if progress_cb:
                    progress_cb(100)
                break

            if IP not in pkt:
                continue

            proto = None
            src_port = None
            dst_port = None

            if TCP in pkt:
                proto = "TCP"
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = pkt[TCP].flags
            elif UDP in pkt:
                proto = "UDP"
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport
                flags = None
            else:
                continue  # 忽略非 TCP/UDP

            key_fwd = (pkt[IP].src, pkt[IP].dst, src_port, dst_port, proto)
            key_bwd = (pkt[IP].dst, pkt[IP].src, dst_port, src_port, proto)

            ts = float(pkt.time)
            length = len(pkt)

            if key_fwd in flows:
                flows[key_fwd]["fwd_pkts"].append(length)
                flows[key_fwd]["fwd_times"].append(ts)
                flows[key_fwd]["fwd_bytes"] += length
                if flags:
                    flows[key_fwd]["fwd_flags"][str(flags)] += 1
            elif key_bwd in flows:
                flows[key_bwd]["bwd_pkts"].append(length)
                flows[key_bwd]["bwd_times"].append(ts)
                flows[key_bwd]["bwd_bytes"] += length
                if flags:
                    flows[key_bwd]["bwd_flags"][str(flags)] += 1
            else:
                flows[key_fwd]["fwd_pkts"].append(length)
                flows[key_fwd]["fwd_times"].append(ts)
                flows[key_fwd]["fwd_bytes"] += length
                if flags:
                    flows[key_fwd]["fwd_flags"][str(flags)] += 1

            processed += 1
            if progress_cb and total_packets:
                if processed % 500 == 0 or processed == total_packets:
                    progress_cb(int(processed * 100 / total_packets))

    records = []
    for (src_ip, dst_ip, sport, dport, proto), data in flows.items():
        all_times = data["fwd_times"] + data["bwd_times"]
        if not all_times:
            continue

        flow_dur = max(all_times) - min(all_times)

        def safe_mean(values):
            return statistics.mean(values) if values else 0

        def safe_std(values):
            return statistics.pstdev(values) if len(values) > 1 else 0

        def safe_max(values):
            return max(values) if values else 0

        def safe_min(values):
            return min(values) if values else 0

        record = {
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": sport,
            "dst_port": dport,
            "protocol": proto,
            "flow_duration": flow_dur,
            "total_fwd_pkts": len(data["fwd_pkts"]),
            "total_bwd_pkts": len(data["bwd_pkts"]),
            "total_len_fwd_pkts": sum(data["fwd_pkts"]),
            "total_len_bwd_pkts": sum(data["bwd_pkts"]),
            "fwd_pkt_len_max": safe_max(data["fwd_pkts"]),
            "fwd_pkt_len_min": safe_min(data["fwd_pkts"]),
            "fwd_pkt_len_mean": safe_mean(data["fwd_pkts"]),
            "fwd_pkt_len_std": safe_std(data["fwd_pkts"]),
            "bwd_pkt_len_max": safe_max(data["bwd_pkts"]),
            "bwd_pkt_len_min": safe_min(data["bwd_pkts"]),
            "bwd_pkt_len_mean": safe_mean(data["bwd_pkts"]),
            "bwd_pkt_len_std": safe_std(data["bwd_pkts"]),
            "flow_byts_per_s": (
                (sum(data["fwd_pkts"]) + sum(data["bwd_pkts"])) / flow_dur if flow_dur > 0 else 0
            ),
            "flow_pkts_per_s": (
                (len(data["fwd_pkts"]) + len(data["bwd_pkts"])) / flow_dur if flow_dur > 0 else 0
            ),
        }
        records.append(record)

    df = pd.DataFrame(records)

    duration = time.time() - start_time
    print(f"[INFO] 提取完成: {len(df)} 条流, 耗时 {duration:.2f}s")

    if progress_cb:
        progress_cb(100)

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