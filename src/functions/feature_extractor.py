"""PCAP 特征提取模块，输出高维流量特征。"""

import glob
import os
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import dpkt
import numpy as np
import pandas as pd

Packet = Tuple[float, bytes]
ProgressCallback = Optional[Callable[[int], None]]

TCP_FLAG_MAP = {
    "fin": dpkt.tcp.TH_FIN,
    "syn": dpkt.tcp.TH_SYN,
    "rst": dpkt.tcp.TH_RST,
    "psh": dpkt.tcp.TH_PUSH,
    "ack": dpkt.tcp.TH_ACK,
    "urg": dpkt.tcp.TH_URG,
}

LENGTH_BINS = np.linspace(0, 8192, 257)
INTERVAL_BINS = np.linspace(0.0, 5.0, 257)


def _ip_to_str(raw: bytes) -> str:
    try:
        return socket.inet_ntoa(raw)
    except Exception:
        try:
            return socket.inet_ntop(socket.AF_INET6, raw)
        except Exception:
            return ""


def _protocol_name(proto: int) -> str:
    mapping = {
        1: "ICMP",
        6: "TCP",
        17: "UDP",
        58: "ICMPv6",
    }
    return mapping.get(proto, str(proto))


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


def _hist_features(values: List[float], bins: np.ndarray, prefix: str) -> Dict[str, int]:
    if not values:
        return {f"{prefix}_{i:03d}": 0 for i in range(len(bins) - 1)}
    hist, _ = np.histogram(values, bins=bins)
    return {f"{prefix}_{i:03d}": int(hist[i]) for i in range(hist.size)}


def _inter_arrivals(times: List[float]) -> np.ndarray:
    if len(times) < 2:
        return np.empty(0, dtype=float)
    arr = np.diff(np.array(times, dtype=float))
    return arr[arr >= 0]


@dataclass
class FlowAccumulator:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    pcap_file: str
    forward_lengths: List[int] = field(default_factory=list)
    backward_lengths: List[int] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    backward_times: List[float] = field(default_factory=list)
    forward_flags: Counter = field(default_factory=Counter)
    backward_flags: Counter = field(default_factory=Counter)

    def add_packet(
        self,
        *,
        length: int,
        timestamp: float,
        direction: str,
        flags: Optional[int] = None,
    ) -> None:
        if direction == "fwd":
            self.forward_lengths.append(length)
            self.forward_times.append(timestamp)
            if flags is not None:
                for name, mask in TCP_FLAG_MAP.items():
                    if flags & mask:
                        self.forward_flags[name] += 1
        else:
            self.backward_lengths.append(length)
            self.backward_times.append(timestamp)
            if flags is not None:
                for name, mask in TCP_FLAG_MAP.items():
                    if flags & mask:
                        self.backward_flags[name] += 1

    def to_row(self) -> Dict[str, float]:
        def _stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {
                    "count": 0,
                    "total": 0,
                    "mean": 0.0,
                    "std": 0.0,
                    "max": 0,
                    "min": 0,
                }
            arr = np.array(values, dtype=float)
            return {
                "count": int(arr.size),
                "total": float(arr.sum()),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                "max": float(arr.max()),
                "min": float(arr.min()),
            }

        fwd_stats = _stats(self.forward_lengths)
        bwd_stats = _stats(self.backward_lengths)
        total_packets = fwd_stats["count"] + bwd_stats["count"]
        total_bytes = fwd_stats["total"] + bwd_stats["total"]

        all_times = self.forward_times + self.backward_times
        if all_times:
            duration = float(max(all_times) - min(all_times))
        else:
            duration = 0.0

        fwd_intervals = _inter_arrivals(self.forward_times)
        bwd_intervals = _inter_arrivals(self.backward_times)

        features: Dict[str, float] = {
            "pcap_file": self.pcap_file,
            "flow_id": f"{self.src_ip}:{self.src_port}->{self.dst_ip}:{self.dst_port}/{self.protocol}",
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "flow_duration": duration,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "fwd_packets": fwd_stats["count"],
            "bwd_packets": bwd_stats["count"],
            "fwd_bytes": fwd_stats["total"],
            "bwd_bytes": bwd_stats["total"],
            "fwd_pkt_len_mean": fwd_stats["mean"],
            "bwd_pkt_len_mean": bwd_stats["mean"],
            "fwd_pkt_len_std": fwd_stats["std"],
            "bwd_pkt_len_std": bwd_stats["std"],
            "fwd_pkt_len_max": fwd_stats["max"],
            "bwd_pkt_len_max": bwd_stats["max"],
            "fwd_pkt_len_min": fwd_stats["min"],
            "bwd_pkt_len_min": bwd_stats["min"],
            "throughput_bytes_s": total_bytes / duration if duration > 0 else 0.0,
            "packets_per_s": total_packets / duration if duration > 0 else 0.0,
            "fwd_bwd_pkt_ratio": (fwd_stats["count"] / bwd_stats["count"]) if bwd_stats["count"] else float(fwd_stats["count"]),
            "fwd_bwd_byte_ratio": (fwd_stats["total"] / bwd_stats["total"]) if bwd_stats["total"] else float(fwd_stats["total"]),
            "mean_fwd_inter": float(fwd_intervals.mean()) if fwd_intervals.size else 0.0,
            "mean_bwd_inter": float(bwd_intervals.mean()) if bwd_intervals.size else 0.0,
            "std_fwd_inter": float(fwd_intervals.std(ddof=0)) if fwd_intervals.size > 1 else 0.0,
            "std_bwd_inter": float(bwd_intervals.std(ddof=0)) if bwd_intervals.size > 1 else 0.0,
        }

        features.update(_hist_features(self.forward_lengths, LENGTH_BINS, "fwd_len_bin"))
        features.update(_hist_features(self.backward_lengths, LENGTH_BINS, "bwd_len_bin"))
        features.update(_hist_features(fwd_intervals.tolist(), INTERVAL_BINS, "fwd_gap_bin"))
        features.update(_hist_features(bwd_intervals.tolist(), INTERVAL_BINS, "bwd_gap_bin"))

        for flag, count in self.forward_flags.items():
            features[f"fwd_flag_{flag}"] = int(count)
        for flag, count in self.backward_flags.items():
            features[f"bwd_flag_{flag}"] = int(count)

        # 确保标志位列存在
        for flag in TCP_FLAG_MAP:
            features.setdefault(f"fwd_flag_{flag}", 0)
            features.setdefault(f"bwd_flag_{flag}", 0)

        return features


def extract_features(
    pcap_path: str,
    output_csv: str,
    progress_cb: ProgressCallback = None,
) -> str:
    """从单个 PCAP 文件提取高维流量特征并保存为 CSV。"""

    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"文件不存在: {pcap_path}")

    total_packets = _count_packets(pcap_path)
    flows: Dict[Tuple[str, str, int, int, str], FlowAccumulator] = {}
    processed = 0
    pcap_name = os.path.basename(pcap_path)

    with open(pcap_path, "rb") as handle:
        reader = _open_reader(handle)
        for ts, buf in _iterate_packets(reader):
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip_layer = eth.data
                if not isinstance(ip_layer, (dpkt.ip.IP, dpkt.ip6.IP6)):
                    continue

                proto_num = ip_layer.p if isinstance(ip_layer, dpkt.ip.IP) else ip_layer.nxt
                proto_name = _protocol_name(int(proto_num))

                src_ip = _ip_to_str(ip_layer.src)
                dst_ip = _ip_to_str(ip_layer.dst)

                transport = ip_layer.data
                src_port = int(getattr(transport, "sport", 0))
                dst_port = int(getattr(transport, "dport", 0))

                key_fwd = (src_ip, dst_ip, src_port, dst_port, proto_name)
                key_bwd = (dst_ip, src_ip, dst_port, src_port, proto_name)

                if key_fwd in flows:
                    acc = flows[key_fwd]
                    direction = "fwd"
                elif key_bwd in flows:
                    acc = flows[key_bwd]
                    direction = "bwd"
                else:
                    acc = FlowAccumulator(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                        protocol=proto_name,
                        pcap_file=pcap_name,
                    )
                    flows[key_fwd] = acc
                    direction = "fwd"

                flags = None
                if proto_name == "TCP" and hasattr(transport, "flags"):
                    flags = int(transport.flags)

                acc.add_packet(length=len(buf), timestamp=float(ts), direction=direction, flags=flags)
            except Exception:
                continue

            processed += 1
            if progress_cb and total_packets:
                if processed % 200 == 0 or processed == total_packets:
                    progress_cb(min(99, int(processed * 100 / total_packets)))

    if not flows:
        raise ValueError(f"{pcap_path} 未提取到任何有效数据")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    records = [acc.to_row() for acc in flows.values()]
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    if progress_cb:
        progress_cb(100)

    return output_csv


def _notify(cb: ProgressCallback, value: int) -> None:
    if cb:
        cb(max(0, min(100, value)))


def extract_features_dir(
    pcap_dir: str,
    output_dir: str,
    *,
    workers: int = 4,
    progress_cb: ProgressCallback = None,
) -> List[str]:
    """批量提取目录下 PCAP/PCAPNG 文件的高维特征（多线程）。"""

    if not os.path.isdir(pcap_dir):
        raise FileNotFoundError(f"目录不存在: {pcap_dir}")

    patterns = [os.path.join(pcap_dir, "*.pcap"), os.path.join(pcap_dir, "*.pcapng")]
    pcap_files = sorted({path for pattern in patterns for path in glob.glob(pattern)})

    if not pcap_files:
        raise RuntimeError(f"目录中没有找到 pcap/pcapng 文件: {pcap_dir}")

    os.makedirs(output_dir, exist_ok=True)

    total = len(pcap_files)
    _notify(progress_cb, 0)

    def _task(pcap_path: str) -> Tuple[str, str]:
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_features.csv")
        extract_features(pcap_path, csv_path, progress_cb=None)
        return pcap_path, csv_path

    results: List[Tuple[str, str]] = []
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_task, path): path for path in pcap_files}
        completed = 0
        for future in as_completed(future_map):
            pcap_path, csv_path = future.result()
            results.append((pcap_path, csv_path))
            completed += 1
            _notify(progress_cb, int(completed * 100 / total))

    results.sort(key=lambda item: item[0])
    return [csv_path for _, csv_path in results]