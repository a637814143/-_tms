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

from src.functions.logging_utils import get_logger

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

logger = get_logger(__name__)

LENGTH_BINS = np.linspace(0, 8192, 257)
INTERVAL_BINS = np.linspace(0.0, 5.0, 257)
MAX_PKTS_PER_FLOW = 10_000
FAST_PACKET_THRESHOLD = 1_000_000
FAST_SAMPLE_RATE = 10


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


def _burstiness(arr: np.ndarray) -> float:
    if arr.size < 2:
        return 0.0
    mean = float(arr.mean())
    if mean <= 1e-9:
        return 0.0
    std = float(arr.std(ddof=0))
    if std <= 1e-9:
        return 0.0
    return float(np.clip(std / mean, 0.0, 1e3))


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
    truncated: int = 0

    def add_packet(
        self,
        *,
        length: int,
        timestamp: float,
        direction: str,
        flags: Optional[int] = None,
    ) -> None:
        if self.truncated:
            return

        total_packets = len(self.forward_lengths) + len(self.backward_lengths)
        if total_packets >= MAX_PKTS_PER_FLOW:
            self.truncated = 1
            return

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
                    "median": 0.0,
                    "iqr": 0.0,
                }
            arr = np.array(values, dtype=float)
            return {
                "count": int(arr.size),
                "total": float(arr.sum()),
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                "max": float(arr.max()),
                "min": float(arr.min()),
                "median": float(np.median(arr)),
                "iqr": float(max(0.0, np.quantile(arr, 0.75) - np.quantile(arr, 0.25))),
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

        def _interval_metrics(arr: np.ndarray) -> Dict[str, float]:
            if arr.size == 0:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "median": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "max": 0.0,
                    "burstiness": 0.0,
                }
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                "median": float(np.quantile(arr, 0.5)),
                "p90": float(np.quantile(arr, 0.9)),
                "p95": float(np.quantile(arr, 0.95)),
                "max": float(arr.max()),
                "burstiness": _burstiness(arr),
            }

        fwd_interval_stats = _interval_metrics(fwd_intervals)
        bwd_interval_stats = _interval_metrics(bwd_intervals)

        def _flag_rate(counter: Counter, name: str, total: int) -> float:
            if total <= 0:
                return 0.0
            return float(counter.get(name, 0)) / float(total)

        def _flag_imbalance(flag: str) -> float:
            numerator = abs(
                float(self.forward_flags.get(flag, 0))
                - float(self.backward_flags.get(flag, 0))
            )
            denominator = float(
                self.forward_flags.get(flag, 0) + self.backward_flags.get(flag, 0)
            )
            if denominator <= 0:
                return 0.0
            return numerator / denominator

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
            "fwd_pkt_len_median": fwd_stats["median"],
            "bwd_pkt_len_median": bwd_stats["median"],
            "fwd_pkt_len_iqr": fwd_stats["iqr"],
            "bwd_pkt_len_iqr": bwd_stats["iqr"],
            "throughput_bytes_s": total_bytes / duration if duration > 0 else 0.0,
            "packets_per_s": total_packets / duration if duration > 0 else 0.0,
            "fwd_bwd_pkt_ratio": (fwd_stats["count"] / bwd_stats["count"]) if bwd_stats["count"] else float(fwd_stats["count"]),
            "fwd_bwd_byte_ratio": (fwd_stats["total"] / bwd_stats["total"]) if bwd_stats["total"] else float(fwd_stats["total"]),
            "flow_duration_log": float(np.log1p(duration)),
            "avg_pkt_size": (total_bytes / total_packets) if total_packets else 0.0,
            "byte_symmetry": float(
                abs(fwd_stats["total"] - bwd_stats["total"])
                / max(total_bytes, 1.0)
            ),
            "packet_symmetry": float(
                abs(fwd_stats["count"] - bwd_stats["count"])
                / max(total_packets, 1)
            ),
            "mean_fwd_inter": fwd_interval_stats["mean"],
            "mean_bwd_inter": bwd_interval_stats["mean"],
            "std_fwd_inter": fwd_interval_stats["std"],
            "std_bwd_inter": bwd_interval_stats["std"],
            "median_fwd_inter": fwd_interval_stats["median"],
            "median_bwd_inter": bwd_interval_stats["median"],
            "p90_fwd_inter": fwd_interval_stats["p90"],
            "p90_bwd_inter": bwd_interval_stats["p90"],
            "p95_fwd_inter": fwd_interval_stats["p95"],
            "p95_bwd_inter": bwd_interval_stats["p95"],
            "max_fwd_inter": fwd_interval_stats["max"],
            "max_bwd_inter": bwd_interval_stats["max"],
            "burstiness_fwd": fwd_interval_stats["burstiness"],
            "burstiness_bwd": bwd_interval_stats["burstiness"],
        }

        max_idle = max(fwd_interval_stats["max"], bwd_interval_stats["max"])
        features["idle_time_fraction"] = (
            float(np.clip(max_idle / duration, 0.0, 1.0)) if duration > 0 else 0.0
        )

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
            features[f"fwd_flag_{flag}_rate"] = _flag_rate(
                self.forward_flags, flag, fwd_stats["count"]
            )
            features[f"bwd_flag_{flag}_rate"] = _flag_rate(
                self.backward_flags, flag, bwd_stats["count"]
            )
            features[f"flag_{flag}_imbalance"] = _flag_imbalance(flag)

        rst_total = self.forward_flags.get("rst", 0) + self.backward_flags.get("rst", 0)
        features["rst_to_packet_ratio"] = (
            float(rst_total) / float(total_packets) if total_packets else 0.0
        )
        features["flow_truncated"] = int(self.truncated)

        return features


def extract_features(
    pcap_path: str,
    output_csv: str,
    progress_cb: ProgressCallback = None,
    *,
    fast: bool = False,
) -> str:
    """从单个 PCAP 文件提取高维流量特征并保存为 CSV。"""

    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"文件不存在: {pcap_path}")

    total_packets = _count_packets(pcap_path)
    flows: Dict[Tuple[str, str, int, int, str], FlowAccumulator] = {}
    pcap_name = os.path.basename(pcap_path)
    fast_mode = bool(fast) or total_packets > FAST_PACKET_THRESHOLD

    try:
        with open(pcap_path, "rb") as handle:
            reader = _open_reader(handle)
            for idx, (ts, buf) in enumerate(_iterate_packets(reader), start=1):
                if fast_mode and total_packets > FAST_PACKET_THRESHOLD and idx % FAST_SAMPLE_RATE != 0:
                    if progress_cb and total_packets and (idx % 200 == 0 or idx == total_packets):
                        progress_cb(min(95, int(idx * 100 / max(total_packets, 1))))
                    continue

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

                if progress_cb and total_packets and (idx % 200 == 0 or idx == total_packets):
                    progress_cb(min(99, int(idx * 100 / max(total_packets, 1))))
    except Exception as exc:
        logger.error("读取失败 %s: %s", pcap_path, exc)
        raise RuntimeError(f"解析失败 {pcap_path}: {exc}") from exc

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
    fast: bool = False,
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

    def _task(pcap_path: str) -> Tuple[str, Optional[str], Optional[Exception]]:
        base = os.path.splitext(os.path.basename(pcap_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_features.csv")
        last_error: Optional[Exception] = None
        attempt_sequence = [fast, False, True]
        for attempt_fast in dict.fromkeys(attempt_sequence):
            try:
                extract_features(
                    pcap_path,
                    csv_path,
                    progress_cb=None,
                    fast=bool(attempt_fast),
                )
                return pcap_path, csv_path, None
            except Exception as exc:  # noqa: PERF203
                last_error = exc
                if os.path.exists(csv_path):
                    try:
                        os.remove(csv_path)
                    except Exception:
                        pass
        return pcap_path, None, last_error

    results: List[Tuple[str, str]] = []
    failures: List[Tuple[str, Exception]] = []
    max_workers = max(1, workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_task, path): path for path in pcap_files}
        completed = 0
        for future in as_completed(future_map):
            pcap_path = future_map[future]
            try:
                original, csv_path, error = future.result()
            except Exception as exc:  # unexpected errors
                failures.append((pcap_path, exc))
                completed += 1
                _notify(progress_cb, int(completed * 100 / total))
                continue

            if error is not None:
                failures.append((original or pcap_path, error))
            elif csv_path:
                results.append((original, csv_path))
            completed += 1
            _notify(progress_cb, int(completed * 100 / total))

    results.sort(key=lambda item: item[0])
    if failures:
        failed_list = ", ".join(f"{path}: {err}" for path, err in failures)
        logger.error("部分 PCAP 解析失败：%s", failed_list)
        if len(results) == 0:
            raise RuntimeError("所有 PCAP 文件解析失败，请检查日志。")
    return [csv_path for _, csv_path in results]