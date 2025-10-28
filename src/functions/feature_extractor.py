"""PCAP 特征提取模块，输出高维流量特征。"""

import glob
import importlib.util
import inspect
import os
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import dpkt
from dpkt import icmp, icmp6, udp
import numpy as np
import pandas as pd

from src.configuration import load_config
from src.functions.logging_utils import get_logger
from src.functions.annotations import configured_plugin_dirs

Packet = Tuple[float, bytes]
ProgressCallback = Optional[Callable[[int], None]]

TCP_FLAG_MAP = {
    "fin": dpkt.tcp.TH_FIN,
    "syn": dpkt.tcp.TH_SYN,
    "rst": dpkt.tcp.TH_RST,
    "psh": dpkt.tcp.TH_PUSH,
    "ack": dpkt.tcp.TH_ACK,
    "urg": dpkt.tcp.TH_URG,
    "cwr": getattr(dpkt.tcp, "TH_CWR", 0),
    "ece": getattr(dpkt.tcp, "TH_ECE", 0),
}

logger = get_logger(__name__)

PluginExtractor = Callable[["FlowAccumulator", Dict[str, float]], Dict[str, float]]

LENGTH_BINS = np.linspace(0, 8192, 257)
INTERVAL_BINS = np.linspace(0.0, 5.0, 257)
MAX_PKTS_PER_FLOW = 10_000
FAST_PACKET_THRESHOLD = 1_000_000
FAST_SAMPLE_RATE = 10

PLUGIN_EXTRACTORS: List[PluginExtractor] = []
PLUGIN_INFO: List[Dict[str, object]] = []
_PLUGINS_INITIALIZED = False


def _wrap_plugin_function(func: Callable, module_name: str, display_name: str) -> PluginExtractor:
    def wrapper(flow: "FlowAccumulator", base_features: Dict[str, float]) -> Dict[str, float]:
        try:
            try:
                result = func(flow, base_features)
            except TypeError:
                result = func(flow)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Feature plugin %s.%s failed: %s", module_name, display_name, exc)
            return {}

        if result is None:
            return {}
        if not isinstance(result, dict):
            logger.debug(
                "Plugin %s.%s returned %s instead of dict; ignored",
                module_name,
                display_name,
                type(result).__name__,
            )
            return {}

        sanitized: Dict[str, float] = {}
        for key, value in result.items():
            if value is None:
                continue
            new_key = str(key)
            try:
                sanitized[new_key] = float(value)
            except Exception:
                try:
                    sanitized[new_key] = float(np.asarray(value).astype(float))
                except Exception:
                    logger.debug(
                        "Plugin %s.%s feature %s is non-numeric; skipped",
                        module_name,
                        display_name,
                        new_key,
                    )
        return sanitized

    wrapper.__name__ = f"{module_name}:{display_name}"
    return wrapper


def load_feature_plugins(force_reload: bool = False) -> List[PluginExtractor]:
    global PLUGIN_EXTRACTORS, PLUGIN_INFO, _PLUGINS_INITIALIZED

    config = load_config()
    plugins_cfg = config.get("plugins") if isinstance(config, dict) else {}
    autoload = True
    reload_on_start = False
    if isinstance(plugins_cfg, dict):
        autoload = bool(plugins_cfg.get("autoload", True))
        reload_on_start = bool(plugins_cfg.get("reload_on_start", False))

    if not autoload and not force_reload:
        _PLUGINS_INITIALIZED = True
        PLUGIN_EXTRACTORS = []
        PLUGIN_INFO = []
        return PLUGIN_EXTRACTORS

    if force_reload or reload_on_start:
        PLUGIN_EXTRACTORS = []
        PLUGIN_INFO = []

    if PLUGIN_EXTRACTORS and not force_reload:
        _PLUGINS_INITIALIZED = True
        return PLUGIN_EXTRACTORS

    seen_modules = set()
    for directory in configured_plugin_dirs():
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.name.startswith("__"):
                continue
            module_name = f"feature_plugin_{path.stem}"
            if module_name in seen_modules and not force_reload:
                continue
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.warning("加载特征插件 %s 失败: %s", path, exc)
                continue

            extractors: List[Callable] = []
            registrar = getattr(module, "register_feature_extractors", None)
            if callable(registrar):
                try:
                    registered = registrar()
                    if registered:
                        extractors.extend(list(registered))
                except Exception as exc:
                    logger.warning("插件 %s register_feature_extractors 执行失败: %s", path, exc)

            feature_attr = getattr(module, "FEATURE_EXTRACTORS", None)
            if feature_attr:
                try:
                    if isinstance(feature_attr, (list, tuple, set)):
                        extractors.extend(list(feature_attr))
                    elif callable(feature_attr):
                        generated = feature_attr()
                        if generated:
                            extractors.extend(list(generated))
                except Exception as exc:
                    logger.warning("插件 %s FEATURE_EXTRACTORS 解析失败: %s", path, exc)

            wrappers: List[PluginExtractor] = []
            names: List[str] = []
            for entry in extractors:
                func = None
                display = None
                if callable(entry):
                    func = entry
                    display = getattr(entry, "__name__", path.stem)
                elif isinstance(entry, dict):
                    candidate = (
                        entry.get("callable")
                        or entry.get("func")
                        or entry.get("function")
                        or entry.get("handler")
                    )
                    if callable(candidate):
                        func = candidate
                        display = str(entry.get("name", getattr(candidate, "__name__", path.stem)))
                if func is None:
                    continue
                wrappers.append(_wrap_plugin_function(func, module.__name__, display or path.stem))
                names.append(display or getattr(func, "__name__", path.stem))

            if wrappers:
                PLUGIN_EXTRACTORS.extend(wrappers)
                PLUGIN_INFO.append(
                    {
                        "module": module.__name__,
                        "path": str(path),
                        "extractors": names,
                    }
                )
                seen_modules.add(module_name)

    _PLUGINS_INITIALIZED = True
    return PLUGIN_EXTRACTORS


def get_loaded_plugin_info() -> List[Dict[str, object]]:
    """Return metadata about currently loaded feature plugins."""

    load_feature_plugins()
    return list(PLUGIN_INFO)


load_feature_plugins()


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


def _payload_length(transport) -> int:
    if transport is None:
        return 0
    if isinstance(transport, dpkt.tcp.TCP):
        data = transport.data or b""
        return int(len(data))
    if isinstance(transport, udp.UDP):
        data = transport.data or b""
        return int(len(data))
    data = getattr(transport, "data", b"")
    if isinstance(data, (bytes, bytearray)):
        return int(len(data))
    try:
        return int(len(bytes(data)))
    except Exception:
        if hasattr(data, "pack"):
            try:
                return int(len(data.pack()))
            except Exception:
                return 0
        return 0


def _header_length(ip_layer, transport) -> int:
    ip_header = 0
    if isinstance(ip_layer, dpkt.ip.IP):
        ip_header = int(getattr(ip_layer, "hl", 0)) * 4
        if ip_header <= 0:
            ip_header = 20
    elif isinstance(ip_layer, dpkt.ip6.IP6):
        ip_header = 40

    transport_header = 0
    if isinstance(transport, dpkt.tcp.TCP):
        offset = int(getattr(transport, "off", 0))
        transport_header = offset * 4 if offset else 20
    elif isinstance(transport, udp.UDP):
        transport_header = 8
    elif isinstance(transport, (icmp.ICMP, icmp6.ICMP6)):
        transport_header = 8
    else:
        try:
            raw = bytes(transport)
            data = getattr(transport, "data", b"")
            payload_len = len(data) if isinstance(data, (bytes, bytearray)) else 0
            if payload_len <= len(raw):
                transport_header = len(raw) - payload_len
        except Exception:
            transport_header = 0

    total = ip_header + transport_header
    return int(total) if total > 0 else 0


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


def _temporal_entropy(arr: np.ndarray, bins: int = 20) -> float:
    if arr.size == 0:
        return 0.0
    try:
        hist, _ = np.histogram(arr, bins=bins, density=True)
    except Exception:
        return 0.0
    hist = hist[np.isfinite(hist) & (hist > 0)]
    if hist.size == 0:
        return 0.0
    probs = hist / np.sum(hist)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _burst_statistics(arr: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    if arr.size == 0:
        return {"count": 0.0, "max": 0.0, "avg": 0.0, "density": 0.0}
    mask = arr <= threshold
    if not np.any(mask):
        return {"count": 0.0, "max": 0.0, "avg": 0.0, "density": 0.0}
    bursts: List[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            bursts.append(current)
            current = 0
    if current:
        bursts.append(current)
    if not bursts:
        return {"count": 0.0, "max": 0.0, "avg": 0.0, "density": 0.0}
    bursts_arr = np.asarray(bursts, dtype=float)
    return {
        "count": float(len(bursts)),
        "max": float(np.max(bursts_arr)),
        "avg": float(np.mean(bursts_arr)),
        "density": float(np.sum(bursts_arr) / arr.size),
    }


def _duration_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def _activity_idle_metrics(times: List[float], threshold: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
    if len(times) <= 1:
        return _duration_stats([]), _duration_stats([])
    arr = np.sort(np.asarray(times, dtype=float))
    active_periods: List[float] = []
    idle_periods: List[float] = []
    start = float(arr[0])
    last = float(arr[0])
    for value in arr[1:]:
        gap = float(max(value - last, 0.0))
        if gap <= threshold:
            last = float(value)
            continue
        active_periods.append(max(last - start, 0.0))
        idle_periods.append(gap)
        start = float(value)
        last = float(value)
    active_periods.append(max(last - start, 0.0))
    return _duration_stats(active_periods), _duration_stats(idle_periods)


def _bulk_metrics(lengths: List[int], times: List[float]) -> Dict[str, float]:
    if not lengths:
        return {"avg_bytes": 0.0, "avg_packets": 0.0, "avg_rate": 0.0}
    total_bytes = float(sum(int(v) for v in lengths))
    total_packets = float(len(lengths))
    if len(times) >= 2:
        duration = float(max(times) - min(times))
    else:
        duration = 0.0
    avg_rate = total_bytes / duration if duration > 0 else 0.0
    return {
        "avg_bytes": total_bytes,
        "avg_packets": total_packets,
        "avg_rate": avg_rate,
    }


def _window_activity_metrics(times: List[float], window_size: float = 1.0) -> Dict[str, float]:
    if len(times) == 0:
        return {
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "active_ratio": 0.0,
        }
    arr = np.sort(np.asarray(times, dtype=float))
    counts: List[int] = []
    start = 0
    for end, value in enumerate(arr):
        while arr[start] < value - window_size:
            start += 1
        counts.append(end - start + 1)
    counts_arr = np.asarray(counts, dtype=float)
    duration = float(arr[-1] - arr[0]) if arr.size > 1 else window_size
    total_windows = max(int(np.ceil(duration / max(window_size, 1e-6))), 1)
    active_windows = int(np.sum(counts_arr > 0))
    return {
        "max": float(np.max(counts_arr)),
        "mean": float(np.mean(counts_arr)),
        "std": float(np.std(counts_arr, ddof=0)),
        "active_ratio": float(active_windows / max(total_windows, 1)),
    }


def _autocorr_strength(arr: np.ndarray) -> float:
    if arr.size < 3:
        return 0.0
    centered = arr - arr.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 1e-9:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")
    mid = corr.size // 2
    tail = corr[mid + 1 :]
    if tail.size == 0:
        return 0.0
    peak = float(np.max(tail))
    return float(np.clip(peak / denom, 0.0, 1.0))


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
    forward_payload_lengths: List[int] = field(default_factory=list)
    backward_payload_lengths: List[int] = field(default_factory=list)
    forward_header_lengths: List[int] = field(default_factory=list)
    backward_header_lengths: List[int] = field(default_factory=list)
    forward_flags: Counter = field(default_factory=Counter)
    backward_flags: Counter = field(default_factory=Counter)
    forward_windows: List[int] = field(default_factory=list)
    backward_windows: List[int] = field(default_factory=list)
    truncated: int = 0

    def add_packet(
        self,
        *,
        length: int,
        timestamp: float,
        direction: str,
        flags: Optional[int] = None,
        window: Optional[int] = None,
        payload_length: Optional[int] = None,
        header_length: Optional[int] = None,
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
            payload = int(payload_length or 0)
            header = int(header_length or 0)
            self.forward_payload_lengths.append(payload)
            self.forward_header_lengths.append(header)
            if window is not None:
                try:
                    self.forward_windows.append(int(window))
                except (TypeError, ValueError):
                    pass
            if flags is not None:
                for name, mask in TCP_FLAG_MAP.items():
                    if flags & mask:
                        self.forward_flags[name] += 1
        else:
            self.backward_lengths.append(length)
            self.backward_times.append(timestamp)
            payload = int(payload_length or 0)
            header = int(header_length or 0)
            self.backward_payload_lengths.append(payload)
            self.backward_header_lengths.append(header)
            if window is not None:
                try:
                    self.backward_windows.append(int(window))
                except (TypeError, ValueError):
                    pass
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

        combined_lengths = self.forward_lengths + self.backward_lengths
        combined_stats = _stats(combined_lengths)
        combined_arr = (
            np.asarray(combined_lengths, dtype=float)
            if combined_lengths
            else np.asarray([], dtype=float)
        )
        combined_var = float(np.var(combined_arr, ddof=0)) if combined_arr.size > 1 else 0.0

        all_times = self.forward_times + self.backward_times
        flow_start_time = float(min(all_times)) if all_times else 0.0

        active_stats, idle_stats = _activity_idle_metrics(all_times)

        fwd_bulk = _bulk_metrics(self.forward_payload_lengths, self.forward_times)
        bwd_bulk = _bulk_metrics(self.backward_payload_lengths, self.backward_times)

        fwd_header_total = (
            float(sum(self.forward_header_lengths))
            if self.forward_header_lengths
            else 0.0
        )
        bwd_header_total = (
            float(sum(self.backward_header_lengths))
            if self.backward_header_lengths
            else 0.0
        )

        act_data_fwd = int(sum(1 for value in self.forward_payload_lengths if value > 0))
        min_seg_fwd = float(
            min((value for value in self.forward_payload_lengths if value > 0), default=0)
        )
        init_win_fwd = float(self.forward_windows[0]) if self.forward_windows else 0.0
        init_win_bwd = float(self.backward_windows[0]) if self.backward_windows else 0.0

        if all_times:
            duration = float(max(all_times) - min(all_times))
        else:
            duration = 0.0

        fwd_packets_per_s = (
            float(fwd_stats["count"]) / duration if duration > 0 else 0.0
        )
        bwd_packets_per_s = (
            float(bwd_stats["count"]) / duration if duration > 0 else 0.0
        )
        down_up_ratio = (
            float(bwd_stats["total"]) / float(fwd_stats["total"])
            if fwd_stats["total"] > 0
            else float(bwd_stats["total"])
        )

        fwd_intervals = _inter_arrivals(self.forward_times)
        bwd_intervals = _inter_arrivals(self.backward_times)

        def _interval_metrics(arr: np.ndarray) -> Dict[str, float]:
            if arr.size == 0:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "var": 0.0,
                    "median": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "burstiness": 0.0,
                    "cv": 0.0,
                    "autocorr": 0.0,
                }
            std_val = float(arr.std(ddof=0)) if arr.size > 1 else 0.0
            mean_val = float(arr.mean())
            var_val = float(arr.var(ddof=0)) if arr.size > 1 else 0.0
            cv_val = float(std_val / mean_val) if mean_val > 1e-9 else 0.0
            return {
                "mean": mean_val,
                "std": std_val,
                "var": var_val,
                "median": float(np.quantile(arr, 0.5)),
                "p90": float(np.quantile(arr, 0.9)),
                "p95": float(np.quantile(arr, 0.95)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "burstiness": _burstiness(arr),
                "cv": float(np.clip(cv_val, 0.0, 1e3)),
                "autocorr": _autocorr_strength(arr),
            }

        fwd_interval_stats = _interval_metrics(fwd_intervals)
        bwd_interval_stats = _interval_metrics(bwd_intervals)
        interval_arrays = [arr for arr in (fwd_intervals, bwd_intervals) if arr.size]
        all_intervals = (
            np.concatenate(interval_arrays)
            if interval_arrays
            else np.empty(0, dtype=float)
        )
        combined_interval_stats = _interval_metrics(all_intervals)
        fwd_bursts = _burst_statistics(fwd_intervals)
        bwd_bursts = _burst_statistics(bwd_intervals)
        combined_bursts = _burst_statistics(all_intervals)
        entropy_all = _temporal_entropy(all_intervals)

        flow_iat_total = float(all_intervals.sum()) if all_intervals.size else 0.0
        flow_iat_min = float(all_intervals.min()) if all_intervals.size else 0.0
        flow_iat_max = float(all_intervals.max()) if all_intervals.size else 0.0
        flow_iat_std = float(all_intervals.std(ddof=0)) if all_intervals.size > 1 else 0.0

        fwd_iat_total = float(fwd_intervals.sum()) if fwd_intervals.size else 0.0
        fwd_iat_min = float(fwd_intervals.min()) if fwd_intervals.size else 0.0
        fwd_iat_max = float(fwd_intervals.max()) if fwd_intervals.size else 0.0
        fwd_iat_std = float(fwd_intervals.std(ddof=0)) if fwd_intervals.size > 1 else 0.0

        bwd_iat_total = float(bwd_intervals.sum()) if bwd_intervals.size else 0.0
        bwd_iat_min = float(bwd_intervals.min()) if bwd_intervals.size else 0.0
        bwd_iat_max = float(bwd_intervals.max()) if bwd_intervals.size else 0.0
        bwd_iat_std = float(bwd_intervals.std(ddof=0)) if bwd_intervals.size > 1 else 0.0

        total_window_stats = _window_activity_metrics(all_times)
        fwd_window_activity = _window_activity_metrics(self.forward_times)
        bwd_window_activity = _window_activity_metrics(self.backward_times)

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
            "var_fwd_inter": fwd_interval_stats["var"],
            "var_bwd_inter": bwd_interval_stats["var"],
            "cv_fwd_inter": fwd_interval_stats["cv"],
            "cv_bwd_inter": bwd_interval_stats["cv"],
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
            "autocorr_fwd_inter": fwd_interval_stats["autocorr"],
            "autocorr_bwd_inter": bwd_interval_stats["autocorr"],
            "mean_inter_all": combined_interval_stats["mean"],
            "std_inter_all": combined_interval_stats["std"],
            "var_inter_all": combined_interval_stats["var"],
            "burstiness_inter_all": combined_interval_stats["burstiness"],
            "cv_inter_all": combined_interval_stats["cv"],
            "autocorr_inter_all": combined_interval_stats["autocorr"],
            "burst_count_fwd": fwd_bursts["count"],
            "burst_count_bwd": bwd_bursts["count"],
            "burst_count_all": combined_bursts["count"],
            "burst_max_fwd": fwd_bursts["max"],
            "burst_max_bwd": bwd_bursts["max"],
            "burst_max_all": combined_bursts["max"],
            "burst_avg_fwd": fwd_bursts["avg"],
            "burst_avg_bwd": bwd_bursts["avg"],
            "burst_avg_all": combined_bursts["avg"],
            "burst_density_fwd": fwd_bursts["density"],
            "burst_density_bwd": bwd_bursts["density"],
            "burst_density_all": combined_bursts["density"],
            "inter_arrival_entropy": entropy_all,
            "window_packets_max": total_window_stats["max"],
            "window_packets_mean": total_window_stats["mean"],
            "window_packets_std": total_window_stats["std"],
            "window_activity_ratio": total_window_stats["active_ratio"],
            "window_packets_fwd_max": fwd_window_activity["max"],
            "window_packets_bwd_max": bwd_window_activity["max"],
            "window_activity_fwd": fwd_window_activity["active_ratio"],
            "window_activity_bwd": bwd_window_activity["active_ratio"],
            "combined_pkt_len_mean": combined_stats["mean"],
            "combined_pkt_len_std": combined_stats["std"],
            "combined_pkt_len_max": combined_stats["max"],
            "combined_pkt_len_min": combined_stats["min"],
            "combined_pkt_len_var": combined_var,
            "fwd_header_length_total": fwd_header_total,
            "bwd_header_length_total": bwd_header_total,
            "fwd_packets_per_second": fwd_packets_per_s,
            "bwd_packets_per_second": bwd_packets_per_s,
            "down_up_ratio": down_up_ratio,
            "act_data_pkt_fwd": float(act_data_fwd),
            "min_seg_size_forward": min_seg_fwd,
            "init_win_bytes_forward": init_win_fwd,
            "init_win_bytes_backward": init_win_bwd,
            "active_mean": active_stats["mean"],
            "active_std": active_stats["std"],
            "active_max": active_stats["max"],
            "active_min": active_stats["min"],
            "idle_mean": idle_stats["mean"],
            "idle_std": idle_stats["std"],
            "idle_max": idle_stats["max"],
            "idle_min": idle_stats["min"],
            "flow_start_time": flow_start_time,
            "flow_iat_total": flow_iat_total,
            "flow_iat_min": flow_iat_min,
            "flow_iat_max": flow_iat_max,
            "flow_iat_std": flow_iat_std,
            "fwd_iat_total": fwd_iat_total,
            "fwd_iat_min": fwd_iat_min,
            "fwd_iat_max": fwd_iat_max,
            "fwd_iat_std": fwd_iat_std,
            "bwd_iat_total": bwd_iat_total,
            "bwd_iat_min": bwd_iat_min,
            "bwd_iat_max": bwd_iat_max,
            "bwd_iat_std": bwd_iat_std,
            "fwd_bulk_avg_bytes": fwd_bulk["avg_bytes"],
            "fwd_bulk_avg_packets": fwd_bulk["avg_packets"],
            "fwd_bulk_avg_rate": fwd_bulk["avg_rate"],
            "bwd_bulk_avg_bytes": bwd_bulk["avg_bytes"],
            "bwd_bulk_avg_packets": bwd_bulk["avg_packets"],
            "bwd_bulk_avg_rate": bwd_bulk["avg_rate"],
            "subflow_fwd_packets": fwd_stats["count"],
            "subflow_bwd_packets": bwd_stats["count"],
            "subflow_fwd_bytes": fwd_stats["total"],
            "subflow_bwd_bytes": bwd_stats["total"],
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
        ack_total = self.forward_flags.get("ack", 0) + self.backward_flags.get("ack", 0)
        features["ack_to_packet_ratio"] = (
            float(ack_total) / float(total_packets) if total_packets else 0.0
        )
        fin_total = self.forward_flags.get("fin", 0) + self.backward_flags.get("fin", 0)
        syn_total = self.forward_flags.get("syn", 0) + self.backward_flags.get("syn", 0)
        psh_total = self.forward_flags.get("psh", 0) + self.backward_flags.get("psh", 0)
        urg_total = self.forward_flags.get("urg", 0) + self.backward_flags.get("urg", 0)
        cwr_total = self.forward_flags.get("cwr", 0) + self.backward_flags.get("cwr", 0)
        ece_total = self.forward_flags.get("ece", 0) + self.backward_flags.get("ece", 0)

        def _window_stats(values: List[int]) -> Dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "max": 0.0, "median": 0.0}
            arr = np.asarray(values, dtype=float)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if arr.size > 1 else 0.0,
                "max": float(arr.max()),
                "median": float(np.median(arr)),
            }

        fwd_window = _window_stats(self.forward_windows)
        bwd_window = _window_stats(self.backward_windows)
        combined_windows = self.forward_windows + self.backward_windows
        total_window_stats_num = _window_stats(combined_windows)
        features["tcp_window_fwd_mean"] = fwd_window["mean"]
        features["tcp_window_bwd_mean"] = bwd_window["mean"]
        features["tcp_window_total_mean"] = total_window_stats_num["mean"]
        features["tcp_window_fwd_std"] = fwd_window["std"]
        features["tcp_window_bwd_std"] = bwd_window["std"]
        features["tcp_window_total_std"] = total_window_stats_num["std"]
        features["tcp_window_fwd_max"] = fwd_window["max"]
        features["tcp_window_bwd_max"] = bwd_window["max"]
        features["tcp_window_total_max"] = total_window_stats_num["max"]
        base_mean = total_window_stats_num["mean"] if total_window_stats_num["mean"] > 0 else 1.0
        features["tcp_window_symmetry"] = float(
            abs(fwd_window["mean"] - bwd_window["mean"]) / base_mean
        ) if base_mean else 0.0

        features["flow_truncated"] = int(self.truncated)

        alias_features = {
            "Flow ID": features["flow_id"],
            "Source IP": self.src_ip,
            "Source Port": int(self.src_port),
            "Destination IP": self.dst_ip,
            "Destination Port": int(self.dst_port),
            "Protocol": self.protocol,
            "Timestamp": flow_start_time,
            "Flow Duration": duration,
            "Total Fwd Packets": float(fwd_stats["count"]),
            "Total Backward Packets": float(bwd_stats["count"]),
            "Total Length of Fwd Packets": float(fwd_stats["total"]),
            "Total Length of Bwd Packets": float(bwd_stats["total"]),
            "Fwd Packet Length Max": float(fwd_stats["max"]),
            "Fwd Packet Length Min": float(fwd_stats["min"]),
            "Fwd Packet Length Mean": float(fwd_stats["mean"]),
            "Fwd Packet Length Std": float(fwd_stats["std"]),
            "Bwd Packet Length Max": float(bwd_stats["max"]),
            "Bwd Packet Length Min": float(bwd_stats["min"]),
            "Bwd Packet Length Mean": float(bwd_stats["mean"]),
            "Bwd Packet Length Std": float(bwd_stats["std"]),
            "Flow Bytes/s": features["throughput_bytes_s"],
            "Flow Packets/s": features["packets_per_s"],
            "Flow IAT Mean": combined_interval_stats["mean"],
            "Flow IAT Std": combined_interval_stats["std"],
            "Flow IAT Max": combined_interval_stats["max"],
            "Flow IAT Min": combined_interval_stats["min"],
            "Fwd IAT Total": fwd_iat_total,
            "Fwd IAT Mean": fwd_interval_stats["mean"],
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Total": bwd_iat_total,
            "Bwd IAT Mean": bwd_interval_stats["mean"],
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": float(self.forward_flags.get("psh", 0)),
            "Bwd PSH Flags": float(self.backward_flags.get("psh", 0)),
            "Fwd URG Flags": float(self.forward_flags.get("urg", 0)),
            "Bwd URG Flags": float(self.backward_flags.get("urg", 0)),
            "Fwd Header Length": fwd_header_total,
            "Bwd Header Length": bwd_header_total,
            "Fwd Packets/s": fwd_packets_per_s,
            "Bwd Packets/s": bwd_packets_per_s,
            "Min Packet Length": float(combined_stats["min"]),
            "Max Packet Length": float(combined_stats["max"]),
            "Packet Length Mean": combined_stats["mean"],
            "Packet Length Std": combined_stats["std"],
            "Packet Length Variance": combined_var,
            "FIN Flag Count": float(fin_total),
            "SYN Flag Count": float(syn_total),
            "RST Flag Count": float(rst_total),
            "PSH Flag Count": float(psh_total),
            "ACK Flag Count": float(ack_total),
            "URG Flag Count": float(urg_total),
            "CWE Flag Count": float(cwr_total),
            "ECE Flag Count": float(ece_total),
            "Down/Up Ratio": down_up_ratio,
            "Average Packet Size": features["avg_pkt_size"],
            "Avg Fwd Segment Size": float(fwd_stats["mean"]),
            "Avg Bwd Segment Size": float(bwd_stats["mean"]),
            "Fwd Avg Bytes/Bulk": fwd_bulk["avg_bytes"],
            "Fwd Avg Packets/Bulk": fwd_bulk["avg_packets"],
            "Fwd Avg Bulk Rate": fwd_bulk["avg_rate"],
            "Bwd Avg Bytes/Bulk": bwd_bulk["avg_bytes"],
            "Bwd Avg Packets/Bulk": bwd_bulk["avg_packets"],
            "Bwd Avg Bulk Rate": bwd_bulk["avg_rate"],
            "Subflow Fwd Packets": float(fwd_stats["count"]),
            "Subflow Fwd Bytes": float(fwd_stats["total"]),
            "Subflow Bwd Packets": float(bwd_stats["count"]),
            "Subflow Bwd Bytes": float(bwd_stats["total"]),
            "Init_Win_bytes_forward": init_win_fwd,
            "Init_Win_bytes_backward": init_win_bwd,
            "act_data_pkt_fwd": float(act_data_fwd),
            "min_seg_size_forward": min_seg_fwd,
            "Active Mean": active_stats["mean"],
            "Active Std": active_stats["std"],
            "Active Max": active_stats["max"],
            "Active Min": active_stats["min"],
            "Idle Mean": idle_stats["mean"],
            "Idle Std": idle_stats["std"],
            "Idle Max": idle_stats["max"],
            "Idle Min": idle_stats["min"],
            "Label": "",
        }
        for key, value in alias_features.items():
            features[key] = value

        if PLUGIN_EXTRACTORS:
            base_snapshot = dict(features)
            for extractor in PLUGIN_EXTRACTORS:
                try:
                    extra = extractor(self, base_snapshot)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.warning("特征插件执行失败 %s: %s", getattr(extractor, "__name__", extractor), exc)
                    continue
                if not extra:
                    continue
                for key, value in extra.items():
                    if value is None:
                        continue
                    target_key = str(key)
                    if target_key in features:
                        suffix = 1
                        while f"{target_key}_plugin{suffix}" in features:
                            suffix += 1
                        target_key = f"{target_key}_plugin{suffix}"
                    try:
                        features[target_key] = float(value)
                    except Exception:
                        logger.debug(
                            "插件特征 %s 返回值 %r 无法转换为 float，已忽略",
                            target_key,
                            value,
                        )

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
                    window_size: Optional[int] = None
                    if proto_name == "TCP" and hasattr(transport, "flags"):
                        flags = int(transport.flags)
                        try:
                            window_size = int(getattr(transport, "win", None))
                        except (TypeError, ValueError):
                            window_size = None
                    elif proto_name == "UDP":
                        window_size = None

                    payload_length = _payload_length(transport)
                    header_size = _header_length(ip_layer, transport)

                    acc.add_packet(
                        length=len(buf),
                        timestamp=float(ts),
                        direction=direction,
                        flags=flags,
                        window=window_size,
                        payload_length=payload_length,
                        header_length=header_size,
                    )
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