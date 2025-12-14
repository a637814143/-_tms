# -*- coding: utf-8 -*-
"""从 PCAP 文件提取简化的流量特征。"""

import argparse
import glob
import importlib
import importlib.util
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _load_module(module_path: str, friendly_name: Optional[str] = None) -> ModuleType:
    """Load a module dynamically and provide a clear error message when missing."""

    spec = importlib.util.find_spec(module_path)
    if spec is None:
        display = friendly_name or module_path
        raise ImportError(f"缺少依赖: 请先安装 {display}")
    return importlib.import_module(module_path)


def _load_scapy() -> Optional[Any]:
    try:
        import scapy.all as scapy  # type: ignore
        return scapy
    except Exception:
        return None


pd: Any = _load_module("pandas", friendly_name="pandas")
_scapy: Any = _load_scapy()
if _scapy is not None:
    IP = _scapy.IP
    TCP = _scapy.TCP
    UDP = _scapy.UDP
    PcapReader = _scapy.PcapReader
else:  # pragma: no cover - optional dependency guard
    IP = None
    TCP = None
    UDP = None
    PcapReader = None

FlowKey = Tuple[str, str, int, int, str]


@dataclass
class FlowStats:
    """记录单条网络流的基础统计信息。"""

    forward_packets: int = 0
    backward_packets: int = 0
    forward_bytes: int = 0
    backward_bytes: int = 0
    min_time: Optional[float] = None
    max_time: Optional[float] = None

    def add_packet(
        self,
        *,
        length: int,
        timestamp: float,
        direction: str,
    ) -> None:
        if direction == "fwd":
            self.forward_packets += 1
            self.forward_bytes += length
        else:
            self.backward_packets += 1
            self.backward_bytes += length

        if self.min_time is None or timestamp < self.min_time:
            self.min_time = timestamp
        if self.max_time is None or timestamp > self.max_time:
            self.max_time = timestamp


def _resolve_reader_handle(reader) -> Optional[Any]:
    for attr in ("fd", "f", "file", "fh", "fdesc", "reader"):
        handle = getattr(reader, attr, None)
        if hasattr(handle, "tell"):
            return handle
    return None


def _iter_packets(path: str, progress_cb: Optional[Callable[[Any], None]] = None) -> Iterable:
    with PcapReader(path) as reader:
        handle = _resolve_reader_handle(reader)
        for packet in reader:
            yield packet
            if progress_cb and handle is not None:
                try:
                    progress_cb(handle)
                except Exception:
                    pass


def _count_packets(path: str) -> int:
    total = 0
    with PcapReader(path) as reader:
        for _ in reader:
            total += 1
    return total


def _list_pcap_files(directory: str) -> List[str]:
    patterns = [os.path.join(directory, "*.pcap"), os.path.join(directory, "*.pcapng")]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(files)


def _parse_ports(text: str) -> Set[int]:
    ports: Set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if 0 <= value <= 65535:
            ports.add(value)
    return ports


def _select_files(
    path: str,
    *,
    mode: str,
    files: Optional[Sequence[str]],
    batch_size: int,
    start_index: int,
) -> List[str]:
    provided_files = bool(files)
    if provided_files:
        candidates = [f for f in files if os.path.isfile(f)]
    elif os.path.isdir(path):
        candidates = _list_pcap_files(path)
    else:
        candidates = [path]

    if not candidates:
        raise FileNotFoundError("未在指定路径中找到 pcap/pcapng 文件")

    if mode == "batch":
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if not provided_files:
            start = max(0, start_index)
            end = start + batch_size
            candidates = candidates[start:end]
    elif mode == "file":
        if provided_files:
            candidates = candidates[:1]
        else:
            start = max(0, start_index)
            if start < len(candidates):
                candidates = [candidates[start]]
            else:
                candidates = [candidates[-1]]
    elif mode == "all":
        pass
    else:  # auto
        if os.path.isdir(path):
            # 目录按照 all 处理
            pass
        else:
            candidates = candidates[:1]

    return candidates


def _write_temp_csv(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    fd, temp_path = tempfile.mkstemp(prefix="pcap_info_", suffix=".csv")
    os.close(fd)
    df.to_csv(temp_path, index=False, encoding="utf-8")
    return temp_path


def _flows_to_records(
    flows_map: Dict[FlowKey, FlowStats],
    *,
    file_name: str,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], bool]:
    records: List[Dict[str, object]] = []
    truncated = False

    for (src_ip, dst_ip, src_port, dst_port, proto), stats in flows_map.items():
        if stats.min_time is None or stats.max_time is None:
            continue

        duration = stats.max_time - stats.min_time
        total_fwd = stats.forward_bytes
        total_bwd = stats.backward_bytes
        total_packets_flow = stats.forward_packets + stats.backward_packets

        records.append(
            {
                "pcap_file": file_name,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": proto,
                "flow_duration": duration,
                "total_fwd_pkts": stats.forward_packets,
                "total_bwd_pkts": stats.backward_packets,
                "total_len_fwd_pkts": total_fwd,
                "total_len_bwd_pkts": total_bwd,
                "flow_byts_per_s": (total_fwd + total_bwd) / duration if duration > 0 else 0.0,
                "flow_pkts_per_s": total_packets_flow / duration if duration > 0 else 0.0,
            }
        )

        if limit is not None and limit > 0 and len(records) >= limit:
            truncated = True
            break

    return records, truncated


def _process_file(
    file_path: str,
    *,
    proto_filter: str,
    whitelist: Set[int],
    blacklist: Set[int],
    cancel_cb: Optional[Callable[[], bool]],
    cancel_event: threading.Event,
    progress_hook: Optional[Callable[[int], None]],
) -> Tuple[List[Dict[str, object]], int, Optional[str], bool, bool]:
    return _process_file_limited(
        file_path,
        proto_filter=proto_filter,
        whitelist=whitelist,
        blacklist=blacklist,
        cancel_cb=cancel_cb,
        cancel_event=cancel_event,
        progress_hook=progress_hook,
        progress_ratio=None,
        record_limit=None,
        packet_limit=None,
    )


def _process_file_limited(
    file_path: str,
    *,
    proto_filter: str,
    whitelist: Set[int],
    blacklist: Set[int],
    cancel_cb: Optional[Callable[[], bool]],
    cancel_event: threading.Event,
    progress_hook: Optional[Callable[[int], None]],
    progress_ratio: Optional[Callable[[str, int], None]],
    record_limit: Optional[int],
    packet_limit: Optional[int],
) -> Tuple[List[Dict[str, object]], int, Optional[str], bool, bool]:
    flows_map: Dict[FlowKey, FlowStats] = {}
    processed = 0
    last_reported = 0
    cancelled = False
    truncated = False

    try:
        file_size = os.path.getsize(file_path)
    except OSError:
        file_size = 0
    last_ratio = -1

    def _byte_progress(handle) -> None:
        nonlocal last_ratio
        if not progress_ratio or file_size <= 0:
            return
        try:
            position = handle.tell()
        except Exception:
            return
        pct = min(99, max(0, int(position * 100 / file_size)))
        if pct != last_ratio:
            last_ratio = pct
            progress_ratio(file_path, pct)

    try:
        if progress_ratio:
            progress_ratio(file_path, 0)

        for pkt in _iter_packets(file_path, progress_cb=_byte_progress):
            if cancel_event.is_set() or (cancel_cb and cancel_cb()):
                cancel_event.set()
                cancelled = True
                break

            keys = _flow_key(pkt)
            if keys is None:
                continue

            key_fwd, key_bwd = keys
            proto = key_fwd[4]
            if proto_filter == "tcp" and proto.upper() != "TCP":
                continue
            if proto_filter == "udp" and proto.upper() != "UDP":
                continue

            src_port, dst_port = key_fwd[2], key_fwd[3]
            if whitelist and (src_port not in whitelist and dst_port not in whitelist):
                continue
            if blacklist and (src_port in blacklist or dst_port in blacklist):
                continue

            timestamp = float(pkt.time)
            length = len(pkt)

            stats = flows_map.get(key_fwd)
            if stats is not None:
                stats.add_packet(length=length, timestamp=timestamp, direction="fwd")
            else:
                stats_bwd = flows_map.get(key_bwd)
                if stats_bwd is not None:
                    stats_bwd.add_packet(length=length, timestamp=timestamp, direction="bwd")
                else:
                    if record_limit is not None and record_limit > 0 and len(flows_map) >= record_limit:
                        truncated = True
                        continue
                    stats_new = FlowStats()
                    stats_new.add_packet(length=length, timestamp=timestamp, direction="fwd")
                    flows_map[key_fwd] = stats_new

            processed += 1
            if packet_limit is not None and packet_limit > 0 and processed >= packet_limit:
                truncated = True
                cancelled = True
                cancel_event.set()
                break
            if progress_hook and processed - last_reported >= 200:
                progress_hook(processed - last_reported)
                last_reported = processed

        if progress_hook and processed > last_reported:
            progress_hook(processed - last_reported)

    except Exception as exc:
        if progress_ratio:
            try:
                progress_ratio(file_path, 100)
            except Exception:
                pass
        return [], processed, f"[ERROR] 解析失败 {file_path}: {exc}", cancelled, truncated

    if progress_ratio:
        try:
            progress_ratio(file_path, 100)
        except Exception:
            pass

    file_name = os.path.basename(file_path)
    record_limit = record_limit if record_limit and record_limit > 0 else None
    records, records_truncated = _flows_to_records(
        flows_map,
        file_name=file_name,
        limit=record_limit,
    )
    truncated = truncated or records_truncated
    return records, processed, None, cancelled, truncated


def _flow_key(pkt) -> Optional[Tuple[FlowKey, FlowKey]]:
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
    return forward, backward


def get_pcap_features(
    path: str,
    *,
    workers: int = 1,
    mode: str = "auto",
    batch_size: int = 1,
    start_index: int = 0,
    files: Optional[Sequence[str]] = None,
    proto_filter: str = "both",
    port_whitelist_text: str = "",
    port_blacklist_text: str = "",
    fast: bool = False,
    progress_cb: Optional[Callable[[int], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    record_limit: Optional[int] = None,
    packet_limit: Optional[int] = None,
) -> pd.DataFrame:
    """提取一个或多个 PCAP 文件的流量统计信息，兼容 GUI 所需的参数。"""

    if _scapy is None or IP is None or TCP is None or UDP is None or PcapReader is None:
        raise RuntimeError("缺少依赖 scapy，请先执行：pip install scapy>=2.5")

    if not path and not files:
        raise ValueError("必须提供有效的文件路径或文件列表")

    target_files = _select_files(
        path,
        mode=mode,
        files=files,
        batch_size=batch_size,
        start_index=start_index,
    )

    whitelist = _parse_ports(port_whitelist_text)
    blacklist = _parse_ports(port_blacklist_text)

    record_cap: Optional[int]
    if record_limit is not None:
        try:
            record_cap = int(record_limit)
        except (TypeError, ValueError):
            record_cap = None
        if record_cap is not None and record_cap <= 0:
            record_cap = None
    else:
        record_cap = None

    packet_cap: Optional[int]
    if packet_limit is not None:
        try:
            packet_cap = int(packet_limit)
        except (TypeError, ValueError):
            packet_cap = None
        if packet_cap is not None and packet_cap <= 0:
            packet_cap = None
    else:
        packet_cap = None

    if proto_filter not in {"both", "tcp", "udp"}:
        proto_filter = "both"

    total_packets = 0
    if not fast:
        for file_path in target_files:
            try:
                total_packets += _count_packets(file_path)
            except Exception:
                # 如果计数失败，退化为 fast 模式
                total_packets = 0
                break

    workers = max(1, int(workers))
    file_errors: List[str] = []
    cancel_event = threading.Event()
    processed_packets = 0
    last_emit_pct = -1
    completed_files = 0
    progress_lock = threading.Lock()
    file_progress: Dict[str, int] = {file_path: 0 for file_path in target_files}
    remaining_limit = record_cap
    limit_notes: List[str] = []

    def _packet_progress(delta: int) -> None:
        nonlocal processed_packets, last_emit_pct
        if not progress_cb or total_packets <= 0:
            return
        if delta <= 0:
            return
        with progress_lock:
            processed_packets += delta
            pct = min(99, int(processed_packets * 100 / total_packets)) if total_packets else 0
            if pct != last_emit_pct:
                last_emit_pct = pct
                progress_cb(pct)

    def _file_progress() -> None:
        nonlocal completed_files
        if not progress_cb or total_packets > 0:
            return
        with progress_lock:
            completed_files += 1
            pct = min(99, int(completed_files * 100 / max(1, len(target_files))))
            progress_cb(pct)

    def _per_file_progress(file_path: str, pct: int) -> None:
        if not progress_cb:
            return
        with progress_lock:
            file_progress[file_path] = max(0, min(100, int(pct)))
            if total_packets > 0:
                return
            overall = sum(file_progress.values()) / max(1, len(file_progress))
            progress_cb(min(99, int(overall)))

    start_time = time.time()
    records: List[Dict[str, object]] = []

    use_pool = workers > 1 and len(target_files) > 1

    if use_pool:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_file = {
                executor.submit(
                    _process_file_limited,
                    file_path,
                    proto_filter=proto_filter,
                    whitelist=whitelist,
                    blacklist=blacklist,
                    cancel_cb=cancel_cb,
                    cancel_event=cancel_event,
                    progress_hook=_packet_progress if total_packets and not fast else None,
                    progress_ratio=_per_file_progress,
                    record_limit=remaining_limit,
                    packet_limit=packet_cap,
                ): file_path
                for file_path in target_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    recs, processed, err, cancelled, truncated = future.result()
                    if recs:
                        if remaining_limit is not None and remaining_limit >= 0:
                            if len(recs) > remaining_limit:
                                records.extend(recs[:remaining_limit])
                            else:
                                records.extend(recs)
                            remaining_limit = max(0, remaining_limit - len(recs))
                        else:
                            records.extend(recs)
                    if err:
                        file_errors.append(err)
                    if total_packets <= 0 or fast:
                        _file_progress()
                    if truncated:
                        limit_notes.append(
                            f"{os.path.basename(file_path)} 仅显示前 {len(recs)} 条流量"
                        )
                    if cancelled or (remaining_limit is not None and remaining_limit <= 0):
                        cancel_event.set()
                        break
                except Exception as exc:
                    file_errors.append(f"[ERROR] 解析失败 {file_path}: {exc}")
                    if total_packets <= 0 or fast:
                        _file_progress()
                if cancel_event.is_set():
                    break
    else:
        for file_path in target_files:
            if cancel_event.is_set() or (cancel_cb and cancel_cb()):
                cancel_event.set()
                break

            recs, processed, err, cancelled, truncated = _process_file_limited(
                file_path,
                proto_filter=proto_filter,
                whitelist=whitelist,
                blacklist=blacklist,
                cancel_cb=cancel_cb,
                cancel_event=cancel_event,
                progress_hook=_packet_progress if total_packets and not fast else None,
                progress_ratio=_per_file_progress,
                record_limit=remaining_limit,
                packet_limit=packet_cap,
            )
            if recs:
                if remaining_limit is not None and remaining_limit >= 0:
                    if len(recs) > remaining_limit:
                        records.extend(recs[:remaining_limit])
                    else:
                        records.extend(recs)
                    remaining_limit = max(0, remaining_limit - len(recs))
                else:
                    records.extend(recs)
            if err:
                file_errors.append(err)
            if total_packets <= 0 or fast:
                _file_progress()
            if truncated:
                limit_notes.append(
                    f"{os.path.basename(file_path)} 仅显示前 {len(recs)} 条流量"
                )
            if cancelled or (remaining_limit is not None and remaining_limit <= 0):
                cancel_event.set()
                break

    if limit_notes:
        file_errors.extend(limit_notes)

    df = pd.DataFrame.from_records(records)

    out_csv = _write_temp_csv(df)
    df.attrs["out_csv"] = out_csv
    df.attrs["files_total"] = len(target_files)
    df.attrs["errors"] = "\n".join(file_errors)
    if record_cap is not None:
        df.attrs["record_limit"] = int(record_cap)
    if packet_cap is not None:
        df.attrs["packet_limit"] = int(packet_cap)
    df.attrs["limited"] = bool(limit_notes or (record_cap is not None and remaining_limit is not None and remaining_limit <= 0))
    if limit_notes:
        df.attrs["limit_notes"] = list(limit_notes)

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