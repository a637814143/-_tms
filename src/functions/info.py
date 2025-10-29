# -*- coding: utf-8 -*-
"""
极速查看流量信息（保持功能与列不变）
- 解析层：RawPcapReader/RawPcapNgReader + struct，避免构造庞大的 Scapy 包对象
- 支持：IPv4/IPv6 的 TCP/UDP（常见扩展头已跳过）
- 并行：每个 pcap 一个进程，聚合回收后再拼接
- UI 兼容：返回 DataFrame（UI 预览前 50 行），attrs['out_csv'] 指向全量 CSV
"""

import os
import math
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, Optional, List

import pandas as pd

try:
    from scapy.all import RawPcapReader  # pcap
except Exception:
    RawPcapReader = None

try:
    from scapy.utils import RawPcapNgReader  # pcapng
except Exception:
    RawPcapNgReader = None

# ---------------------- 文件类型检测 ----------------------
_PC_MAGS = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
_PNG_MAG = 0x0A0D0D0A

def _detect_type(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read(4)
    if len(b) < 4:
        return "unknown"
    be = struct.unpack(">I", b)[0]
    le = struct.unpack("<I", b)[0]
    if be in _PC_MAGS or le in _PC_MAGS:
        return "pcap"
    if be == _PNG_MAG or le == _PNG_MAG:
        return "pcapng"
    return "unknown"

# ---------------------- 会话键与累加器 ----------------------
def _flow_key_canonical(src: str, dst: str, sport: int, dport: int, proto: int) -> Tuple[Tuple, int]:
    a = (src, int(sport)); b = (dst, int(dport))
    if a <= b:
        return (src, int(sport), dst, int(dport), int(proto)), +1
    else:
        return (dst, int(dport), src, int(sport), int(proto)), -1

def _new_acc():
    return dict(
        a_ip=None, a_port=0, b_ip=None, b_port=0, proto=0,
        first_ts=None, last_ts=None, bytes=0, pkt_count=0,
        sum_len=0.0, sumsq_len=0.0, min_len=float("inf"), max_len=float("-inf"),
        pkts_fwd=0, pkts_bwd=0, bytes_fwd=0, bytes_bwd=0,
        last_ts_fwd=None, last_ts_bwd=None,
        iat_fwd_sum=0.0, iat_fwd_sumsq=0.0, iat_fwd_cnt=0,
        iat_bwd_sum=0.0, iat_bwd_sumsq=0.0, iat_bwd_cnt=0,
    )

def _agg_update(acc: Dict, length: int, ts: float, direction: int,
                a_ip: str, a_port: int, b_ip: str, b_port: int, proto: int):
    acc["a_ip"] = a_ip; acc["a_port"] = a_port
    acc["b_ip"] = b_ip; acc["b_port"] = b_port
    acc["proto"] = proto
    acc["pkt_count"] += 1
    acc["bytes"] += length
    acc["sum_len"] += length
    acc["sumsq_len"] += (length * length)
    if length < acc["min_len"]: acc["min_len"] = length
    if length > acc["max_len"]: acc["max_len"] = length
    if acc["first_ts"] is None or ts < acc["first_ts"]: acc["first_ts"] = ts
    if acc["last_ts"] is None or ts > acc["last_ts"]: acc["last_ts"] = ts

    if direction >= 0:
        acc["pkts_fwd"] += 1; acc["bytes_fwd"] += length
        if acc["last_ts_fwd"] is not None:
            diff = ts - acc["last_ts_fwd"]
            if diff >= 0:
                acc["iat_fwd_sum"] += diff
                acc["iat_fwd_sumsq"] += diff * diff
                acc["iat_fwd_cnt"] += 1
        acc["last_ts_fwd"] = ts
    else:
        acc["pkts_bwd"] += 1; acc["bytes_bwd"] += length
        if acc["last_ts_bwd"] is not None:
            diff = ts - acc["last_ts_bwd"]
            if diff >= 0:
                acc["iat_bwd_sum"] += diff
                acc["iat_bwd_sumsq"] += diff * diff
                acc["iat_bwd_cnt"] += 1
        acc["last_ts_bwd"] = ts

# ---------------------- 端口白/黑名单 ----------------------
def _parse_ports(s: str) -> set:
    s = (s or "").strip()
    if not s: return set()
    out = set()
    for p in s.split(","):
        p = p.strip()
        if not p: continue
        try: out.add(int(p))
        except Exception: pass
    return out

# ---------------------- 快速解析工具 ----------------------
_ETH_HLEN = 14
_ETH_TYPE_IPV4 = 0x0800
_ETH_TYPE_IPV6 = 0x86DD
_IP_PROTO_TCP = 6
_IP_PROTO_UDP = 17

def _inet_ntoa4(b: bytes) -> str:
    return ".".join(str(x) for x in b)

def _inet_ntoa6(b: bytes) -> str:
    words = struct.unpack("!8H", b)
    best_base = -1; best_len = 0; cur_base = -1; cur_len = 0
    for i in range(8):
        if words[i] == 0:
            if cur_base == -1: cur_base = i; cur_len = 1
            else: cur_len += 1
        else:
            if cur_len > best_len: best_base, best_len = cur_base, cur_len
            cur_base = -1; cur_len = 0
    if cur_len > best_len: best_base, best_len = cur_base, cur_len
    if best_len <= 1: best_base = -1
    parts = []; i = 0
    while i < 8:
        if i == best_base:
            parts.append(""); i += best_len
            if i >= 8: parts.append("")
            continue
        parts.append(hex(words[i])[2:]); i += 1
    return ":".join(parts)

def _parse_ipv4(pkt: bytes, off_ip: int):
    if len(pkt) < off_ip + 20: return None
    vihl = pkt[off_ip]; ihl = (vihl & 0x0F) * 4
    if ihl < 20 or len(pkt) < off_ip + ihl: return None
    proto = pkt[off_ip + 9]
    src = _inet_ntoa4(pkt[off_ip + 12: off_ip + 16])
    dst = _inet_ntoa4(pkt[off_ip + 16: off_ip + 20])
    return proto, src, dst, off_ip + ihl

def _parse_ipv6(pkt: bytes, off_ip: int):
    if len(pkt) < off_ip + 40: return None
    nh = pkt[off_ip + 6]
    src = _inet_ntoa6(pkt[off_ip + 8: off_ip + 24])
    dst = _inet_ntoa6(pkt[off_ip + 24: off_ip + 40])
    l4_off = off_ip + 40
    hop = 0
    while nh in (0, 43, 44, 60) and hop < 3:
        if len(pkt) < l4_off + 2: return None
        next_nh = pkt[l4_off]
        hdr_len = (pkt[l4_off + 1] + 1) * 8
        nh = next_nh; l4_off += hdr_len; hop += 1
    return nh, src, dst, l4_off

def _parse_tcp_udp_ports(pkt: bytes, off_l4: int):
    if len(pkt) < off_l4 + 4: return 0, 0
    sport, dport = struct.unpack_from("!HH", pkt, off_l4)
    return int(sport), int(dport)

# ---------------------- 单文件解析（快速路径） ----------------------
def _process_one_pcap_fast(pcap_path: str, proto_filter: str,
                           wl: set, bl: set) -> pd.DataFrame:
    """
    原地解析一个 pcap/pcapng，返回会话聚合表（列保持与旧版一致）
    """
    if RawPcapReader is None and RawPcapNgReader is None:
        raise RuntimeError("缺少 scapy，无法读取 pcap。请安装：pip install scapy")

    flows: Dict[Tuple, Dict] = defaultdict(_new_acc)

    ftype = _detect_type(pcap_path)
    if ftype == "pcapng" and RawPcapNgReader is None:
        # 没有 pcapng reader 的环境
        raise RuntimeError("当前环境不支持 pcapng，请安装 scapy 完整版本或先转换为 pcap。")

    try:
        # 选择合适的 reader
        reader = RawPcapNgReader(pcap_path) if ftype == "pcapng" else RawPcapReader(pcap_path)
        for pkt_data, meta in reader:
            length = len(pkt_data)
            sec = getattr(meta, "sec", 0)
            usec = getattr(meta, "usec", 0)
            ts = float(sec) + float(usec) / 1e6

            if length < _ETH_HLEN:
                continue
            eth_type = struct.unpack_from("!H", pkt_data, 12)[0]

            proto = None; src = ""; dst = ""; l4_off = 0
            if eth_type == _ETH_TYPE_IPV4:
                parsed = _parse_ipv4(pkt_data, _ETH_HLEN)
                if not parsed: continue
                proto, src, dst, l4_off = parsed
            elif eth_type == _ETH_TYPE_IPV6:
                parsed = _parse_ipv6(pkt_data, _ETH_HLEN)
                if not parsed: continue
                proto, src, dst, l4_off = parsed
            else:
                continue  # 非 IP

            # 协议过滤
            if proto_filter == "tcp" and proto != _IP_PROTO_TCP: continue
            if proto_filter == "udp" and proto != _IP_PROTO_UDP: continue

            sport = dport = 0
            if proto in (_IP_PROTO_TCP, _IP_PROTO_UDP):
                sport, dport = _parse_tcp_udp_ports(pkt_data, l4_off)

            # 端口白/黑名单
            if wl and (sport not in wl and dport not in wl): continue
            if bl and (sport in bl or dport in bl): continue

            key, dir_flag = _flow_key_canonical(src, dst, sport, dport, proto)
            a_ip, a_port, b_ip, b_port, pr_ = key
            _agg_update(flows[key], length, ts, dir_flag, a_ip, a_port, b_ip, b_port, pr_)

    except Exception as e:
        print(f"[WARN] 读取失败: {pcap_path} ({e})")
        return pd.DataFrame()

    # 汇总
    recs: List[Dict] = []
    for _, acc in flows.items():
        cnt = acc["pkt_count"]
        if cnt <= 0: continue
        duration = 0.0
        if (acc["first_ts"] is not None) and (acc["last_ts"] is not None):
            duration = max(0.0, float(acc["last_ts"]) - float(acc["first_ts"]))

        mean_len = acc["sum_len"] / cnt
        var_len = (acc["sumsq_len"] / cnt) - (mean_len * mean_len)
        if var_len < 0: var_len = 0.0
        std_len = math.sqrt(var_len)

        dur = duration if duration > 0 else 0.0
        pps = (cnt / dur) if dur > 0 else 0.0
        bps = (acc["bytes"] * 8 / dur) if dur > 0 else 0.0
        pps_fwd = (acc["pkts_fwd"] / dur) if dur > 0 else 0.0
        pps_bwd = (acc["pkts_bwd"] / dur) if dur > 0 else 0.0
        fwd_bwd_ratio = (acc["bytes_fwd"] / float(acc["bytes_bwd"])) if (acc["bytes_bwd"] > 0) else 0.0

        recs.append({
            "file": os.path.basename(pcap_path),
            "a_ip": acc["a_ip"], "a_port": acc["a_port"],
            "b_ip": acc["b_ip"], "b_port": acc["b_port"],
            "protocol": acc["proto"],
            "pkt_count": cnt, "bytes": acc["bytes"],
            "pkt_len_mean": round(mean_len, 3),
            "pkt_len_std": round(std_len, 3),
            "pkt_len_min": 0 if acc["min_len"] == float("inf") else int(acc["min_len"]),
            "pkt_len_max": 0 if acc["max_len"] == float("-inf") else int(acc["max_len"]),
            "start_ts": acc["first_ts"], "end_ts": acc["last_ts"],
            "duration": round(duration, 6),
            "pps": round(pps, 3), "bps": round(bps, 3),
            "pkts_fwd": acc["pkts_fwd"], "pkts_bwd": acc["pkts_bwd"],
            "bytes_fwd": acc["bytes_fwd"], "bytes_bwd": acc["bytes_bwd"],
            "pps_fwd": round(pps_fwd, 3), "pps_bwd": round(pps_bwd, 3),
            "fwd_bwd_ratio": round(fwd_bwd_ratio, 6),
            "iat_fwd_mean": round((acc['iat_fwd_sum']/acc['iat_fwd_cnt']) if acc['iat_fwd_cnt'] else 0.0, 6),
            "iat_fwd_std":  round(math.sqrt(max(0.0,
                                (acc['iat_fwd_sumsq']/acc['iat_fwd_cnt']) -
                                ((acc['iat_fwd_sum']/acc['iat_fwd_cnt'])**2)
                              )) if acc['iat_fwd_cnt'] else 0.0, 6),
            "iat_bwd_mean": round((acc['iat_bwd_sum']/acc['iat_bwd_cnt']) if acc['iat_bwd_cnt'] else 0.0, 6),
            "iat_bwd_std":  round(math.sqrt(max(0.0,
                                (acc['iat_bwd_sumsq']/acc['iat_bwd_cnt']) -
                                ((acc['iat_bwd_sum']/acc['iat_bwd_cnt'])**2)
                              )) if acc['iat_bwd_cnt'] else 0.0, 6),
        })

    return pd.DataFrame.from_records(recs)

# ---------------------- 对外主函数 ----------------------
def get_pcap_features(path: str,
                      workers: Optional[int] = None,
                      progress_cb=None,
                      mode: str = "auto",
                      batch_size: int = 10,
                      start_index: int = 0,
                      files: Optional[List[str]] = None,
                      proto_filter: str = "both",
                      port_whitelist_text: str = "",
                      port_blacklist_text: str = "",
                      fast: bool = True,
                      cancel_cb=None) -> pd.DataFrame:
    """
    参数/返回与旧版完全一致。
    - 返回 df 仅用于预览；全量 CSV 写在 <项目根>/results/pcap_info_all.csv，路径放在 attrs['out_csv']
    """
    if workers is None:
        import os as _os
        cpu = _os.cpu_count() or 4
        workers = min(max(4, cpu), 16)

    wl = _parse_ports(port_whitelist_text)
    bl = _parse_ports(port_blacklist_text)

    if files:
        file_list = [f for f in files if os.path.isfile(f)]
    elif os.path.isdir(path):
        names = [n for n in os.listdir(path) if n.lower().endswith((".pcap", ".pcapng"))]
        names.sort()
        allf = [os.path.join(path, n) for n in names]
        if mode == "batch":
            s = max(0, int(start_index)); e = min(len(allf), s + max(1, int(batch_size)))
            file_list = allf[s:e]
        elif mode in ("all", "auto"):
            file_list = allf
        else:
            file_list = allf[:1] if allf else []
    else:
        file_list = [path]

    if not file_list:
        raise FileNotFoundError(f"未找到 pcap 文件: {path}")

    results, errors = [], []
    total = len(file_list); done = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one_pcap_fast, f, proto_filter, wl, bl): f for f in file_list}
        for fut in as_completed(futs):
            try:
                df = fut.result()
            except Exception as e:
                errors.append(f"[WARN] 解析失败: {futs[fut]} ({e})")
                df = pd.DataFrame()
            if not df.empty:
                results.append(df)

            done += 1
            if progress_cb: progress_cb(int(done / total * 100))
            if cancel_cb and cancel_cb(): break

    if results:
        out = pd.concat(results, ignore_index=True)
    else:
        out = pd.DataFrame(columns=[
            "file", "a_ip", "a_port", "b_ip", "b_port", "protocol",
            "pkt_count", "bytes", "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max",
            "start_ts", "end_ts", "duration", "pps", "bps",
            "pkts_fwd", "pkts_bwd", "bytes_fwd", "bytes_bwd", "pps_fwd", "pps_bwd",
            "fwd_bwd_ratio", "iat_fwd_mean", "iat_fwd_std", "iat_bwd_mean", "iat_bwd_std"
        ])

    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        results_dir = os.path.join(project_root, "results")
        os.makedirs(results_dir, exist_ok=True)
        out_csv = os.path.join(results_dir, "pcap_info_all.csv")
        out.to_csv(out_csv, index=False, encoding="utf-8")
        out.attrs["out_csv"] = out_csv
    except Exception:
        pass

    out.attrs["files_total"] = total
    if errors: out.attrs["errors"] = "\n".join(errors)
    return out
