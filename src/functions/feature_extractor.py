# src/functions/feature_extractor.py
# -*- coding: utf-8 -*-
"""
轻量级特征提取（兼容 pcap / pcapng）
- 自动识别 pcap / pcapng（dpkt优先，pcapng 无则 Scapy 兜底）
- 会话聚合：把正反五元组合并（A=首包来源；forward=A->B，backward=B->A）
- 输出保留原字段，并新增方向统计与 IAT（前向/后向）
- 支持：
    1) 单包模式：packet_index
    2) 全文件模式：单个 pcap/pcapng
    3) 目录批量：并发处理小文件目录
"""
import os
import socket
import struct
from typing import Dict, List, Tuple, Optional

import dpkt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# pcapng（dpkt）可选
try:
    import dpkt.pcapng as pcapng_mod  # type: ignore
except Exception:
    pcapng_mod = None

# Scapy 兜底（仅在 dpkt 不支持 pcapng 时）
try:
    from scapy.utils import RawPcapNgReader  # type: ignore
    _SCAPY_PCAPNG_AVAILABLE = True
except Exception:
    _SCAPY_PCAPNG_AVAILABLE = False


PCAP_MAGICS = {0xA1B2C3D4, 0xD4C3B2A1, 0xA1B23C4D, 0x4D3CB2A1}
PCAPNG_MAGIC = 0x0A0D0D0A


# ---------------- 文件识别 & 迭代器 ----------------
def _detect_file_type(path: str) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
        if len(head) < 4:
            return "unknown"
        magic_be = struct.unpack(">I", head)[0]
        if magic_be in PCAP_MAGICS:
            return "pcap"
        if magic_be == PCAPNG_MAGIC:
            return "pcapng"
        magic_le = struct.unpack("<I", head)[0]
        if magic_le in PCAP_MAGICS:
            return "pcap"
        if magic_le == PCAPNG_MAGIC:
            return "pcapng"
        return "unknown"


def _iter_packets(path: str):
    """
    逐包读取生成器：yield (ts, buf, pos, total)
    - ts: 浮点时间戳
    - buf: L2 原始字节
    - pos/total: 仅用于进度计算
    """
    f = open(path, "rb")
    try:
        total_size = os.fstat(f.fileno()).st_size
        ftype = _detect_file_type(path)

        if ftype == "pcap":
            reader = dpkt.pcap.Reader(f)
            for ts, buf in reader:
                yield float(ts), buf, f.tell(), total_size

        elif ftype == "pcapng":
            if pcapng_mod is not None:
                reader = pcapng_mod.Reader(f)
                for ts, buf in reader:
                    yield float(ts), buf, f.tell(), total_size
            elif _SCAPY_PCAPNG_AVAILABLE:
                # scapy 兜底：无法提供 pos/total，这里返回 0
                f.close()
                for pkt_data, meta in RawPcapNgReader(path):
                    sec = getattr(meta, "sec", None)
                    usec = getattr(meta, "usec", 0)
                    ts = 0.0 if sec is None else (float(sec) + float(usec) / 1e6)
                    yield ts, pkt_data, 0, 0
            else:
                f.close()
                raise RuntimeError(
                    "pcapng 不受当前环境支持。请安装 dpkt>=1.9.8 或 scapy，或先转换为 pcap。"
                )
        else:
            raise RuntimeError("无法识别的抓包格式。")
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        raise
    else:
        try:
            f.close()
        except Exception:
            pass


# ---------------- 会话聚合（方向感知） ----------------
def _ip_to_str(raw: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, raw)
    except Exception:
        try:
            return socket.inet_ntop(socket.AF_INET6, raw)
        except Exception:
            return ""


def _canon_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: int) -> Tuple[Tuple, int]:
    """方向无关的会话key + 当前包方向（+1 A->B / -1 B->A）"""
    a = (src_ip, int(src_port))
    b = (dst_ip, int(dst_port))
    if a <= b:
        return (src_ip, int(src_port), dst_ip, int(dst_port), int(proto)), +1
    else:
        return (dst_ip, int(dst_port), src_ip, int(src_port), int(proto)), -1


def _new_bucket():
    return dict(
        a_ip=None, a_port=0, b_ip=None, b_port=0, proto=0,
        first_ts=None, last_ts=None,
        pkt_count=0, bytes=0,
        sum_len=0.0, sumsq_len=0.0, min_len=float("inf"), max_len=float("-inf"),
        # 方向统计
        pkts_fwd=0, pkts_bwd=0, bytes_fwd=0, bytes_bwd=0,
        last_ts_fwd=None, last_ts_bwd=None,
        iat_fwd_sum=0.0, iat_fwd_sumsq=0.0, iat_fwd_cnt=0,
        iat_bwd_sum=0.0, iat_bwd_sumsq=0.0, iat_bwd_cnt=0,
        # 这里留一个 TCP 旗标计数位（与你原版保持一致）
        tcp_flag_count=0
    )


def _update_bucket(bkt: Dict, ts: float, length: int, direction: int,
                   a_ip: str, a_port: int, b_ip: str, b_port: int, proto: int, is_tcp: bool):
    bkt["a_ip"] = a_ip; bkt["a_port"] = a_port
    bkt["b_ip"] = b_ip; bkt["b_port"] = b_port
    bkt["proto"] = proto

    bkt["pkt_count"] += 1
    bkt["bytes"] += length
    bkt["sum_len"] += length
    bkt["sumsq_len"] += length * length
    if length < bkt["min_len"]: bkt["min_len"] = length
    if length > bkt["max_len"]: bkt["max_len"] = length

    if bkt["first_ts"] is None or ts < bkt["first_ts"]:
        bkt["first_ts"] = ts
    if bkt["last_ts"] is None or ts > bkt["last_ts"]:
        bkt["last_ts"] = ts

    if is_tcp:
        bkt["tcp_flag_count"] += 1

    if direction >= 0:
        bkt["pkts_fwd"] += 1; bkt["bytes_fwd"] += length
        if bkt["last_ts_fwd"] is not None:
            diff = ts - bkt["last_ts_fwd"]
            if diff >= 0:
                bkt["iat_fwd_sum"] += diff
                bkt["iat_fwd_sumsq"] += diff * diff
                bkt["iat_fwd_cnt"] += 1
        bkt["last_ts_fwd"] = ts
    else:
        bkt["pkts_bwd"] += 1; bkt["bytes_bwd"] += length
        if bkt["last_ts_bwd"] is not None:
            diff = ts - bkt["last_ts_bwd"]
            if diff >= 0:
                bkt["iat_bwd_sum"] += diff
                bkt["iat_bwd_sumsq"] += diff * diff
                bkt["iat_bwd_cnt"] += 1
        bkt["last_ts_bwd"] = ts


# ---------------- 核心：单文件提取 ----------------
def extract_features(pcap_path: str, output_csv: str,
                     packet_index: Optional[int] = None,
                     progress_cb=None) -> str:
    """
    - packet_index: 若提供，仅抽取该索引对应“单包所在会话”的一条记录；否则全文件聚合。
    - 输出 CSV：字段与你原版一致，额外增加 `pcap_file`（=file）以兼容下游分析模块。
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"pcap 不存在: {pcap_path}")

    flows: Dict[Tuple, Dict] = {}

    last_pct = -1
    idx = -1
    for ts, buf, pos, total in _iter_packets(pcap_path):
        idx += 1

        # 单包模式：只在目标索引处做处理，其余快速跳过并更新进度
        if packet_index is not None and idx != packet_index:
            if progress_cb and total:
                pct = int(min(99, max(0, pos * 100.0 / (total or 1))))
                if pct != last_pct:
                    last_pct = pct; progress_cb(pct)
            continue

        try:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
            # 仅处理 IP 帧
            if not isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                continue

            proto = int(ip.p if isinstance(ip, dpkt.ip.IP) else ip.nxt)
            src_ip = _ip_to_str(ip.src); dst_ip = _ip_to_str(ip.dst)

            src_port = dst_port = 0
            is_tcp = False
            if isinstance(ip.data, dpkt.tcp.TCP):
                t = ip.data; src_port = int(t.sport); dst_port = int(t.dport); is_tcp = True
            elif isinstance(ip.data, dpkt.udp.UDP):
                u = ip.data; src_port = int(u.sport); dst_port = int(u.dport)

            key, dir_flag = _canon_key(src_ip, src_port, dst_ip, dst_port, proto)
            if key not in flows:
                flows[key] = _new_bucket()

            length = int(len(ip))
            _update_bucket(
                flows[key], float(ts), length, dir_flag,
                key[0], key[1], key[2], key[3], key[4], is_tcp
            )

        except Exception:
            # 坏帧忽略
            pass

        if progress_cb and total:
            pct = int(min(99, max(0, pos * 100.0 / (total or 1))))
            if pct != last_pct:
                last_pct = pct; progress_cb(pct)

        if packet_index is not None and idx == packet_index:
            # 单包模式：目标索引处理完即退出
            break

    # 汇总
    rows: List[Dict] = []
    base_name = os.path.basename(pcap_path)
    for key, b in flows.items():
        cnt = b["pkt_count"]
        duration = 0.0
        if (b["first_ts"] is not None) and (b["last_ts"] is not None):
            duration = max(0.0, float(b["last_ts"]) - float(b["first_ts"]))

        mean_len = (b["sum_len"] / cnt) if cnt > 0 else 0.0
        var_len = max(0.0, (b["sumsq_len"] / max(1, cnt)) - mean_len * mean_len)
        std_len = var_len ** 0.5

        def _stat(sum_, sumsq_, n_):
            if n_ <= 0:
                return 0.0, 0.0
            m = sum_ / n_
            v = max(0.0, (sumsq_ / n_) - m * m)
            return m, v ** 0.5

        iat_fwd_mean, iat_fwd_std = _stat(b["iat_fwd_sum"], b["iat_fwd_sumsq"], b["iat_fwd_cnt"])
        iat_bwd_mean, iat_bwd_std = _stat(b["iat_bwd_sum"], b["iat_bwd_sumsq"], b["iat_bwd_cnt"])

        pps = (cnt / duration) if duration > 0 else 0.0
        bps = (b["bytes"] * 8 / duration) if duration > 0 else 0.0
        pps_fwd = (b["pkts_fwd"] / duration) if duration > 0 else 0.0
        pps_bwd = (b["pkts_bwd"] / duration) if duration > 0 else 0.0
        fwd_bwd_ratio = (b["bytes_fwd"] / max(1.0, float(b["bytes_bwd"]))) if b["bytes_bwd"] > 0 else float("inf")

        rows.append({
            # 为兼容下游分析：两列等价
            "file": base_name,
            "pcap_file": base_name,

            "flow_id": f"{b['a_ip']}:{b['a_port']}-{b['b_ip']}:{b['b_port']}-{b['proto']}",
            "src_ip": b["a_ip"], "src_port": b["a_port"],
            "dst_ip": b["b_ip"], "dst_port": b["b_port"],
            "protocol": str(b["proto"]),

            "flow_duration": duration * 1e6,  # 微秒
            "pkt_count": cnt,
            "pkt_len_mean": round(mean_len, 3),
            "pkt_len_std": round(std_len, 3),
            "pkt_len_min": 0 if b["min_len"] == float("inf") else int(b["min_len"]),
            "pkt_len_max": 0 if b["max_len"] == float("-inf") else int(b["max_len"]),
            "inter_arrival_mean": round((iat_fwd_mean + iat_bwd_mean) / 2.0, 6),
            "inter_arrival_std": round((iat_fwd_std + iat_bwd_std) / 2.0, 6),
            "tcp_flag_count": b["tcp_flag_count"],

            # 方向增强
            "pkts_fwd": b["pkts_fwd"], "pkts_bwd": b["pkts_bwd"],
            "bytes_fwd": b["bytes_fwd"], "bytes_bwd": b["bytes_bwd"],
            "pps": round(pps, 3), "bps": round(bps, 3),
            "pps_fwd": round(pps_fwd, 3), "pps_bwd": round(pps_bwd, 3),
            "fwd_bwd_ratio": 0.0 if fwd_bwd_ratio == float("inf") else round(fwd_bwd_ratio, 6),
            "iat_fwd_mean": round(iat_fwd_mean, 6), "iat_fwd_std": round(iat_fwd_std, 6),
            "iat_bwd_mean": round(iat_bwd_mean, 6), "iat_bwd_std": round(iat_bwd_std, 6),
        })

    out_dir = os.path.dirname(os.path.abspath(output_csv)) or "."
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8")

    if progress_cb:
        progress_cb(100)
    return output_csv


# ---------------- 目录批量（并发） ----------------
def extract_features_dir(split_dir: str, out_dir: str, workers: int = 8, progress_cb=None) -> List[str]:
    """
    对目录下所有 .pcap/.pcapng 并发提取，输出到 out_dir，一文件一 csv（文件名加 _features.csv）
    """
    names = [n for n in os.listdir(split_dir) if n.lower().endswith((".pcap", ".pcapng"))]
    names.sort()
    files = [os.path.join(split_dir, n) for n in names]
    if not files:
        raise RuntimeError(f"目录下无 pcap: {split_dir}")

    os.makedirs(out_dir, exist_ok=True)
    results: List[str] = []

    def _one(f):
        base = os.path.splitext(os.path.basename(f))[0]
        csv = os.path.join(out_dir, f"{base}_features.csv")
        extract_features(f, csv, progress_cb=None)
        return csv

    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, f): f for f in files}
        for fut in as_completed(futs):
            try:
                csv = fut.result()
                results.append(csv)
            except Exception as e:
                print(f"[WARN] 特征失败: {futs[fut]} ({e})")
            finally:
                done += 1
                if progress_cb:
                    progress_cb(int(done / len(files) * 100))

    return results
