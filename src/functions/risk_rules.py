# -*- coding: utf-8 -*-
"""
基于简单启发式的规则评分（0-100）与“理由”生成。
不会抛错：缺列就跳过，不影响流程。
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# 可调阈值（按你数据规模可再微调）
DEFAULTS = dict(
    UDP_PPS_HI = 3000.0,     # UDP 高包速
    SCAN_TARGETS = 30,       # 同一 src 连接的不同 dst_port 或不同 dst_ip 的数量阈值
    BEACON_STD_MAX = 0.005,  # 到达间隔方差很小（单位：秒），视作周期性
    BEACON_MIN_PKTS = 20,    # 需要一定包数才算有意义
    EXFIL_BPS_RATIO = 0.8,   # 近似：bps 很高 + （若有 up/down 列）上行比例高
    SYN_HEAVY = 15,          # SYN 明显多于 ACK 的简易阈
)

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)

def _exists(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def score_rules(df: pd.DataFrame, params: Dict=None) -> Tuple[pd.Series, pd.Series]:
    """
    返回 (rules_score[0-100], reasons[str])
    """
    if params is None:
        params = DEFAULTS

    n = len(df)
    score = np.zeros(n, dtype=np.float32)
    reasons: List[List[str]] = [[] for _ in range(n)]

    # 统一需要的列
    col = lambda name: _to_float(df[name]) if name in df.columns else np.zeros(n, dtype=np.float32)
    proto = df["protocol"].astype(str) if "protocol" in df.columns else pd.Series([""]*n)

    pps = col("pps")
    bps = col("bps")
    pkt_count = col("pkt_count")
    inter_mean = col("inter_arrival_mean")
    inter_std  = col("inter_arrival_std")
    tcp_flags_count = col("tcp_flag_count")

    # 1) UDP Flood：UDP 且包速高
    udp_mask = proto.isin(["17", "udp", "UDP", "Udp"])
    flood_mask = (udp_mask.values) & (pps > params["UDP_PPS_HI"])
    score[flood_mask] += 40
    for i in np.where(flood_mask)[0]:
        reasons[i].append("高包速UDP")

    # 2) 扫描/探测：同一个 src_ip 发起到很多不同端口/不同 IP 的连接（以当前 df 为限定）
    if _exists(df, ["src_ip", "dst_port"]):
        port_cnt = df.groupby("src_ip")["dst_port"].transform("nunique").astype(np.float32)
        scan_mask = port_cnt.values >= params["SCAN_TARGETS"]
        score[scan_mask] += 30
        for i in np.where(scan_mask)[0]:
            reasons[i].append("端口扫描/探测")

    if _exists(df, ["src_ip", "dst_ip"]):
        ip_cnt = df.groupby("src_ip")["dst_ip"].transform("nunique").astype(np.float32)
        sweep_mask = ip_cnt.values >= params["SCAN_TARGETS"]
        score[sweep_mask] += 20
        for i in np.where(sweep_mask)[0]:
            reasons[i].append("IP段扫描/探测")

    # 3) Beacon：到达间隔稳定且包数达阈值
    if "inter_arrival_std" in df.columns:
        beacon_mask = (inter_std <= params["BEACON_STD_MAX"]) & (pkt_count >= params["BEACON_MIN_PKTS"])
        score[beacon_mask] += 30
        for i in np.where(beacon_mask)[0]:
            reasons[i].append("周期心跳(Beacon)")

    # 4) 外泄：bps 高 + （如果存在上/下行）上行占比高
    # 你当前没有 up/down 列，这里先用 bps 高做近似；若将来加上 bytes_up/bytes_down 再增强
    exfil_mask = (bps >= np.percentile(bps[bps>0], 90) if (bps>0).any() else (bps > 0))
    score[exfil_mask] += 20
    for i in np.where(exfil_mask)[0]:
        reasons[i].append("大带宽传输(疑似外泄)")

    # 5) SYN 异常（近似）：tcp_flag_count 不为 0 且超过阈值（如果你后续把各 flag 单独计数，这里可更精准）
    synheavy_mask = (tcp_flags_count >= params["SYN_HEAVY"])
    score[synheavy_mask] += 20
    for i in np.where(synheavy_mask)[0]:
        reasons[i].append("SYN 异常")

    # 归一 & 文本
    score = np.clip(score, 0, 100).astype(np.float32)
    reason_str = pd.Series(["; ".join(r) for r in reasons], index=df.index)
    return pd.Series(score, index=df.index, name="rules_score"), reason_str.rename("rules_reasons")
