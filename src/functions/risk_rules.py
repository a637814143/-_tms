# -*- coding: utf-8 -*-
"""
基于简单启发式的规则评分（0-100）与“理由”生成。
不会抛错：缺列就跳过，不影响流程。
"""
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Tuple, List, Dict, Any

try:  # 配置是可选依赖，缺失时保持默认行为
    from src.configuration import load_config
except Exception:  # pragma: no cover - 运行环境最小化时忽略配置
    load_config = None  # type: ignore

# 可调阈值（按你数据规模可再微调）
DEFAULTS = dict(
    UDP_PPS_HI = 3000.0,     # UDP 高包速
    SCAN_TARGETS = 30,       # 同一 src 连接的不同 dst_port 或不同 dst_ip 的数量阈值
    BEACON_STD_MAX = 0.005,  # 到达间隔方差很小（单位：秒），视作周期性
    BEACON_MIN_PKTS = 20,    # 需要一定包数才算有意义
    EXFIL_BPS_RATIO = 0.8,   # bps 的经验百分位阈值（兼容旧逻辑）
    EXFIL_UP_RATIO = 0.8,    # 上行/总流量占比异常阈值
    EXFIL_STRICT_UP_RATIO = 0.9,  # 上行占比极高时的强烈外泄提示
    SYN_HEAVY = 50,          # SYN 明显多于 ACK 的触发计数
    SYN_HEAVY_RATIO = 3.0,   # SYN/ACK 比例阈值
    SHORT_FLOW_SEC = 3.0,        # “很短”的流，单位秒
    SMALL_FLOW_BYTES = 800.0,    # 总字节很小
    HTTP_BRUTE_MIN = 20,         # 同一 src→dst:port 的短小流次数
    SLOWLORIS_MIN_DURATION = 60.0, # 慢速连接最小时长（秒）
    SLOWLORIS_MAX_PPS = 5.0,     # 慢速连接的最大包速
    HTTP_LONG_FLOW_SEC = 60.0,   # HTTP 会话判定“长连接”的阈值（秒）
    SLOWLORIS_STRICT_DURATION = 90.0,  # Slowloris 强判定持续时间阈值
    SLOWLORIS_STRICT_PPS = 1.0,        # Slowloris 强判定包速阈值
    SLOWLORIS_MAX_PKT_MEAN = 200.0,    # Slowloris 强判定平均包长阈值
    ONEWAY_MIN_PKTS = 50.0,      # 单向流的最小总包数
    ONEWAY_RATIO = 0.9,          # 单向占比阈值
    DNS_TUNNEL_MIN_DURATION = 30.0,
    DNS_TUNNEL_MIN_PPS = 50.0,
    ICMP_TUNNEL_MIN_PPS = 20.0,
    RATE_SPIKE_ABS = 1000.0, # 速率突增阈值（bytes/s）
    SMALL_PKT_MEAN = 50.0,   # 小包均值阈值（字节）
    SMALL_PKT_PPS = 100.0,   # 小包高频阈值（包/秒）
    ICMP_BPS_HI = 50000.0,   # ICMP 流量速率异常阈值
    TLS_DURATION_MAX = 1.0,  # TLS 握手极短持续时间阈值
    PORT_SCAN_UNIQUE_PORTS = 10,  # 同一 src 访问端口数量阈值
    BPS_SIGMA_MULT = 3.0,    # 动态带宽阈值：均值+N*std
    DDOS_PKT_RATE = 1000.0,  # DDoS：高包速阈值
    DDOS_BYTE_RATE = 1_000_000.0,  # DDoS：高字节速率阈值
    SLOWLORIS_DURATION = 60.0,     # Slowloris 持续时间阈值（秒）
    SLOWLORIS_BPS_MAX = 100.0,     # Slowloris 低速阈值（bytes/s）
    IP_SPOOF_SRC_VARIETY = 10,     # 同一目标的来源 IP 数量
    ARP_PPS_HI = 100.0,            # ARP 流量的包速阈值
    DNS_REQ_BYTES_MAX = 100.0,     # DNS 请求字节阈值
    DNS_RESP_BYTES_MIN = 1000.0,   # DNS 响应字节阈值
    FTP_LARGE_TRANSFER = 50000.0,  # FTP 大文件阈值
    UNIQUE_DST_PORTS = 10,         # 行内 unique_dst_ports 阈值
    UNIQUE_DST_IPS = 10,           # 行内 unique_dst_ips 阈值
    TRAFFIC_SPIKE_BPS = 10000.0,   # 突发流量速率阈值
    TRAFFIC_SPIKE_DURATION = 5.0,  # 突发流量持续时间阈值
    UNUSUAL_PROTO_PPS = 100.0,     # 非 TCP/UDP 协议高频阈值
    SHORT_SESSION_DURATION = 1.0,  # 过短会话
    LONG_SESSION_DURATION = 3600.0,# 过长会话
    HIGH_BANDWIDTH_ABS = 100_000.0,     # 持续高带宽绝对阈值（bytes/s）
    HIGH_BANDWIDTH_DURATION = 10.0,     # 持续高带宽持续时间（秒）
    HIGH_BANDWIDTH_PERCENTILE = 95.0,   # 持续高带宽百分位参考
)

# 规则档位预设：baseline / aggressive
DEFAULT_RULE_PROFILE = "baseline"

PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "baseline": {
        # 规则总分 >= trigger_threshold 才认为“规则强烈命中”
        "trigger_threshold": 60.0,
        # 模型权重更高，规则只做辅助
        "model_weight": 0.7,
        "rule_weight": 0.3,
        # 融合分 >= fusion_threshold 才判异常
        "fusion_threshold": 0.5,
    },
    "aggressive": {
        # 规则总分 >= 25 就认为“规则强烈命中”
        "trigger_threshold": 25.0,
        # 规则权重更高，对恶意流量更敏感
        "model_weight": 0.3,
        "rule_weight": 0.7,
        # 判异常的阈值更低
        "fusion_threshold": 0.2,
    },
}

# 规则判定异常时的默认触发阈值（以 baseline 为默认）
DEFAULT_TRIGGER_THRESHOLD = PROFILE_PRESETS[DEFAULT_RULE_PROFILE]["trigger_threshold"]

# 规则触发的兜底高敏感阈值（UI/导出使用）
RULE_TRIGGER_THRESHOLD = 25.0

# 模型与规则融合的默认权重与阈值（以 baseline 为默认）
DEFAULT_MODEL_WEIGHT = PROFILE_PRESETS[DEFAULT_RULE_PROFILE]["model_weight"]
DEFAULT_RULE_WEIGHT = PROFILE_PRESETS[DEFAULT_RULE_PROFILE]["rule_weight"]
DEFAULT_FUSION_THRESHOLD = PROFILE_PRESETS[DEFAULT_RULE_PROFILE]["fusion_threshold"]


def _load_rules_config() -> Dict[str, Any]:
    if load_config is None:  # pragma: no cover - 最小环境
        return {}
    try:
        config = load_config() or {}
    except Exception:  # pragma: no cover - 配置解析失败
        return {}
    rules_cfg = config.get("rules") if isinstance(config, dict) else {}
    return rules_cfg if isinstance(rules_cfg, dict) else {}


def get_rule_settings(profile: Optional[str] = None) -> Dict[str, Any]:
    """从全局配置中解析规则参数、融合权重及触发阈值。"""

    rules_cfg = _load_rules_config()

    # 1) 确定当前 profile：UI 显式传入 > 配置 active_profile > 默认 baseline
    selected_profile = profile or rules_cfg.get("active_profile") or DEFAULT_RULE_PROFILE
    if not isinstance(selected_profile, str):
        selected_profile = DEFAULT_RULE_PROFILE
    selected_profile = selected_profile.strip().lower() or DEFAULT_RULE_PROFILE

    # 2) YAML 中若配置了 profiles，优先使用；否则用内置预设
    profiles_cfg_raw = rules_cfg.get("profiles")
    if isinstance(profiles_cfg_raw, dict) and profiles_cfg_raw:
        normalized_profiles: Dict[str, Dict[str, float]] = {}
        for name, payload in profiles_cfg_raw.items():
            if not isinstance(name, str):
                continue
            normalized_name = name.strip().lower()
            if not normalized_name:
                continue
            if isinstance(payload, dict):
                normalized_profiles[normalized_name] = payload
        profiles_cfg = normalized_profiles or PROFILE_PRESETS
    else:
        profiles_cfg = PROFILE_PRESETS

    profile_params: Dict[str, Any] = {}
    if selected_profile in profiles_cfg:
        candidate = profiles_cfg.get(selected_profile, {})
        if isinstance(candidate, dict):
            profile_params = candidate
    else:
        selected_profile = DEFAULT_RULE_PROFILE
        fallback = profiles_cfg.get(DEFAULT_RULE_PROFILE, {})
        if isinstance(fallback, dict):
            profile_params = fallback

    # 3) 规则阈值：拷贝 DEFAULTS，再用 profile 中同名键覆盖
    params: Dict[str, float] = dict(DEFAULTS)
    for key, value in profile_params.items():
        if key in params:
            try:
                params[key] = float(value)
            except (TypeError, ValueError):
                continue

    # 4) 融合相关参数：预设 + YAML 覆盖
    fusion_cfg = rules_cfg.get("fusion") if isinstance(rules_cfg.get("fusion"), dict) else {}

    preset_trigger = profile_params.get("trigger_threshold") or PROFILE_PRESETS.get(
        selected_profile,
        {},
    ).get("trigger_threshold", DEFAULT_TRIGGER_THRESHOLD)

    preset_model_w = profile_params.get("model_weight") or PROFILE_PRESETS.get(
        selected_profile,
        {},
    ).get("model_weight", DEFAULT_MODEL_WEIGHT)

    preset_rule_w = profile_params.get("rule_weight") or PROFILE_PRESETS.get(
        selected_profile,
        {},
    ).get("rule_weight", DEFAULT_RULE_WEIGHT)

    preset_fusion_th = profile_params.get("fusion_threshold") or PROFILE_PRESETS.get(
        selected_profile,
        {},
    ).get("fusion_threshold", DEFAULT_FUSION_THRESHOLD)

    trigger_threshold = rules_cfg.get("trigger_threshold", preset_trigger)
    model_weight = fusion_cfg.get("model_weight", rules_cfg.get("model_weight", preset_model_w))
    rule_weight = fusion_cfg.get("rule_weight", rules_cfg.get("rule_weight", preset_rule_w))
    fusion_threshold = fusion_cfg.get(
        "decision_threshold",
        rules_cfg.get("fusion_threshold", preset_fusion_th),
    )

    try:
        trigger_threshold = float(trigger_threshold)
    except (TypeError, ValueError):
        trigger_threshold = float(DEFAULT_TRIGGER_THRESHOLD)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    settings = {
        "params": params,
        "trigger_threshold": trigger_threshold,
        "model_weight": _as_float(model_weight, DEFAULT_MODEL_WEIGHT),
        "rule_weight": _as_float(rule_weight, DEFAULT_RULE_WEIGHT),
        "fusion_threshold": _as_float(fusion_threshold, DEFAULT_FUSION_THRESHOLD),
        "profile": selected_profile,
    }
    return settings


def _refresh_module_defaults() -> None:
    settings = get_rule_settings()
    global DEFAULT_TRIGGER_THRESHOLD, DEFAULT_MODEL_WEIGHT, DEFAULT_RULE_WEIGHT, DEFAULT_FUSION_THRESHOLD
    try:
        DEFAULT_TRIGGER_THRESHOLD = float(settings.get("trigger_threshold", DEFAULT_TRIGGER_THRESHOLD))
    except (TypeError, ValueError):  # pragma: no cover - 兜底
        DEFAULT_TRIGGER_THRESHOLD = 40.0
    try:
        DEFAULT_MODEL_WEIGHT = float(settings.get("model_weight", DEFAULT_MODEL_WEIGHT))
    except (TypeError, ValueError):  # pragma: no cover - 兜底
        DEFAULT_MODEL_WEIGHT = 0.3
    try:
        DEFAULT_RULE_WEIGHT = float(settings.get("rule_weight", DEFAULT_RULE_WEIGHT))
    except (TypeError, ValueError):  # pragma: no cover
        DEFAULT_RULE_WEIGHT = 0.7
    try:
        DEFAULT_FUSION_THRESHOLD = float(settings.get("fusion_threshold", DEFAULT_FUSION_THRESHOLD))
    except (TypeError, ValueError):  # pragma: no cover
        DEFAULT_FUSION_THRESHOLD = 0.2


try:  # 模块导入时同步一次配置（如果配置可用）
    _refresh_module_defaults()
except Exception:  # pragma: no cover - 容错
    pass

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)

def _exists(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)

def score_rules(
    df: pd.DataFrame,
    params: Optional[Dict[str, float]] = None,
    *,
    profile: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    返回 (rules_score[0-100], reasons[str])
    """
    if params is None:
        params = get_rule_settings(profile).get("params", DEFAULTS)

    n = len(df)
    score = np.zeros(n, dtype=np.float32)
    reasons: List[List[str]] = [[] for _ in range(n)]

    # 统一需要的列（兼容多种命名）
    col = (
        lambda name: _to_float(df[name])
        if name in df.columns
        else pd.Series(np.zeros(n, dtype=np.float32), index=df.index)
    )
    proto = (
        df["protocol"].astype(str)
        if "protocol" in df.columns
        else pd.Series([""] * n, index=df.index)
    )
    proto_upper = proto.str.upper()

    # 目标端口（字符串形式，缺失时填充空串）
    dst_port_series = (
        df["dst_port"].astype(str)
        if "dst_port" in df.columns
        else pd.Series([""] * n, index=df.index, dtype=str)
    )
    http_service_ports = {"80", "443", "8080", "8000", "8443"}
    http_port_mask_global = dst_port_series.isin(http_service_ports)
    tcp_http_mask = proto_upper.eq("TCP") & http_port_mask_global

    slowloris_strict_mask = pd.Series(np.zeros(n, dtype=bool), index=df.index)

    # 会话时长
    flow_duration_raw = col("flow_duration") + col("Flow Duration")
    flow_duration_vals = flow_duration_raw.to_numpy()
    if flow_duration_vals.size and np.nanmax(np.abs(flow_duration_vals)) > 1_000_000:
        flow_duration = pd.Series(
            flow_duration_vals / 1_000_000.0, index=df.index, dtype=np.float32
        )
    else:
        flow_duration = flow_duration_raw

    # 包速：优先用 Flow Packets/s 或 flow_pkts_per_s
    pps = (
        col("pps")
        + col("flow_pkts_per_s")
        + col("Flow Packets/s")
        + col("Fwd Packets/s")
        + col("Bwd Packets/s")
    )

    # 字节速率：优先用 Flow Bytes/s 或 flow_byts_per_s
    bps = col("bps") + col("flow_byts_per_s") + col("Flow Bytes/s")

    # 总包数：前向 + 后向
    pkt_count = (
        col("pkt_count")
        + col("Total Fwd Packets")
        + col("Tot Fwd Pkts")
        + col("Total Backward Packets")
        + col("Tot Bwd Pkts")
    )

    # IAT 统计：flow 级别
    inter_mean = col("inter_arrival_mean") + col("Flow IAT Mean")
    inter_std = col("inter_arrival_std") + col("Flow IAT Std")

    # TCP 标志计数：如果没有聚合列，就把各 flag 加起来
    tcp_flags_count = (
        col("tcp_flag_count")
        + col("FIN Flag Count")
        + col("SYN Flag Count")
        + col("RST Flag Count")
        + col("PSH Flag Count")
        + col("ACK Flag Count")
        + col("URG Flag Count")
    )

    # 总字节/上行字节（如果 CSV 没有 flow_bytes/flow_up_bytes，这里仍可能为 0）
    flow_bytes = col("flow_bytes")
    flow_up_bytes = col("flow_up_bytes")

    # 统一速率、平均包长
    flow_rate = col("flow_byts_per_s") if "flow_byts_per_s" in df.columns else bps
    flow_pkts_per_s = col("flow_pkts_per_s") if "flow_pkts_per_s" in df.columns else pps
    packet_len_mean = col("packet_length_mean") + col("Packet Length Mean")
    unique_ports = col("unique_ports_accessed")
    unique_dst_ports = col("unique_dst_ports")
    unique_dst_ips = col("unique_dst_ips")
    unique_src_ips = col("unique_src_ips")
    total_len_fwd_pkts = col("total_len_fwd_pkts")
    total_len_bwd_pkts = col("total_len_bwd_pkts")

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
    if (bps > 0).any():
        percentile_param = params["EXFIL_BPS_RATIO"]
        percentile_value = percentile_param * 100 if percentile_param <= 1 else percentile_param
        valid_bps = bps[bps > 0].to_numpy()
        percentile_threshold = np.percentile(valid_bps, percentile_value)
        exfil_mask = bps >= percentile_threshold
    else:
        exfil_mask = bps > 0
    for i in np.where(exfil_mask)[0]:
        score[i] += 20
        reasons[i].append("大带宽传输(疑似外泄)")

    if _exists(df, ["flow_bytes", "flow_up_bytes"]):
        total = flow_bytes.to_numpy()
        up_bytes = flow_up_bytes.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(up_bytes, total, out=np.zeros(n, dtype=np.float32), where=total > 0)
        exfil_up_mask = (total > 0) & (ratio >= params["EXFIL_UP_RATIO"])
        for i in np.where(exfil_up_mask)[0]:
            score[i] += 15
            reasons[i].append("上行占比异常(疑似外泄)")

    # 动态检测异常高带宽
    valid_rate = flow_rate[flow_rate > 0]
    percentile_threshold = 0.0
    if len(valid_rate) > 0:
        percentile_threshold = np.percentile(
            valid_rate.to_numpy(), params.get("HIGH_BANDWIDTH_PERCENTILE", 95.0)
        )

    dynamic_threshold = 0.0
    if len(valid_rate) > 1:
        mean_rate = valid_rate.mean()
        std_rate = valid_rate.std(ddof=0)
        dynamic_threshold = mean_rate + params["BPS_SIGMA_MULT"] * std_rate

    absolute_threshold = params.get("HIGH_BANDWIDTH_ABS", 100_000.0)
    effective_threshold = max(absolute_threshold, percentile_threshold, dynamic_threshold)
    duration_threshold = params.get("HIGH_BANDWIDTH_DURATION", 10.0)

    high_bandwidth_mask = (flow_rate >= effective_threshold) & (flow_duration >= duration_threshold)
    for i in np.where(high_bandwidth_mask.to_numpy())[0]:
        score[i] += 10
        reasons[i].append("持续高带宽传输")

    # 5) SYN 异常（近似）：SYN/ACK 比例显著失衡
    syn_count = col("SYN Flag Count") + col("Syn Flag Count") + col("syn_flag_count")
    ack_count = col("ACK Flag Count") + col("Ack Flag Count") + col("ack_flag_count")
    with np.errstate(divide="ignore", invalid="ignore"):
        syn_ack_ratio = np.divide(
            syn_count.to_numpy(),
            np.maximum(ack_count.to_numpy(), 1.0),
            out=np.ones(n, dtype=np.float32),
        )
    syn_ratio_series = pd.Series(syn_ack_ratio, index=df.index, dtype=np.float32)
    synheavy_mask = (syn_count >= params["SYN_HEAVY"]) & (
        syn_ratio_series >= params.get("SYN_HEAVY_RATIO", 3.0)
    )
    for i in np.where(synheavy_mask.to_numpy())[0]:
        score[i] += 20
        reasons[i].append("SYN/ACK 比例异常")

    # === 6) HTTP 短小高频访问：近似 Web 扫描 / 爆破 / 批量注入 ===
    # 不看 HTTP 内容，只用统计特征：同一 src_ip 对同一 dst_ip:dst_port 发起很多「很短 + 很小」的流
    if _exists(df, ["src_ip", "dst_ip", "dst_port"]):
        total_fwd = col("total_len_fwd_pkts")
        total_bwd = col("total_len_bwd_pkts")
        total_bytes = total_fwd + total_bwd

        total_bytes_arr = total_bytes.to_numpy()
        approx_total = bps.to_numpy() * np.maximum(flow_duration.to_numpy(), 1.0)
        zero_mask = total_bytes_arr == 0
        total_bytes_arr[zero_mask] = approx_total[zero_mask]
        total_bytes = pd.Series(total_bytes_arr, index=df.index, dtype=np.float32)

        short_small = (
            (flow_duration > 0)
            & (flow_duration <= params["SHORT_FLOW_SEC"])
            & (total_bytes <= params["SMALL_FLOW_BYTES"])
        )

        http_port_mask = http_port_mask_global

        if http_port_mask.any():
            grp = df[http_port_mask].groupby(["src_ip", "dst_ip", "dst_port"], dropna=False)
            http_small_counts = grp["src_ip"].transform("size").astype(np.float32)
            http_small_counts_full = pd.Series(np.zeros(n, dtype=np.float32), index=df.index)
            http_small_counts_full.loc[http_port_mask] = http_small_counts.values

            brute_mask = (
                short_small
                & http_port_mask
                & (http_small_counts_full >= params["HTTP_BRUTE_MIN"])
            )

            brute_idx = np.where(brute_mask.to_numpy())[0]
            score[brute_idx] += 25
            for i in brute_idx:
                reasons[i].append("HTTP短小高频访问(疑似爆破/批量注入/扫描)")

    # === 7) 慢速 DoS / Slowloris 近似：长连接 + 包速极低 ===
    http_long_threshold = params.get(
        "HTTP_LONG_FLOW_SEC", params.get("SLOWLORIS_MIN_DURATION", 60.0)
    )
    slow_base_mask = (
        tcp_http_mask
        & (flow_duration >= http_long_threshold)
        & (pps > 0)
        & (pps <= params["SLOWLORIS_MAX_PPS"])
    )

    strict_duration = max(
        params.get("SLOWLORIS_STRICT_DURATION", http_long_threshold),
        http_long_threshold,
    )
    strict_pps = params.get("SLOWLORIS_STRICT_PPS", 1.0)
    max_pkt_mean = params.get("SLOWLORIS_MAX_PKT_MEAN", 200.0)

    slowloris_strict_mask = (
        tcp_http_mask
        & (flow_duration >= strict_duration)
        & (pps > 0)
        & (pps <= strict_pps)
        & (packet_len_mean > 0)
        & (packet_len_mean <= max_pkt_mean)
    )

    strict_idx = np.where(slowloris_strict_mask.to_numpy())[0]
    if strict_idx.size:
        score[strict_idx] += 20
        for i in strict_idx:
            reasons[i].append("长时间低包速连接(疑似Slowloris/慢速DoS)")
            reasons[i].append("疑似Slowloris攻击")

    # === 8) 单向流异常：大量包几乎只在一个方向 ===
    fwd_pkts = col("Total Fwd Packets") + col("total_fwd_pkts")
    bwd_pkts = col("Total Backward Packets") + col("total_bwd_pkts")

    total_pkts = fwd_pkts + bwd_pkts
    with np.errstate(divide="ignore", invalid="ignore"):
        major_ratio = np.where(
            total_pkts.to_numpy() > 0,
            np.maximum(fwd_pkts.to_numpy(), bwd_pkts.to_numpy())
            / np.maximum(total_pkts.to_numpy(), 1.0),
            0.0,
        )
    major_ratio_series = pd.Series(major_ratio, index=df.index, dtype=np.float32)

    oneway_mask = (total_pkts >= params["ONEWAY_MIN_PKTS"]) & (
        major_ratio_series >= params["ONEWAY_RATIO"]
    )
    oneway_idx = np.where(oneway_mask.to_numpy())[0]
    score[oneway_idx] += 50
    for i in oneway_idx:
        reasons[i].append("单向流量占比异常(可能为扫描/单向攻击/隧道)")

    # === 9) DNS 隧道近似：长时间 + 高 PPS 的 53 端口 UDP ===
    if "dst_port" in df.columns:
        dns_mask = df["dst_port"].astype(str) == "53"
        dns_long = dns_mask & (flow_duration >= params["DNS_TUNNEL_MIN_DURATION"]) & (
            pps >= params["DNS_TUNNEL_MIN_PPS"]
        )
        dns_idx = np.where(dns_long.to_numpy())[0]
        score[dns_idx] += 25
        for i in dns_idx:
            reasons[i].append("DNS长时间高包速(疑似DNS隧道)")

    # === 10) ICMP 异常 / 潜在隧道：ICMP + 高 PPS ===
    icmp_mask = proto.str.lower().isin(["1", "icmp"])
    icmp_susp = icmp_mask & (pps >= params["ICMP_TUNNEL_MIN_PPS"])
    icmp_idx = np.where(icmp_susp.to_numpy())[0]
    score[icmp_idx] += 20
    for i in icmp_idx:
        reasons[i].append("ICMP高包速(疑似ICMP隧道/探测)")

    # 11) 流量速率突增（DoS/Bot 异常指征）
    if "src_ip" in df.columns:
        previous_rate = flow_rate.groupby(df["src_ip"]).shift(1)
    else:
        previous_rate = flow_rate.shift(1)
    previous_rate = previous_rate.fillna(flow_rate)
    flow_rate_arr = flow_rate.to_numpy()
    spike_mask = np.abs(flow_rate_arr - previous_rate.to_numpy()) > params["RATE_SPIKE_ABS"]
    for i in np.where(spike_mask)[0]:
        score[i] += 20
        reasons[i].append("流量速率突增")

    # 12) 小包泛洪
    small_packet_mask = (packet_len_mean > 0) & (packet_len_mean < params["SMALL_PKT_MEAN"]) & (flow_pkts_per_s > params["SMALL_PKT_PPS"])
    for i in np.where(small_packet_mask)[0]:
        score[i] += 25
        reasons[i].append("小包高频(疑似洪泛)")

    # 13) 协议分布异常（ICMP 突出）
    icmp_mask = proto_upper.eq("ICMP") & (flow_rate > params["ICMP_BPS_HI"])
    for i in np.where(icmp_mask.values)[0]:
        score[i] += 30
        reasons[i].append("异常 ICMP 流量")

    # 14) TLS 握手异常
    tls_mask = proto_upper.eq("TLS") & (flow_duration > 0) & (flow_duration < params["TLS_DURATION_MAX"])
    for i in np.where(tls_mask.values)[0]:
        score[i] += 30
        reasons[i].append("TLS 握手持续时间过短")

    # 15) 端口扫描 - 独立字段支持
    if "unique_ports_accessed" in df.columns:
        port_scan_mask = unique_ports >= params["PORT_SCAN_UNIQUE_PORTS"]
        for i in np.where(port_scan_mask)[0]:
            score[i] += 35
            reasons[i].append("端口扫描(统计)")

    # 16) DDoS：包速与字节速率均极高
    ddos_mask = (flow_pkts_per_s >= params["DDOS_PKT_RATE"]) & (flow_rate >= params["DDOS_BYTE_RATE"])
    for i in np.where(ddos_mask)[0]:
        score[i] += 50
        reasons[i].append("疑似DDoS攻击(高包速/高带宽)")

    # 17) Slowloris：长连接且速率极低
    slowloris_mask = (flow_duration >= params["SLOWLORIS_DURATION"]) & (flow_rate <= params["SLOWLORIS_BPS_MAX"])
    for i in np.where(slowloris_mask)[0]:
        score[i] += 30
        reasons[i].append("疑似Slowloris攻击")

    # 18) IP 欺骗：同一目标对应来源 IP 非常多
    if "unique_src_ips" in df.columns:
        spoof_mask = unique_src_ips >= params["IP_SPOOF_SRC_VARIETY"]
        for i in np.where(spoof_mask)[0]:
            score[i] += 40
            reasons[i].append("疑似IP欺骗(来源IP异常多)")

    # 19) ARP 欺骗：ARP 协议且包速高
    arp_mask = proto_upper.eq("ARP") & (flow_pkts_per_s >= params["ARP_PPS_HI"])
    for i in np.where(arp_mask)[0]:
        score[i] += 50
        reasons[i].append("疑似ARP欺骗攻击")

    # 20) DNS 放大：请求小但响应巨大
    dns_mask = proto_upper.eq("DNS")
    dns_amp_mask = dns_mask & (total_len_fwd_pkts <= params["DNS_REQ_BYTES_MAX"]) & (total_len_bwd_pkts >= params["DNS_RESP_BYTES_MIN"])
    for i in np.where(dns_amp_mask)[0]:
        score[i] += 40
        reasons[i].append("疑似DNS放大攻击")

    # 21) 额外的外泄提示：上行占比极高
    if _exists(df, ["flow_bytes", "flow_up_bytes"]):
        total = flow_bytes.to_numpy()
        up_bytes = flow_up_bytes.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            strict_ratio = np.divide(up_bytes, total, out=np.zeros(n, dtype=np.float32), where=total > 0)
        strict_mask = (total > 0) & (strict_ratio >= params["EXFIL_STRICT_UP_RATIO"])
        for i in np.where(strict_mask)[0]:
            score[i] += 30
            reasons[i].append("上行占比极高(强烈外泄信号)")

    # 22) 大文件 FTP 传输
    ftp_mask = proto_upper.eq("FTP") & (total_len_fwd_pkts + total_len_bwd_pkts >= params["FTP_LARGE_TRANSFER"])
    for i in np.where(ftp_mask)[0]:
        score[i] += 60
        reasons[i].append("FTP大文件传输(疑似外泄)")

    # 23) 行内端口扫描/IP 扫描字段支持
    if "unique_dst_ports" in df.columns:
        dst_port_mask = unique_dst_ports >= params["UNIQUE_DST_PORTS"]
        for i in np.where(dst_port_mask)[0]:
            score[i] += 40
            reasons[i].append("疑似端口扫描(多端口)")

    if "unique_dst_ips" in df.columns:
        dst_ip_mask = unique_dst_ips >= params["UNIQUE_DST_IPS"]
        for i in np.where(dst_ip_mask)[0]:
            score[i] += 40
            reasons[i].append("疑似IP扫描(多目标)")

    # 24) 突发流量：短时高带宽
    spike_short_mask = (flow_rate >= params["TRAFFIC_SPIKE_BPS"]) & (flow_duration > 0) & (flow_duration <= params["TRAFFIC_SPIKE_DURATION"])
    for i in np.where(spike_short_mask)[0]:
        score[i] += 50
        reasons[i].append("短时流量突增")

    # 25) 协议异常：非常见协议高速率
    common_proto = {"TCP", "UDP"}
    unusual_proto_mask = ~proto_upper.isin(common_proto) & (flow_pkts_per_s >= params["UNUSUAL_PROTO_PPS"])
    for i in np.where(unusual_proto_mask)[0]:
        score[i] += 30
        reasons[i].append("非常见协议高频通信")

    # 26) 会话持续时间异常
    short_session_mask = (flow_duration > 0) & (flow_duration < params["SHORT_SESSION_DURATION"])
    for i in np.where(short_session_mask)[0]:
        score[i] += 20
        reasons[i].append("会话持续时间过短")

    http_long_threshold = params.get(
        "HTTP_LONG_FLOW_SEC", params.get("SLOWLORIS_MIN_DURATION", 60.0)
    )
    http_long_mask = tcp_http_mask & (flow_duration >= http_long_threshold)
    http_long_idx = np.where(http_long_mask.to_numpy())[0]
    for i in http_long_idx:
        score[i] += 10
        if bool(slow_base_mask.iloc[i]):
            reasons[i].append("HTTP会话持续时间较长(低包速)")
        else:
            reasons[i].append("HTTP会话持续时间过长")

    very_long_mask = (~http_long_mask) & (flow_duration >= params["LONG_SESSION_DURATION"])
    for i in np.where(very_long_mask.to_numpy())[0]:
        score[i] += 10
        reasons[i].append("会话持续时间过长")

    # 归一 & 文本
    score = np.clip(score, 0, 100).astype(np.float32)
    reason_str = pd.Series(["; ".join(r) for r in reasons], index=df.index)
    return pd.Series(score, index=df.index, name="rules_score"), reason_str.rename("rules_reasons")


def fuse_model_rule_votes(
    model_flags: Iterable[object],
    rule_scores: Optional[Iterable[object]],
    *,
    model_weight: float = DEFAULT_MODEL_WEIGHT,
    rule_weight: float = DEFAULT_RULE_WEIGHT,
    threshold: float = DEFAULT_FUSION_THRESHOLD,
    profile: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine模型预测标记与规则得分，返回融合分数与最终判定。"""

    if profile is not None:
        try:
            settings = get_rule_settings(profile)
        except Exception:
            settings = {}
        else:
            if isinstance(settings, dict):
                preset_model = settings.get("model_weight")
                preset_rule = settings.get("rule_weight")
                preset_threshold = settings.get("fusion_threshold")
                if preset_model is not None and model_weight == DEFAULT_MODEL_WEIGHT:
                    try:
                        model_weight = float(preset_model)
                    except (TypeError, ValueError):
                        pass
                if preset_rule is not None and rule_weight == DEFAULT_RULE_WEIGHT:
                    try:
                        rule_weight = float(preset_rule)
                    except (TypeError, ValueError):
                        pass
                if preset_threshold is not None and threshold == DEFAULT_FUSION_THRESHOLD:
                    try:
                        threshold = float(preset_threshold)
                    except (TypeError, ValueError):
                        pass

    model_arr = np.asarray(model_flags, dtype=np.float64).reshape(-1)
    if model_arr.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.astype(bool)

    model_arr = np.clip(model_arr, 0.0, 1.0)

    normalized_rules: np.ndarray
    active_rule_weight = float(rule_weight)

    if rule_scores is None:
        normalized_rules = np.zeros_like(model_arr)
        active_rule_weight = 0.0
    else:
        try:
            rule_arr = np.asarray(rule_scores, dtype=np.float64).reshape(-1)
        except Exception:
            rule_arr = None

        if rule_arr is None or rule_arr.size == 0:
            normalized_rules = np.zeros_like(model_arr)
            active_rule_weight = 0.0
        else:
            if rule_arr.size != model_arr.size:
                if rule_arr.size < model_arr.size:
                    padded = np.zeros_like(model_arr)
                    padded[: rule_arr.size] = rule_arr
                    rule_arr = padded
                else:
                    rule_arr = rule_arr[: model_arr.size]
            normalized_rules = np.clip(rule_arr / 100.0, 0.0, 1.0)

    total_weight = float(model_weight + active_rule_weight)
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        model_w = 1.0
        rule_w = 0.0
    else:
        model_w = float(model_weight) / total_weight
        rule_w = float(active_rule_weight) / total_weight

    fused_scores = model_w * model_arr + rule_w * normalized_rules
    fused_flags = fused_scores >= float(threshold)

    return fused_scores, fused_flags
