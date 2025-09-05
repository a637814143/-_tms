# src/functions/feature_extractor.py
import os
import dpkt
import socket
import pandas as pd

def _ip_to_str(raw):
    try:
        return socket.inet_ntoa(raw)
    except Exception:
        # 不是 IPv4 时兜底
        try:
            return socket.inet_ntop(socket.AF_INET6, raw)
        except Exception:
            return ""

def extract_features(pcap_path: str, output_csv: str, progress_cb=None) -> str:
    """
    从单个 pcap 文件提取“按包”的轻量特征，并保存为 CSV。
    字段：timestamp, src_ip, dst_ip, protocol, length, pcap_file
    说明：轻量 & 快速，便于无监督训练（IsolationForest）。
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"文件不存在: {pcap_path}")

    flows = []
    total = 0

    # 先统计包数，用于进度条
    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.NeedData):
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        for _ in reader:
            total += 1

    with open(pcap_path, "rb") as f:
        # 再次读取
        try:
            reader = dpkt.pcap.Reader(f)
        except (ValueError, dpkt.NeedData):
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)

        for i, (ts, buf) in enumerate(reader, 1):
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                # 只处理 IP 包（IPv4/IPv6）
                if not hasattr(ip, "p"):
                    continue

                proto = int(getattr(ip, "p", 0))
                src_raw = getattr(ip, "src", b"")
                dst_raw = getattr(ip, "dst", b"")
                src_ip = _ip_to_str(src_raw)
                dst_ip = _ip_to_str(dst_raw)
                length = len(ip)

                flows.append({
                    "timestamp": ts,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": proto,
                    "length": length,
                    "pcap_file": os.path.basename(pcap_path)
                })
            except Exception:
                # 坏包/非IP 包等，直接跳过
                pass

            if progress_cb and total > 0 and (i % 200 == 0 or i == total):
                progress_cb(int(i * 100 / total))

    if not flows:
        raise ValueError(f"{pcap_path} 未提取到任何有效数据")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df = pd.DataFrame(flows)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    if progress_cb:
        progress_cb(100)
    return output_csv

