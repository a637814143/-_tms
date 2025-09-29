"""PCAP 分割工具。"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from scapy.utils import PcapReader, PcapWriter


def split_pcap(input_path: str, output_dir: str, packets_per_file: int = 200) -> List[str]:
    """将 PCAP 文件按数据包数量拆分为多个小文件。"""
    if packets_per_file <= 0:
        raise ValueError("packets_per_file 必须为正整数")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part_paths: List[str] = []
    part_num = 0
    packet_count = 0
    writer: Optional[PcapWriter] = None

    with PcapReader(input_path) as reader:
        for pkt in reader:
            if packet_count % packets_per_file == 0:
                if writer:
                    writer.close()
                part_num += 1
                part_path = os.path.join(output_dir, f"part_{part_num:05d}.pcap")
                writer = PcapWriter(part_path, append=False, sync=True)
                part_paths.append(part_path)
                print(f"[+] 创建文件: {part_path}")

            if writer is None:
                raise RuntimeError("PcapWriter 未正确初始化")

            writer.write(pkt)
            packet_count += 1

    if writer:
        writer.close()

    print(f"[+] 完成分割: 共 {len(part_paths)} 个文件")
    return part_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a PCAP file into multiple smaller PCAP files.",
    )
    parser.add_argument("input", help="Path to the input PCAP/PCAPNG file")
    parser.add_argument(
        "-o",
        "--output",
        help="Directory to save the splitted files (default: <input_dir>/split)",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--packets-per-file",
        type=int,
        default=200,
        help="Number of packets per splitted file",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")

    output_dir = args.output or os.path.join(os.path.dirname(args.input), "split")
    part_paths = split_pcap(args.input, output_dir, args.packets_per_file)
    print(f"[+] 输出目录: {output_dir}")
    print(f"[+] 生成文件数: {len(part_paths)}")


if __name__ == "__main__":
    main()
