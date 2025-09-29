import os

from scapy.utils import PcapReader, PcapWriter


def split_pcap(input_path, output_dir, packets_per_file=200):
  """Utility helpers for splitting large PCAP files into smaller chunks."""

import argparse
import os
from typing import Optional

from scapy.utils import PcapReader, PcapWriter


def split_pcap(input_path, output_dir, packets_per_file=200):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part_num = 0
    packet_count = 0
    writer = None

    with PcapReader(input_path) as reader:
        for pkt in reader:
            if packet_count % packets_per_file == 0:
                if writer:
                    writer.close()
                part_num += 1
                part_path = os.path.join(output_dir, f"part_{part_num}.pcap")
                writer = PcapWriter(part_path, append=False, sync=True)
                print(f"[+] 创建文件: {part_path}")

            writer.write(pkt)
            packet_count += 1

    if writer:
        writer.close()

    print(f"[+] 完成分割: 共 {part_num} 个文件")


parts = split_pcap("data/Friday-WorkingHours.pcap", "data/split", packets_per_file=1000000)
if __name__ == "__main__":
    parts = split_pcap(
        "data/Friday-WorkingHours.pcap",
        "data/split",
        packets_per_file=1000000,
    )
    print(f"[+] 完成分割: 共 {part_num} 个文件")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Split a PCAP file into multiple smaller PCAP files."
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


def main(argv: Optional[list[str]] = None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        parser.error(f"输入文件不存在: {args.input}")

    output_dir = args.output or os.path.join(os.path.dirname(args.input), "split")
    split_pcap(args.input, output_dir, args.packets_per_file)


if __name__ == "__main__":
    main()