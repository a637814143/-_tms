from scapy.all import PcapReader, PcapWriter
import os

def split_pcap(input_path, output_dir, packets_per_file=200):
    """
    将 pcap 文件分割为多个小文件
    :param input_path: 原始 pcap 路径
    :param output_dir: 输出目录
    :param packets_per_file: 每个小文件的包数量
    """
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
