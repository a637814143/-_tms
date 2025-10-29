# src/functions/split.py
import os
from scapy.all import PcapReader, PcapWriter


def split_pcap(input_path, output_dir, packets_per_file=500000):
    """
    将 pcap 文件分割为多个小文件（流式读取，支持大文件）

    :param input_path: 原始 pcap 文件路径
    :param output_dir: 输出目录
    :param packets_per_file: 每个小文件的包数量（默认 50 万）
    :return: 生成的小文件路径列表
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part_num = 0
    packet_count = 0
    writer = None
    output_files = []

    try:
        with PcapReader(input_path) as reader:
            for pkt in reader:
                # 每 packets_per_file 个包新建一个文件
                if packet_count % packets_per_file == 0:
                    # 如果有上一个 writer，先关闭
                    if writer:
                        writer.close()

                    part_num += 1
                    part_path = os.path.join(output_dir, f"part_{part_num}.pcap")
                    writer = PcapWriter(part_path, append=False, sync=True)
                    output_files.append(part_path)
                    print(f"[+] 创建文件: {part_path}")

                writer.write(pkt)
                packet_count += 1

        if writer:
            writer.close()

    except Exception as e:
        raise RuntimeError(f"分割 pcap 文件时出错: {str(e)}")

    print(f"[+] 完成分割: 共 {part_num} 个文件, 总包数 {packet_count}")
    return output_files


# 测试用例（直接运行 python src/functions/split.py 会触发）
if __name__ == "__main__":
    test_input = "data/example.pcapng"  # 你的大 pcap 路径
    test_output = "data/split"
    files = split_pcap(test_input, test_output, packets_per_file=1000)
    print("生成的文件：", files)
