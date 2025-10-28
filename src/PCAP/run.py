from static_features import extract_pcap_features

result = extract_pcap_features(r"D:\pythonProject8\data\split\22.pcapng")
if result["success"]:
    print(f"共解析到 {len(result['flows'])} 条流")
    print(result["flows"][0])  # 示例：打印第一条流的全部特征
else:
    print("解析失败：", result["error"])
