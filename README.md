# -_tms

面向网络流量异常检测的工具集，包含特征提取、预处理、无监督检测、结果分析等模块。本仓库现在增强了面向恶意流量的特征工程，并提供实战手册帮助判断恶意流量。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 提取特征：`python -m src.functions.feature_extractor <pcap路径> <输出csv>`
3. 预处理并训练：`python -m src.functions.unsupervised_train ...`（参考脚本目录中的示例）。
4. 使用 `scripts/` 下的脚本批量处理或集成 GUI。

## 新增能力速览

- **更丰富的流量特征**：新增多项对恶意流量更敏感的统计量（包长中位数/IQR、突发度、TCP 标志位比例、空闲时间占比等），提升模型识别爆发式攻击、扫描和单向渗透的能力。
- **恶意流量判别手册**：位于 [`docs/malicious_traffic_guide.md`](docs/malicious_traffic_guide.md)，总结了理论框架、常见攻击模式及特征解读，便于人工审计与调参与模型训练配合使用。

## 推荐流程

1. 运行 `extract_features` 重新生成特征 CSV，确保包含新的指标。
2. 通过 `preprocess_feature_dir` 与 `unsupervised_train` 训练 Ensemble 异常检测模型，观察 `calibration_report` 与 `supervised_metrics`。
3. 使用 `analyze_results` 生成可视化，并结合文档中的阈值建议对高风险流量进行复核。

更多细节和优化建议请参阅 [恶意流量判别速查手册](docs/malicious_traffic_guide.md)。
