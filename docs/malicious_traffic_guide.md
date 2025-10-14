# 恶意流量判别速查手册

本手册整理了检测网络恶意流量的关键理论、实战策略以及在本项目中可以直接使用的特征指标，帮助你在查看流量或训练模型时做到**有据可依、快速定位**。

## 1. 恶意流量判别的整体框架

1. **建立基线**：
   - 对业务正常流量做时间、协议、端口、IP 分布统计，记住峰值/平均模式。
   - 使用项目提供的 `summary_by_file.csv`、`anomaly_score_distribution.png` 先找出异常波动的文件或时间窗口。

2. **特征工程**：
   - 关注每条流的**方向性**（fwd/bwd 比例）、**包长分布**、**间隔分布**、**TCP 标志位组合**、**突发度（burstiness）**等指标。
   - 本次更新新增了 `fwd_pkt_len_median`、`burstiness_fwd`、`idle_time_fraction` 等 20+ 个面向威胁的特征，能够更精确地捕捉扫描、爆发式攻击、单向传输等模式。

3. **多维度佐证**：
   - 结合统计图、模型分数、端口/协议知识，优先锁定风险最大的 IP/端口段。
   - 在模型给出高置信度告警时，从 PCAP 中抽样复核（推荐查看 TCP 握手完整性、重复报文、应用层指纹）。

4. **闭环验证**：
   - 若存在标签或情报，使用 `unsupervised_train` 里的监督校准结果 (`calibration_report`、`supervised_metrics`) 验证精度。
   - 将确认后的恶意样本加入训练集或特征库，不断迭代。

## 2. 如何理解新增的核心特征

| 特征名称 | 指标解释 | 典型异常含义 |
| --- | --- | --- |
| `byte_symmetry` / `packet_symmetry` | 双向字节/包数量差异占比 | 单向渗透、数据外泄、拒绝响应型攻击 |
| `fwd_pkt_len_iqr`、`bwd_pkt_len_iqr` | 包长分布的四分位距 | DDoS 中同质化包；恶意工具常呈低 IQR |
| `burstiness_fwd`、`burstiness_bwd` | 间隔标准差 / 均值 | 扫描、爆发式请求、僵尸网络心跳 |
| `idle_time_fraction` | 最大空闲时间 / 总时长 | 长时间潜伏后突发的 C2 或批量下载 |
| `fwd_flag_syn_rate` 等 | 各标志位在当前方向出现比例 | SYN 泛洪（SYN 占比高）、RST 泛滥（异常复位）|
| `rst_to_packet_ratio` | RST 包占比 | 探测/拒绝连接、扫描被拦截的迹象 |
| `p95_fwd_inter` 等 | 95 分位的时间间隔 | 区分周期性心跳、低速扫描 |

这些特征与原有的包长直方图、时间间隔直方图组合，可以让 IsolationForest/OCSVM 更容易区分出“低方差、单向、突发”的恶意流量。

## 3. 常见且容易识别的恶意流量模式

| 类型 | 典型行为 | 重点特征/阈值提示 |
| --- | --- | --- |
| **端口扫描**（SYN/FIN/NULL/UDP） | 短时访问大量端口；无完整握手 | `burstiness_fwd` ↑、`rst_to_packet_ratio` ↑、`fwd_flag_syn_rate` ≥ 0.8；`packets_per_s` 高但 `byte_symmetry` → 1 |
| **暴力破解/字典攻击** | 针对单个服务反复登陆 | `idle_time_fraction` 低、`packets_per_s` 中等；`byte_symmetry` 偏高（请求远大于响应）；同一源 IP 高频失败 |
| **DDoS / DoS** | 大量同构小包、单向打满 | `fwd_pkt_len_iqr` ≈ 0、`burstiness_fwd` 低但 `packets_per_s` 极高；`byte_symmetry` 接近 1 |
| **数据渗透/下载** | 长时间单向传输大文件 | `byte_symmetry`、`packet_symmetry` → 1；`avg_pkt_size` 大；`idle_time_fraction` 高后突增 |
| **C2 心跳 / 僵尸网络** | 周期性小包心跳 | `p90_*_inter` 与 `median_*_inter` 非常接近；`burstiness` 极低；总包数少但持续时间长 |
| **恶意横向移动** | 内网高端口交互、SMB/RDP | 关注端口 135/139/445/3389/5985 等；`rst_to_packet_ratio` 中等、`fwd_flag_ack_rate` ≈ 1 表示会话保持 |

> 建议：结合 `summary_by_file.csv` 查看同一源 IP 的恶意占比，确认是否存在横向扩散。

## 4. 实战排查清单

1. **按模型置信度排序**：查看 `anomaly_score` 排名前 5% 的流，优先分析 `flag_*_imbalance`、`byte_symmetry`。
2. **按端口/协议聚合**：
   - 服务端口异常升高：可能遭受扫描或爆破。
   - 非常见端口（≥1024）突然出现大量外连：检查是否为代理、隧道、C2。
3. **时间序列复盘**：利用新增的 `idle_time_fraction`、`max_*_inter`，判断是否“潜伏—爆发”模式。
4. **日志回溯**：
   - 对高 `rst_to_packet_ratio` 的流量，查阅防火墙/IDS 日志验证拦截记录。
   - 对高 `burstiness` 的流量，尝试关联主机进程、排查批量任务或脚本。
5. **复核与调参**：
   - 如果告警多但确认少，可适当提高 `vote_threshold` 或在 `unsupervised_train` 中调低 `contamination`。
   - 有标签时，关注 `calibration_report['f0.5']` 是否超过 0.8，以衡量精准率优先的表现。

## 5. 与项目结合的建议

- **特征导出**：重新运行 `extract_features`，新的统计量会直接写入 CSV，再经 `preprocess_feature_dir` 进入训练管线。
- **模型训练**：`EnsembleAnomalyDetector` 会自动利用这些特征，并通过 `burstiness` 等维度在 `QuantileTransformer + RBFSampler` 中得到更分散的表征，有助于提升 `f0.5` 和召回率。
- **人工审计**：利用 `analyze_results` 生成的图表配合本手册的阈值参考，可以快速筛出“明显恶意”与“疑似异常”。

---

> 需要打印或分享时，可直接将本文件导出为 PDF。建议在关键特征旁边补充自己业务的经验阈值，形成组织内的“流量画像”。
