# 流量分析平台

## 项目概述

基于 Python 的流量分析平台，支持 PCAP/CVS 特征提取、模型训练与预测、规则与模型融合的异常检测，以及可视化与结果导出。既可通过命令行批处理，也提供基于 PyQt5 的图形界面。

## 项目结构

```
./
├── README.md                    # 项目说明
├── requirements.txt             # 依赖列表（包含 UI、特征提取、模型训练所需包）
├── config/
│   └── default.yaml             # 默认配置
├── src/
│   ├── configuration.py         # 路径/配置加载工具
│   ├── ui/
│   │   └── ui_main.py           # PyQt5 GUI 入口（训练、预测、分析一体化）
│   ├── services/
│   │   └── pipeline_service.py  # CLI 入口：extract/train/predict/analyze
│   └── functions/               # 特征工程、模型、规则、日志等业务逻辑
│       ├── feature_extractor.py
│       ├── info.py
│       ├── modeling.py
│       ├── analyze_results.py
│       ├── risk_rules.py
│       └── ...
├── tests/                       # 测试用例
└── test.py                      # 示例/快速验证脚本
```

## 使用方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 启动图形界面（PyQt5）：
   ```bash
   python -m src.ui.ui_main
   ```
3. 命令行模式（适合批量处理/训练/预测/分析）：
   ```bash
   # 提取特征
   python -m src.services.pipeline_service extract --input <pcap目录或文件> --output <features.csv>

   # 训练模型（需要已分好训练/验证集的目录或 CSV）
   python -m src.services.pipeline_service train --input <split目录或csv> --output <模型输出目录>

   # 预测
   python -m src.services.pipeline_service predict --pipeline <模型文件> --features <features.csv> --output <结果路径>

   # 结果分析/可解释性统计
   python -m src.services.pipeline_service analyze --input <预测结果csv>
   ```

## 数据格式说明
- 支持 pcap/pcapng（通过 `feature_extractor.py` 与 `info.py` 提取统计与特征）。
- 训练/预测特征采用 CSV，标签列默认 `Label`/`label`/`class`/`ground_truth`（界面与 CLI 会自动探测）。

## 版本信息
- 当前版本：v1.0.0
- 最后更新：2025年1月