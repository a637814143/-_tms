# 流量分析平台

## 项目概述

这是一个基于Python开发的流量分析平台，支持流量数据的导入、预处理、特征提取、机器学习建模、评估与可视化。平台采用模块化设计，既可图形界面操作，也支持命令行。

## 项目结构

```
pythonProject8/
├── README.md                          # 项目说明文档（功能简介、使用方式、数据来源等）
├── requirements.txt                   # 项目依赖列表（如 pandas, scikit-learn, matplotlib 等）
├── run.py                             # 程序主入口（启动图形界面或 CLI）
├── src/                               # 源代码目录
│   ├── __init__.py                    # 包声明
│   ├── main.py                        # 控制主逻辑（整合模型、界面、流程调度）
│   ├── learn.py                       # 封装机器学习训练与预测流程（如 fit → predict）
│   ├── 功能/                           # 主要功能模块
│   │   ├── __init__.py
│   │   ├── preprocessor.py            # 数据预处理模块（缺失值、编码、标准化等）
│   │   ├── extractor.py               # 特征提取模块（从 pcap 提取流量特征）
│   │   ├── evaluator.py               # 模型评估模块（准确率、F1、AUC、对比等）
│   │   └── utils.py                   # 公共工具函数（日志、文件加载等）
│   └── 可视化/                         # 界面与图表展示模块
│       ├── __init__.py
│       ├── ui.py                      # 图形界面主程序（Tkinter）
│       └── chart.py                   # 图表绘制（性能对比、热力图、柱状图等）
├── data/                              # 数据目录（放入.pcap/.csv 或转换后数据）
│   ├── example.pcapng
│   └── cleaned.csv
├── models/                            # 模型存储目录（如 joblib 格式的模型）
│   ├── rf_model.pkl
│   └── iforest_model.pkl
├── results/                           # 实验结果目录（日志、图表、报告）
│   ├── output.log
│   └── comparison.png
├── .venv/                             # Python虚拟环境
└── .idea/                             # IDE 配置文件夹
```

## 主要模块说明

- **requirements.txt**：项目依赖包列表，便于环境搭建。
- **run.py**：程序主入口，可选择启动图形界面或命令行分析。
- **src/main.py**：主逻辑调度，负责各模块的调用与流程控制。
- **src/learn.py**：机器学习训练与预测流程的封装。
- **src/功能/preprocessor.py**：数据清洗、缺失值处理、特征标准化等。
- **src/功能/extractor.py**：从原始流量文件（如pcap）中提取特征。
- **src/功能/evaluator.py**：模型评估与对比分析。
- **src/功能/utils.py**：通用工具函数，如日志、文件加载等。
- **src/可视化/ui.py**：基于Tkinter的图形界面主程序。
- **src/可视化/chart.py**：各类图表绘制与展示。
- **data/**：存放原始数据和处理后数据。
- **models/**：存放训练好的模型文件。
- **results/**：存放实验日志、图表和分析报告。

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 运行主程序：`python run.py`
3. 按照界面或命令行提示操作。

## 数据格式说明
- 支持.pcap/.csv等格式，建议先用extractor模块提取特征。

## 版本信息
- 当前版本：v1.0.0
- 最后更新：2025年1月

---
© 2025 by AI Project. All rights reserved. 