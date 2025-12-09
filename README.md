# 流量分析平台

## 项目简介
基于机器学习 + 规则引擎的网络流量恶意检测平台，支持命令行与 PyQt5 图形界面两种入口。

## 主要功能
- 流量特征提取（PCAP → CSV）。
- 有监督模型训练（HistGradientBoosting 等集成）。
- 模型 + 风险规则融合检测与告警。
- 在线/批量检测流程与结果汇总。
- PyQt5 可视化界面（训练、预测、结果浏览）。

## 项目结构
```
scripts/
  ROOT.py                       # 项目根路径引用
  run_fix_dataset_label_columns.py # 修复错位标签数据集的小工具
  train_experiment_baseline.py  # 一键跑基线训练并导出指标
  train_experiment_rule_weight.py # 扫描不同模型/规则权重组合
src/
  functions/                    # 特征、建模、规则融合、导出等核心函数
  services/                     # CLI/服务化入口（提取、训练、预测、分析）
  ui/                           # PyQt5 界面
config/
  default.yaml                  # 训练、路径、规则等默认配置
requirements.txt                # 运行依赖
README.md
```

## 环境与依赖安装
- 推荐 Python 3.10+。
- 安装依赖：`pip install -r requirements.txt`。
- 如需处理 PCAP，请确保具备 Scapy/tshark 的读取权限（某些环境需要管理员/根权限）。

## 快速开始
### 命令行流程
1. 准备 PCAP/CSV 数据。
2. 提取特征（示例使用 pipeline_service 的 extract）：
   ```bash
   python -m src.services.pipeline_service extract data/pcap_dir data/CSV/feature --workers 4
   ```
3. 训练模型：
   ```bash
   python scripts/train_experiment_baseline.py --train-csv data/CSV/feature --output-dir data/results/baseline
   ```
4. 使用已训练管线预测并生成结果 CSV：
   ```bash
   python -m src.services.pipeline_service predict \
     data/results/baseline/model.joblib data/CSV/feature/test.csv \
     --output data/results/predict.csv
   ```

### GUI 流程
1. 启动界面：`python scripts/ROOT.py` 或 `python -m src.ui.ui_main`。
2. 按界面步骤：选择数据 → 提取/加载特征 → 训练模型 → 载入模型做检测 → 查看/导出结果。

## 实验脚本说明
- **train_experiment_baseline.py**：读取配置与训练集，调用 `train_supervised_on_split` 训练基线模型，输出模型、元数据与 `baseline_metrics.csv`。
- **train_experiment_rule_weight.py**：基于同一模型扫描不同 `(model_weight, rule_weight)` 组合，使用融合逻辑计算精度/召回/F1/AUC，汇总到 `rule_weight_metrics.csv`。
- **run_fix_dataset_label_columns.py**：修复表头 86 列但行数据 88 列的错位标签 CSV，可对单个文件或目录批量处理，并写出 `_fixed` 版本。

## 配置说明
`config/default.yaml` 关键字段：
- **paths**：数据、模型、结果、日志等默认目录（如 `paths.data_dir`、`paths.models_dir`）。
- **training**：有监督训练与模型/规则融合的核心参数（如 `fusion_alpha`）。
- **rules**：规则 profile、阈值与模型/规则融合权重（`rules.active_profile`、`rules.fusion.model_weight/rule_weight/decision_threshold`）。
- **ui/online_detection**：界面与在线检测轮询配置。

## 数据格式说明
- 训练 CSV 至少包含若干数值特征列与 `LabelBinary` 标签列（0 表示良性，1 表示恶意）。
- 特征列可为通用数值型字段，训练流程会自动筛选数值特征并忽略 `Label/LabelBinary/Attack_cat` 等标签列。
- 预测结果包含 `prediction_status`/`malicious_score` 等列，可配合规则得分进行融合分析。

## 示例结果与截图
- 可在后续实验中补充训练曲线、规则权重对比图、GUI 截图，直接将生成的 PNG 放在 `results/` 或文档中引用。

## 后续工作 / 限制说明
- 丰富特征提取 CLI，支持更多 PCAP 预处理选项。
- 增强规则配置可视化与可编辑性，便于快速实验不同策略。
- 扩展自动化测试覆盖更多真实数据流转场景（特征提取→训练→预测）。
- 考虑加入模型版本管理与在线推理 API，以便部署到生产环境。
