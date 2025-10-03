# 项目改进建议

## 1. 标签关键字配置化与管理界面
当前 `train_unsupervised_on_split` 流程里将 `LABEL_KEYWORDS`、`POSITIVE_TOKENS` 等常量硬编码在模块顶部，一旦需要支持新的字段或企业自定义术语，就需要修改代码并重新发布。建议将这些关键字提取到独立的 YAML/JSON 配置文件，并在界面新增“标签字典”管理页，支持在线增删关键字并一键同步到训练流程，这样可以快速适配不同数据集，同时避免误改代码引入风险。【F:src/functions/unsupervised_train.py†L19-L74】【F:src/ui/ui_main.py†L9-L56】

## 2. 增强模型训练的评估与调参能力
目前的训练逻辑固定使用单组超参数的 IsolationForest，评估指标只记录整体准确率，缺少对敏感阈值和样本不平衡的探索。建议引入自动调参（例如对 `contamination`、`n_estimators` 网格搜索）和交叉验证/留出集评估，同时在结果中输出 ROC/PR 曲线数据，帮助你根据不同场景选择最优模型。【F:src/functions/unsupervised_train.py†L232-L311】

## 3. 增加特征提取的可扩展性与性能监控
`feature_extractor` 目前聚焦于流量统计，但对 TLS 指纹、DNS 字段等高价值特征缺乏支持，且缺少可观察性。建议将 `FlowAccumulator.to_row` 拆分为模块化的特征插件接口，允许按需扩展协议特征；同时在多线程提取时记录处理耗时、异常和速率，方便定位性能瓶颈。【F:src/functions/feature_extractor.py†L70-L176】
