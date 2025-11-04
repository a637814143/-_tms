from src.functions.unsupervised_train import train_unsupervised_on_split

if __name__ == "__main__":
    split_dir   = "data/111"       # 你存放切分包的目录
    results_dir = "data/results"     # 输出结果目录
    models_dir  = "data/models"      # 输出模型目录

    result = train_unsupervised_on_split(
        split_dir=split_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        contamination=0.05,   # 假设 5% 是恶意流量
        base_estimators=50    # 每次迭代增加树的数量
    )

    print("✅ 训练完成")
    print(result)
