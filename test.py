from pathlib import Path

from src.functions.modeling import train_supervised_on_split

if __name__ == "__main__":
    split_dir = Path("data/111")  # 你存放切分包的目录
    results_dir = Path("data/results")  # 输出结果目录
    models_dir = Path("data/models")  # 输出模型目录

    if not split_dir.exists():
        raise SystemExit("请先准备带标签的训练 CSV 目录后再运行本示例。")

    result = train_supervised_on_split(
        split_dir=split_dir,
        results_dir=results_dir,
        models_dir=models_dir,
        label_col="Label",
    )

    print("✅ 有监督模型训练完成")
    print(result)
