# tools/split_for_compare_interactive.py
import os
import pandas as pd

BENIGN_TOKENS = {"benign", "normal", "0", "false", "neg", "negative", "clean"}


def detect_label_col(cols):
    """
    Try best to find label column from common candidates.
    Handles weird cols like " Label" with leading spaces.
    """
    candidates = [
        "LabelBinary", "labelbinary",
        "Label", "label",
        "class", "Class",
        "attack_cat", "Attack_cat",
        "ground_truth", "GroundTruth", "Ground Truth",
        " Label", " label"
    ]
    # normalized map: strip + lower
    norm_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]

    # fallback: any column contains "label"
    for c in cols:
        if "label" in str(c).strip().lower():
            return c

    return None


def is_benign_series(s: pd.Series) -> pd.Series:
    """
    Determine which rows are benign.
    - If numeric: <=0 -> benign, >0 -> attack
    - If string: benign/normal -> benign, otherwise -> attack
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return (num.fillna(0.0) <= 0)

    t = s.fillna("").astype(str).str.strip().str.lower()
    return (
        t.isin(BENIGN_TOKENS)
        | t.str.contains("benign")
        | t.str.contains("normal")
    )


def stratified_sample_large_csv(
    in_path: str,
    out_path: str,
    per_class: int = 10000,
    chunksize: int = 50000,
    seed: int = 42,
):
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"输入文件不存在：{in_path}")

    out_dir = os.path.dirname(out_path)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    # read only header to detect label col
    cols = pd.read_csv(in_path, nrows=0).columns.tolist()
    label_col = detect_label_col(cols)
    if label_col is None:
        raise ValueError("找不到标签列：需要 LabelBinary 或 Label（BENIGN/攻击名）等")

    # if out exists, overwrite
    if os.path.exists(out_path):
        os.remove(out_path)

    need_benign = per_class
    need_attack = per_class
    wrote_header = False
    total_written = 0

    for chunk in pd.read_csv(in_path, chunksize=chunksize):
        benign_mask = is_benign_series(chunk[label_col])
        benign_rows = chunk[benign_mask]
        attack_rows = chunk[~benign_mask]

        benign_pick = benign_rows.iloc[0:0]
        attack_pick = attack_rows.iloc[0:0]

        if need_benign > 0 and len(benign_rows) > 0:
            take = min(need_benign, len(benign_rows))
            # sample inside chunk to avoid ordering bias
            benign_pick = benign_rows.sample(n=take, random_state=seed)

        if need_attack > 0 and len(attack_rows) > 0:
            take = min(need_attack, len(attack_rows))
            attack_pick = attack_rows.sample(n=take, random_state=seed + 1)

        picked = pd.concat([benign_pick, attack_pick], axis=0)
        if len(picked) == 0:
            continue

        picked.to_csv(
            out_path,
            mode="a",
            index=False,
            header=(not wrote_header),
            encoding="utf-8-sig",
        )
        wrote_header = True

        need_benign -= len(benign_pick)
        need_attack -= len(attack_pick)
        total_written += len(picked)

        if need_benign <= 0 and need_attack <= 0:
            break

    if total_written == 0:
        raise RuntimeError("没有写出任何样本：请检查标签列/标签取值是否识别正确。")

    print("\n✅ 切割完成")
    print("输出文件：", out_path)
    print("标签列：", label_col)
    print(f"写出总行数：{total_written}（目标≈{per_class*2}）")
    if need_benign > 0 or need_attack > 0:
        print(f"⚠️ 注意：某一类样本不足，未抽到：benign {need_benign}, attack {need_attack}")


def main():
    print("=== 分层抽样切割（用于高敏/低敏对比）===")
    in_path = input("输入原始CSV路径：").strip().strip('"')
    out_path = input("输出小CSV路径：").strip().strip('"')

    per_class = input("每类抽多少行（默认10000）：").strip()
    per_class = int(per_class) if per_class else 10000

    chunksize = input("chunksize（默认50000，越大越快但更吃内存）：").strip()
    chunksize = int(chunksize) if chunksize else 50000

    stratified_sample_large_csv(
        in_path=in_path,
        out_path=out_path,
        per_class=per_class,
        chunksize=chunksize,
    )


if __name__ == "__main__":
    main()
