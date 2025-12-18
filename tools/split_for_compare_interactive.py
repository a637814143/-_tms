# tools/split_for_compare_interactive.py
import os
import time
import pandas as pd

BENIGN_TOKENS = {"benign", "normal", "0", "false", "neg", "negative", "clean"}


def detect_label_col(cols):
    candidates = [
        "LabelBinary", "labelbinary",
        "Label", "label",
        "class", "Class",
        "attack_cat", "Attack_cat",
        "ground_truth", "GroundTruth", "Ground Truth",
        " Label", " label"
    ]
    norm_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]

    for c in cols:
        if "label" in str(c).strip().lower():
            return c
    return None


def is_benign_series(s: pd.Series) -> pd.Series:
    """
    benign 判定：
    - 数值型：<=0 为 benign，>0 为 attack
    - 字符串型：benign/normal 为 benign，其它非空为 attack
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return (num.fillna(0.0) <= 0)

    t = s.fillna("").astype(str).str.strip().str.lower()
    return t.isin(BENIGN_TOKENS) | t.str.contains("benign") | t.str.contains("normal")


def resolve_out_path(in_path: str, out_input: str, tag: str) -> str:
    out_input = out_input.strip().strip('"')

    def auto_name():
        base = os.path.basename(in_path).replace(".csv", "")
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{tag}_{base}_{ts}_small.csv"

    if os.path.isdir(out_input):
        return os.path.join(out_input, auto_name())

    if out_input.endswith("\\") or out_input.endswith("/"):
        os.makedirs(out_input, exist_ok=True)
        return os.path.join(out_input, auto_name())

    out_dir = os.path.dirname(out_input)
    if out_dir and (not os.path.exists(out_dir)):
        os.makedirs(out_dir, exist_ok=True)

    if not out_input.lower().endswith(".csv"):
        out_input += ".csv"
    return out_input


def safe_remove_file(path: str):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            raise IsADirectoryError(f"输出路径是目录不是文件：{path}")


def _get_label_col(in_path: str) -> str:
    cols = pd.read_csv(in_path, nrows=0).columns.tolist()
    label_col = detect_label_col(cols)
    if label_col is None:
        raise ValueError("找不到标签列：需要 LabelBinary 或 Label（BENIGN/攻击名）等")
    return label_col


# -------------------------
# Mode 1: balanced
# -------------------------
def split_balanced(in_path: str, out_path: str, per_class: int, chunksize: int, seed: int = 42):
    label_col = _get_label_col(in_path)
    safe_remove_file(out_path)

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
            benign_pick = benign_rows.sample(n=take, random_state=seed)

        if need_attack > 0 and len(attack_rows) > 0:
            take = min(need_attack, len(attack_rows))
            attack_pick = attack_rows.sample(n=take, random_state=seed + 1)

        picked = pd.concat([benign_pick, attack_pick], axis=0)
        if len(picked) == 0:
            continue

        picked.to_csv(out_path, mode="a", index=False, header=(not wrote_header), encoding="utf-8-sig")
        wrote_header = True

        need_benign -= len(benign_pick)
        need_attack -= len(attack_pick)
        total_written += len(picked)

        if need_benign <= 0 and need_attack <= 0:
            break

    print("\n✅ balanced 切割完成")
    print("输出文件：", out_path)
    print("标签列：", label_col)
    print(f"写出总行数：{total_written}（目标≈{per_class*2}）")
    if need_benign > 0 or need_attack > 0:
        print(f"⚠️ 注意：某一类样本不足：benign 缺 {need_benign}, attack 缺 {need_attack}")


# -------------------------
# Mode 2: attack_all
# -------------------------
def split_attack_all(in_path: str, out_path: str, benign_n: int, chunksize: int, seed: int = 42):
    label_col = _get_label_col(in_path)
    safe_remove_file(out_path)

    need_benign = benign_n
    wrote_header = False
    total_written = 0
    total_attack = 0
    total_benign = 0

    for chunk in pd.read_csv(in_path, chunksize=chunksize):
        benign_mask = is_benign_series(chunk[label_col])
        benign_rows = chunk[benign_mask]
        attack_rows = chunk[~benign_mask]

        picked_attack = attack_rows  # 全取 attack
        if need_benign > 0 and len(benign_rows) > 0:
            take = min(need_benign, len(benign_rows))
            picked_benign = benign_rows.sample(n=take, random_state=seed)
        else:
            picked_benign = benign_rows.iloc[0:0]

        picked = pd.concat([picked_attack, picked_benign], axis=0)
        if len(picked) == 0:
            continue

        picked.to_csv(out_path, mode="a", index=False, header=(not wrote_header), encoding="utf-8-sig")
        wrote_header = True

        total_attack += len(picked_attack)
        total_benign += len(picked_benign)
        total_written += len(picked)

        need_benign -= len(picked_benign)
        # benign 够了也继续扫，因为要把后续 attack 全取

    print("\n✅ attack_all 切割完成")
    print("输出文件：", out_path)
    print("标签列：", label_col)
    print(f"写出总行数：{total_written}（attack 全取 + benign 抽样）")
    print(f"attack 行数：{total_attack}")
    print(f"benign 行数：{total_benign}（目标={benign_n}）")
    if need_benign > 0:
        print(f"⚠️ 注意：benign 不足，仍缺 {need_benign}")


# -------------------------
# Mode 3: ratio
# -------------------------
def split_ratio(
    in_path: str,
    out_path: str,
    attack_n: int,
    benign_ratio: float,
    chunksize: int,
    seed: int = 42
):
    """
    ratio 模式：
    - 抽 attack_n 条攻击
    - 抽 benign_n = int(attack_n * benign_ratio) 条 benign
    例：benign_ratio=10 => benign:attack=10:1
    """
    if attack_n <= 0:
        raise ValueError("attack_n 必须 > 0")
    if benign_ratio <= 0:
        raise ValueError("benign_ratio 必须 > 0")

    label_col = _get_label_col(in_path)
    safe_remove_file(out_path)

    benign_n = int(round(attack_n * benign_ratio))
    need_attack = attack_n
    need_benign = benign_n

    wrote_header = False
    total_written = 0
    got_attack = 0
    got_benign = 0

    for chunk in pd.read_csv(in_path, chunksize=chunksize):
        benign_mask = is_benign_series(chunk[label_col])
        benign_rows = chunk[benign_mask]
        attack_rows = chunk[~benign_mask]

        attack_pick = attack_rows.iloc[0:0]
        benign_pick = benign_rows.iloc[0:0]

        if need_attack > 0 and len(attack_rows) > 0:
            take = min(need_attack, len(attack_rows))
            attack_pick = attack_rows.sample(n=take, random_state=seed + 1)

        if need_benign > 0 and len(benign_rows) > 0:
            take = min(need_benign, len(benign_rows))
            benign_pick = benign_rows.sample(n=take, random_state=seed)

        picked = pd.concat([attack_pick, benign_pick], axis=0)
        if len(picked) == 0:
            continue

        picked.to_csv(out_path, mode="a", index=False, header=(not wrote_header), encoding="utf-8-sig")
        wrote_header = True

        got_attack += len(attack_pick)
        got_benign += len(benign_pick)
        total_written += len(picked)

        need_attack -= len(attack_pick)
        need_benign -= len(benign_pick)

        if need_attack <= 0 and need_benign <= 0:
            break

    print("\n✅ ratio 切割完成")
    print("输出文件：", out_path)
    print("标签列：", label_col)
    print(f"写出总行数：{total_written}")
    print(f"attack 行数：{got_attack}（目标={attack_n}）")
    print(f"benign 行数：{got_benign}（目标={benign_n}，比例≈{benign_ratio}:1）")
    if need_attack > 0 or need_benign > 0:
        print(f"⚠️ 注意：样本不足：attack 缺 {need_attack}, benign 缺 {need_benign}")


def main():
    print("=== 分层切割（用于高敏/低敏对比）===")
    in_path = input("输入原始CSV路径：").strip().strip('"')
    out_input = input("输出路径（可输入目录或文件名）：").strip().strip('"')

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"输入文件不存在：{in_path}")

    mode = input(
        "选择模式：\n"
        "  1=balanced（每类抽样）\n"
        "  2=attack_all（攻击全取+benign抽N）\n"
        "  3=ratio（按比例抽样 benign:attack = R:1）\n"
        "请输入 1/2/3 [默认1]: "
    ).strip()
    mode = mode if mode else "1"

    chunksize = input("chunksize（默认50000）：").strip()
    chunksize = int(chunksize) if chunksize else 50000

    if mode == "1":
        per_class = input("每类抽多少行（默认10000）：").strip()
        per_class = int(per_class) if per_class else 10000
        out_path = resolve_out_path(in_path, out_input, tag=f"balanced_{per_class*2}")
        print("\n将输出到：", out_path)
        split_balanced(in_path, out_path, per_class=per_class, chunksize=chunksize)

    elif mode == "2":
        benign_n = input("benign 抽多少行（默认10000）：").strip()
        benign_n = int(benign_n) if benign_n else 10000
        out_path = resolve_out_path(in_path, out_input, tag=f"attack_all_benign{benign_n}")
        print("\n将输出到：", out_path)
        split_attack_all(in_path, out_path, benign_n=benign_n, chunksize=chunksize)

    elif mode == "3":
        attack_n = input("attack 抽多少行（例如 2000）：").strip()
        attack_n = int(attack_n) if attack_n else 2000

        ratio = input("benign:attack 比例 R（例如 10 表示 10:1，默认10）：").strip()
        ratio = float(ratio) if ratio else 10.0

        out_path = resolve_out_path(in_path, out_input, tag=f"ratio_attack{attack_n}_R{ratio}")
        print("\n将输出到：", out_path)
        split_ratio(in_path, out_path, attack_n=attack_n, benign_ratio=ratio, chunksize=chunksize)

    else:
        print("无效选择，请输入 1 / 2 / 3")


if __name__ == "__main__":
    main()
