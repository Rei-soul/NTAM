# prefilter.py

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from config import DATA_MONTH_START, DATA_MONTH_END

# ========== 参数 ==========
DATA_DIR = "D:/2018Datasets"
OUTPUT_DIR = "datasets/processed"
LOCATION_FILE = "D:/2018Datasets/location_info_of_ssd.csv"
YEAR = 2018
MONTH_START = DATA_MONTH_START
MONTH_END = DATA_MONTH_END

# SMART 特征列名
SMART_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 175,
    177, 180, 181, 182, 183, 184, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 204, 205, 206, 207, 211,
    232, 233, 240, 241, 242, 244, 245
]
N_COLS = [f"n_{sid}" for sid in SMART_IDS]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ====== 第①步：加载全量位置信息 (disk_id, model) → node_id ======
    print("=" * 60)
    print("[1/5] 加载全量位置信息...")
    loc_df = pd.read_csv(LOCATION_FILE, usecols=['disk_id', 'node_id', 'model', 'rack_id'])
    loc_df['disk_id'] = loc_df['disk_id'].astype(int)
    loc_df['node_id'] = loc_df['node_id'].astype(int)
    loc_df['model'] = loc_df['model'].astype(str)

    # 构建 (disk_id, model) → (node_id, rack_id) 映射
    disk_loc_map = {}
    for disk_id, node_id, model, rack_id in zip(
        loc_df['disk_id'].values, loc_df['node_id'].values,
        loc_df['model'].values, loc_df['rack_id'].values
    ):
        key = (int(disk_id), str(model))
        disk_loc_map[key] = {
            'node_id': int(node_id),
            'rack_id': int(rack_id) if not pd.isna(rack_id) else -1
        }

    print(f"  位置信息覆盖 {len(disk_loc_map):,} 个 (disk_id, model) 组合")
    print(f"  唯一 node_id 数: {loc_df['node_id'].nunique():,}")

    # ====== 第②步：扫描多月份CSV，收集所有 (disk_id, model) ======
    print(f"\n[2/5] 扫描{YEAR}年{MONTH_START}~{MONTH_END}月CSV...")
    month_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith(str(YEAR))
        and MONTH_START <= int(f[4:6]) <= MONTH_END
    ])
    print(f"  找到 {len(month_files)} 个文件")

    # 收集所有唯一的 (disk_id, model) 对
    all_disk_model_pairs = set()
    for fname in tqdm(month_files, desc="扫描CSV"):
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath, usecols=['disk_id', 'model'])
        df['disk_id'] = df['disk_id'].astype(int)
        df['model'] = df['model'].astype(str)
        for did, model in df[['disk_id', 'model']].drop_duplicates().values:
            all_disk_model_pairs.add((int(did), str(model)))

    print(f"  总 (disk_id, model) 组合数: {len(all_disk_model_pairs):,}")

    # ====== 第③步：加载故障标签（按 disk_id 打标） ======
    print("\n[3/5] 加载故障标签...")
    tag_path = os.path.join(DATA_DIR, "ssd_failure_tag2.csv")
    tag_df = pd.read_csv(tag_path)
    tag_df['failure_time'] = pd.to_datetime(tag_df['failure_time'])

    failure_map = {}  # (disk_id, model) → failure_time
    for _, row in tag_df.iterrows():
        key = (int(row['disk_id']), str(row['model']))
        failure_map[key] = row['failure_time']

    n_failure_pairs = sum(1 for pair in all_disk_model_pairs if pair in failure_map)
    print(f"  故障标签 (disk_id, model) 对: {len(failure_map):,}")
    print(f"  其中在MONTH_START-MONTH_END中出现的: {n_failure_pairs:,}")
###############################################################################
    # ====== 第④步：筛选能匹配到位置信息的 (disk_id, model) 对 ======
    print("\n[4/5] 筛选 (disk_id, model) 对（保留全量，训练时再下采样）...")

    # 只保留在 location_info 中能找到的磁盘
    # 这确保了每块磁盘都有正确的 node_id，邻居构建不会丢失
    target_pairs = {pair for pair in all_disk_model_pairs if pair in disk_loc_map}

    n_fail_pairs = sum(1 for pair in target_pairs if pair in failure_map)
    n_healthy_pairs = len(target_pairs) - n_fail_pairs

    # 不在位置文件中的磁盘
    n_missing = len(all_disk_model_pairs - target_pairs)

    print(f"  目标 (disk_id, model) 对总数: {len(target_pairs):,}")
    print(f"    故障: {n_fail_pairs:,}, 健康: {n_healthy_pairs:,}")
    print(f"    无位置信息被丢弃: {n_missing:,}")
    print(f"  位置信息覆盖率: {len(target_pairs)/max(len(all_disk_model_pairs),1)*100:.1f}%")

    # ====== 第⑤步：保存目标清单 + 提取 SMART 数据 ======
    print("\n[5/5] 保存目标清单 + 提取SMART数据...")

    # 保存目标磁盘清单
    records = []
    for did, model in target_pairs:
        is_fail = (did, model) in failure_map
        ft = failure_map.get((did, model), pd.NaT)
        loc = disk_loc_map.get((did, model), {}) # 找不到该硬盘位置时返回空字典 {}，后面 .get 兜底。
        records.append({
            'disk_id': did,
            'model': model,
            'pair_id': f"{did}_{model}",  # 唯一标识字符串
            'is_failure': is_fail,
            'failure_time': ft if pd.notna(ft) else pd.NaT,
            'node_id': loc.get('node_id', -1),
            'rack_id': loc.get('rack_id', -1)
        })

    target_df = pd.DataFrame(records)
    target_path = os.path.join(OUTPUT_DIR, "target_disks.csv")
    target_df.to_csv(target_path, index=False)
    print(f"  → 已保存: {target_path} ({len(target_df)} 条)")

    # 统计邻居潜力
    target_node_groups = target_df[target_df['node_id'] >= 0].groupby('node_id').size()
    n_multi = (target_node_groups >= 2).sum()
    print(f"  有≥2个目标磁盘的 node 数: {n_multi}")
    print(f"  平均每 node 目标磁盘数: {target_node_groups.mean():.1f}")

    # parquet 生成已移除（data_utils.py 改为直接从原始 CSV 分块读取）
    print(f"  日期数: {len(month_files)} 天 (SMART 数据由 data_utils.py 按需提取)")

    print("\n" + "=" * 60)
    print("预筛选完成！")
    print(f"  目标磁盘对: {len(target_pairs):,}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()