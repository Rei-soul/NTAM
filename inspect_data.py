# inspect_data.py
# 独立的数据审查脚本：加载按天分片的 .npy 文件，交互式查看磁盘 SMART 特征
#
# 依赖: numpy + pandas（无 torch）
# 用法: python inspect_data.py

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import *

PROCESSED_DIR = "datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")
NEIGHBOR_MAP_FILE = os.path.join(PROCESSED_DIR, "neighbor_map.csv")
PID_INDEX_FILE = os.path.join(PROCESSED_DIR, "pid_index.csv")

N_COLS = [f"n_{sid}" for sid in [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 175,
    177, 180, 181, 182, 183, 184, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 204, 205, 206, 207, 211,
    232, 233, 240, 241, 242, 244, 245
]]


# ============================================================
# 1. 构建 / 加载磁盘索引映射
# ============================================================

def build_pid_index():
    """重建 pid → extract_idx 映射，结果缓存到 pid_index.csv"""
    print("=" * 60)
    print("[inspect_data] 构建磁盘索引映射...")

    # 加载磁盘清单
    target_df = pd.read_csv(TARGET_FILE)
    target_df['failure_time'] = pd.to_datetime(target_df['failure_time'])

    disk_info = {}
    for _, row in target_df.iterrows():
        pid = str(row['pair_id'])
        disk_info[pid] = {
            'model': str(row['model']),
            'is_failure': bool(row['is_failure']),
            'failure_time': row['failure_time'] if pd.notna(row['failure_time']) else None,
            'node_id': int(row['node_id']) if pd.notna(row['node_id']) and row['node_id'] >= 0 else None
        }

    # 加载邻居映射
    if os.path.exists(NEIGHBOR_MAP_FILE):
        neighbor_df = pd.read_csv(NEIGHBOR_MAP_FILE)
        neighbor_df['pair_id'] = neighbor_df['pair_id'].astype(str)
        neighbor_df['neighbor_pair_id'] = neighbor_df['neighbor_pair_id'].astype(str)
        neighbor_map = neighbor_df.groupby('pair_id')['neighbor_pair_id'].apply(list).to_dict()
        for pid in disk_info:
            if pid not in neighbor_map:
                neighbor_map[pid] = []
    else:
        print(f"  警告: neighbor_map.csv 不存在，邻居信息不可用")
        neighbor_map = {pid: [] for pid in disk_info}

    # 重建 extract_pids（与 _extract_and_build_feat 中的 sorted(all_needed) 一致）
    all_needed = set(disk_info.keys())
    for pid in disk_info:
        all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
    extract_pids = sorted(all_needed)
    pid_to_idx = {pid: i for i, pid in enumerate(extract_pids)}

    # 缓存
    rows = []
    for pid in extract_pids:
        info = disk_info[pid]
        rows.append({
            'pair_id': pid,
            'idx': pid_to_idx[pid],
            'is_failure': info['is_failure'],
            'node_id': info['node_id'] if info['node_id'] is not None else -1
        })
    pd.DataFrame(rows).to_csv(PID_INDEX_FILE, index=False)

    # 统计 npy 文件
    feat_files = _find_feat_files()

    print(f"  磁盘总数: {len(extract_pids):,}")
    print(f"  按天文件数: {len(feat_files)}")
    print(f"  → 已缓存: {PID_INDEX_FILE}")
    print("=" * 60)

    return disk_info, neighbor_map, pid_to_idx, feat_files


def _find_feat_files():
    """扫描 PROCESSED_DIR，返回已存在的 feat_day_*.npy 列表"""
    files = sorted([
        f for f in os.listdir(PROCESSED_DIR)
        if f.startswith("feat_day_") and f.endswith(".npy")
    ])
    if not files:
        raise FileNotFoundError(f"{PROCESSED_DIR} 中没有 feat_day_*.npy 文件，请先运行 data_utils.py")
    return [os.path.join(PROCESSED_DIR, f) for f in files]


# ============================================================
# 2. 交互命令处理
# ============================================================

def print_help():
    print("""
交互命令:
  help                         打印帮助
  dates                        列出所有日期及其日索引
  disks                        显示磁盘总数
  disk <pair_id>               查找磁盘的索引和基本信息
  neighbors <pair_id>          列出邻居 pair_id
  stats <day_idx>              显示某天的特征统计(均值/方差/全零率/NaN率)
  show <disk_idx> <day_idx> [n_attrs]  显示磁盘在某天的 SMART 特征(可限制前 n 个属性)
  seq <disk_idx> <di_start> <di_end>   显示磁盘在日索引范围内的完整时序
  quit / exit                  退出
""")


def cmd_dates(feat_files):
    """打印日期列表"""
    n_dates = len(feat_files)
    print(f"  日期索引: 0 ~ {n_dates - 1} (共 {n_dates} 天)")
    # 实际上需要知道具体日期字符串，但 current 没有存储映射
    # 可以反推：从 data_utils._scan_csv_dates 但这里不 import，留一个提示
    print("  提示: 日索引 0 = 第一个 CSV 文件 (如 20180101)，可参考 data_utils._scan_csv_dates()")


def cmd_disk(pid, pid_to_idx, disk_info):
    if pid not in pid_to_idx:
        print(f"  磁盘 '{pid}' 未在提取清单中找到")
        return
    idx = pid_to_idx[pid]
    info = disk_info[pid]
    print(f"  pair_id    : {pid}")
    print(f"  extract_idx: {idx}")
    print(f"  is_failure : {info['is_failure']}")
    print(f"  node_id    : {info['node_id']}")
    if info['failure_time'] is not None:
        print(f"  failure_time: {info['failure_time']}")


def cmd_neighbors(pid, neighbor_map):
    if pid not in neighbor_map:
        print(f"  磁盘 '{pid}' 未在邻居映射中找到")
        return
    nbs = neighbor_map[pid]
    print(f"  邻居数: {len(nbs)}")
    for i, n in enumerate(nbs[:20]):
        print(f"    [{i}] {n}")
    if len(nbs) > 20:
        print(f"    ... (共 {len(nbs)} 个邻居)")


def cmd_stats(day_idx, feat_files, pid_to_idx):
    if day_idx < 0 or day_idx >= len(feat_files):
        print(f"  日索引 {day_idx} 超出范围 [0, {len(feat_files) - 1}]")
        return
    arr = np.load(feat_files[day_idx], mmap_mode='r')
    n_extract = len(pid_to_idx)
    data = arr[:n_extract]
    print(f"  形状: {arr.shape}")
    print(f"  均值: {np.mean(data):.4f}")
    print(f"  标准差: {np.std(data):.4f}")
    print(f"  最小值: {np.min(data):.4f}  |  最大值: {np.max(data):.4f}")
    print(f"  全零行数: {np.sum(np.all(data == 0, axis=1)):,}")
    print(f"  含 NaN 行数: {np.sum(np.any(np.isnan(data), axis=1)):,}")
    # 按属性列统计全零比例
    zero_pct = np.mean(data == 0, axis=0) * 100
    high_zero = [(N_COLS[i], zero_pct[i]) for i in range(len(N_COLS)) if zero_pct[i] > 10]
    if high_zero:
        print(f"  高全零率属性 (>10%):")
        for name, pct in sorted(high_zero, key=lambda x: -x[1])[:10]:
            print(f"    {name}: {pct:.1f}%")


def cmd_show(disk_idx, day_idx, n_attrs, feat_files, n_extract):
    if day_idx < 0 or day_idx >= len(feat_files):
        print(f"  日索引 {day_idx} 超出范围 [0, {len(feat_files) - 1}]")
        return
    if disk_idx < 0 or disk_idx >= n_extract:
        print(f"  磁盘索引 {disk_idx} 超出范围 [0, {n_extract - 1}]")
        return

    arr = np.load(feat_files[day_idx], mmap_mode='r')
    row = arr[disk_idx]
    n = n_attrs if n_attrs else len(N_COLS)
    n = min(n, len(N_COLS))

    print(f"  disk_idx={disk_idx}  day_idx={day_idx}")
    print(f"  全零: {np.all(row == 0)}") if np.all(row == 0) else None
    for i in range(n):
        print(f"    {N_COLS[i]:>6s}: {row[i]:.6g}")


def cmd_seq(disk_idx, di_start, di_end, feat_files, n_extract):
    n_dates = len(feat_files)
    if di_start < 0 or di_end >= n_dates or di_start > di_end:
        print(f"  日索引范围 [{di_start}, {di_end}] 超出 [0, {n_dates - 1}]")
        return
    if disk_idx < 0 or disk_idx >= n_extract:
        print(f"  磁盘索引 {disk_idx} 超出范围 [0, {n_extract - 1}]")
        return

    # 加载该范围内所有天的数据
    days_to_load = list(range(di_start, di_end + 1))
    seq_rows = []
    for di in days_to_load:
        arr = np.load(feat_files[di], mmap_mode='r')
        seq_rows.append(arr[disk_idx].copy())

    data = np.stack(seq_rows, axis=0)  # [n_days, FEAT_DIM]
    print(f"  disk_idx={disk_idx}  day_range=[{di_start}, {di_end}]  →  shape={data.shape}")
    print(f"  全零行: {[di_start + i for i in range(data.shape[0]) if np.all(data[i] == 0)]}")
    # 打印前 5 个属性的变化序列（用首尾对比）
    preview_attrs = min(5, len(N_COLS))
    print(f"  {'day':>4s} | " + " | ".join(f"{N_COLS[i]:>6s}" for i in range(preview_attrs)))
    for di_idx, di in enumerate(days_to_load):
        vals = " | ".join(f"{data[di_idx, i]:6.1f}" for i in range(preview_attrs))
        print(f"  {di:4d} | {vals}")


# ============================================================
# 3. 主循环
# ============================================================

def main():
    # 构建或加载索引
    if not os.path.exists(PID_INDEX_FILE):
        print("[inspect_data] 首次运行，构建 pid_index.csv...")
        disk_info, neighbor_map, pid_to_idx, feat_files = build_pid_index()
    else:
        print("[inspect_data] 从缓存加载 pid_index.csv...")
        idx_df = pd.read_csv(PID_INDEX_FILE)
        pid_to_idx = dict(zip(idx_df['pair_id'].astype(str), idx_df['idx'].astype(int)))
        feat_files = _find_feat_files()
        # 懒加载 disk_info（按需）
        disk_info = None
        neighbor_map = None

    n_extract = len(pid_to_idx)
    n_dates = len(feat_files)

    print(f"\n  磁盘数: {n_extract:,}  天数: {n_dates}  按天文件: {n_dates}")
    print("  输入 'help' 查看命令列表\n")

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  退出")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ('quit', 'exit'):
            break

        elif cmd == 'help':
            print_help()

        elif cmd == 'dates':
            cmd_dates(feat_files)

        elif cmd == 'disks':
            print(f"  磁盘总数: {n_extract:,}")

        elif cmd == 'disk':
            if len(parts) < 2:
                print("  用法: disk <pair_id>")
                continue
            if disk_info is None:
                disk_info = _lazy_load_disk_info()
            cmd_disk(parts[1], pid_to_idx, disk_info)

        elif cmd == 'neighbors':
            if len(parts) < 2:
                print("  用法: neighbors <pair_id>")
                continue
            if neighbor_map is None:
                neighbor_map = _lazy_load_neighbors()
            cmd_neighbors(parts[1], neighbor_map)

        elif cmd == 'stats':
            try:
                day_idx = int(parts[1])
            except (IndexError, ValueError):
                print("  用法: stats <day_idx>")
                continue
            cmd_stats(day_idx, feat_files, pid_to_idx)

        elif cmd == 'show':
            try:
                disk_idx = int(parts[1])
                day_idx = int(parts[2])
                n_attrs = int(parts[3]) if len(parts) > 3 else None
            except (IndexError, ValueError):
                print("  用法: show <disk_idx> <day_idx> [n_attrs]")
                continue
            cmd_show(disk_idx, day_idx, n_attrs, feat_files, n_extract)

        elif cmd == 'seq':
            try:
                disk_idx = int(parts[1])
                di_start = int(parts[2])
                di_end = int(parts[3])
            except (IndexError, ValueError):
                print("  用法: seq <disk_idx> <di_start> <di_end>")
                continue
            cmd_seq(disk_idx, di_start, di_end, feat_files, n_extract)

        else:
            print(f"  未知命令: '{cmd}'。输入 'help' 查看帮助")


def _lazy_load_disk_info():
    target_df = pd.read_csv(TARGET_FILE)
    target_df['failure_time'] = pd.to_datetime(target_df['failure_time'])
    disk_info = {}
    for _, row in target_df.iterrows():
        pid = str(row['pair_id'])
        disk_info[pid] = {
            'is_failure': bool(row['is_failure']),
            'failure_time': row['failure_time'] if pd.notna(row['failure_time']) else None,
            'node_id': int(row['node_id']) if pd.notna(row['node_id']) and row['node_id'] >= 0 else None
        }
    return disk_info


def _lazy_load_neighbors():
    if not os.path.exists(NEIGHBOR_MAP_FILE):
        return {}
    df = pd.read_csv(NEIGHBOR_MAP_FILE)
    df['pair_id'] = df['pair_id'].astype(str)
    df['neighbor_pair_id'] = df['neighbor_pair_id'].astype(str)
    return df.groupby('pair_id')['neighbor_pair_id'].apply(list).to_dict()


if __name__ == "__main__":
    main()