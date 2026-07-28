# data_utils.py
# 样本生成器：时间划分 → 滑动窗口 + TPS 打标签 → 加载邻居 → 提取 SMART → 分片保存
#
# 流程：
#   1. 加载 target_disks.csv（磁盘清单）和 neighbor_map.csv（邻居关系长表）
#   2. 扫描日期，按 TRAIN_CUTOFF 划分训练/测试窗口
#   3. 滑动窗口 + TPS 为每条 (disk, window) 打标签
#      - 故障盘 + 窗口结束日到故障日 ∈ [1, L]      → 正样本 (训练时 TPS 扩增)
#      - 故障盘 + 窗口结束日到故障日 ∈ [1, TEST_LEAD_TIME] → 正样本 (测试时，每盘随机 1 条)
#      - 健康盘                                    → 负样本（每盘随机选 1 条）
#   4. 按天分片从原始 CSV 提取磁盘及邻居的 SMART 特征（避免单一巨大 memmap）
#   5. FeatStore 按需加载 → 构建 (自身, 邻居, mask, label) → 分片压缩为 .npz

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import os
import gc
import shutil
from collections import defaultdict, OrderedDict
from config import *

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")
NEIGHBOR_MAP_FILE = os.path.join(PROCESSED_DIR, "neighbor_map.csv")
TRAIN_SHARD_PATTERN = os.path.join(PROCESSED_DIR, "train_shard_{:02d}.npz")
TEST_SHARD_PATTERN = os.path.join(PROCESSED_DIR, "test_shard_{:02d}.npz")

N_COLS = [f"n_{sid}" for sid in [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 175,
    177, 180, 181, 182, 183, 184, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 204, 205, 206, 207, 211,
    232, 233, 240, 241, 242, 244, 245
]]

RNG = np.random.RandomState(42)


# ============================================================
# 1. 日期扫描 & 窗口生成
# ============================================================

def _scan_csv_dates():
    year = 2018
    month_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith(str(year))
        and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END
    ])
    date_to_file = {}
    dates = []
    for fname in month_files:
        date_str = fname[:8]
        dates.append(date_str)
        date_to_file[date_str] = os.path.join(DATA_DIR, fname)
    return sorted(dates), date_to_file


def _build_window_list(dates):
    windows = []
    for i in range(SEQ_LEN - 1, len(dates)):
        w_idx = list(range(i - SEQ_LEN + 1, i + 1))
        windows.append((i, w_idx, dates[i]))
    return windows


# ============================================================
# 2. 磁盘清单 & 邻居加载
# ============================================================

def _load_disk_info():
    print("  读取 target_disks.csv...")
    target_df = pd.read_csv(TARGET_FILE)
    target_df['failure_time'] = pd.to_datetime(target_df['failure_time'])
    disk_info = {}
    for _, row in target_df.iterrows():
        pid = str(row['pair_id'])
        node_id = int(row['node_id']) if pd.notna(row['node_id']) and row['node_id'] >= 0 else None
        disk_info[pid] = {
            'model': str(row['model']),
            'is_failure': bool(row['is_failure']),
            'failure_time': row['failure_time'] if pd.notna(row['failure_time']) else None,
            'node_id': node_id
        }
    return disk_info


def _load_neighbor_map(disk_info):
    print("  读取 neighbor_map.csv...")
    if not os.path.exists(NEIGHBOR_MAP_FILE):
        raise FileNotFoundError(
            f"邻居映射文件不存在: {NEIGHBOR_MAP_FILE}\n"
            f"请先运行 build_neighbors.py 生成该文件"
        )
    neighbor_df = pd.read_csv(NEIGHBOR_MAP_FILE)
    neighbor_df['pair_id'] = neighbor_df['pair_id'].astype(str)
    neighbor_df['neighbor_pair_id'] = neighbor_df['neighbor_pair_id'].astype(str)

    neighbor_map = (
        neighbor_df.groupby('pair_id')['neighbor_pair_id']
        .apply(list)
        .to_dict()
    )
    for pid in disk_info:
        if pid not in neighbor_map:
            neighbor_map[pid] = []

    n_with = sum(1 for v in neighbor_map.values() if len(v) > 0)
    n_no = len(disk_info) - n_with
    max_n = max((len(v) for v in neighbor_map.values()), default=0)
    print(f"    有邻居: {n_with:,} | 无邻居: {n_no:,}")
    print(f"    最大邻居数: {max_n} | MAX_NEIGHBORS 截断: {MAX_NEIGHBORS}")
    return neighbor_map


def _get_all_pids(disk_info):
    all_pids = sorted(disk_info.keys())
    fail_pids = [p for p in all_pids if disk_info[p]['is_failure']]
    healthy_pids = [p for p in all_pids if not disk_info[p]['is_failure']]

    n_all = len(all_pids)
    n_fail = len(fail_pids)

    if MAX_DISKS > 0 and n_all > MAX_DISKS:
        n_healthy_sample = max(0, MAX_DISKS - n_fail)
        RNG.shuffle(healthy_pids)
        sampled = fail_pids + healthy_pids[:n_healthy_sample]
        print(f"  磁盘采样: 故障 {n_fail:,} (全保留) + 健康 {n_healthy_sample:,} = {len(sampled):,}")
        return sorted(sampled)

    print(f"  全量模式: {n_all:,} 个磁盘全部参与 (故障: {n_fail:,})")
    return all_pids


# ============================================================
# 3. 按天分片提取 SMART 特征（替代原始三维 memmap）
# ============================================================

def _extract_and_build_feat(disk_info, sampled_pids, neighbor_map):
    """
    按天分片保存 SMART 特征，每天一个 feat_day_{di:04d}.npy, shape (n_extract, FEAT_DIM)。
    避免单一巨大 memmap 导致的内存压力。
    返回: dates, extract_pids, pid_to_extract_idx, feat_files
    """
    dates, date_to_file = _scan_csv_dates()
    n_dates = len(dates)

    # 确保输出目录存在
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 收集所有需要提取的盘（目标盘 + 邻居截断到 MAX_NEIGHBORS）
    all_needed = set(sampled_pids)
    for pid in sampled_pids:
        all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
    extract_pids = sorted(all_needed)
    pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
    n_extract = len(extract_pids)

    total_gb = n_extract * n_dates * FEAT_DIM * 4 / 1024**3
    print(f"  需提取盘数: {n_extract:,}, 日期数: {n_dates}")
    print(f"  按天分片总大小: {total_gb:.1f} GB (每天 {total_gb / n_dates:.2f} GB)")

    # 磁盘空间预检
    free_gb = shutil.disk_usage(PROCESSED_DIR).free / 1024**3
    if total_gb > free_gb * 0.9:
        raise RuntimeError(
            f"磁盘空间不足！需要 {total_gb:.1f} GB，剩余 {free_gb:.1f} GB。\n"
            f"建议减小 MAX_DISKS 或 MAX_NEIGHBORS"
        )

    # 预分配每天的全零文件
    feat_files = []
    for di in range(n_dates):
        fpath = os.path.join(PROCESSED_DIR, f"feat_day_{di:04d}.npy")
        feat_files.append(fpath)
        np.save(fpath, np.zeros((n_extract, FEAT_DIM), dtype=np.float32))

    # 构建查找表 DataFrame（用于 pandas merge 向量化匹配）
    lookup_rows = []
    for pid in extract_pids:
        parts = pid.split('_', 1)
        if len(parts) == 2:
            lookup_rows.append({
                'disk_id': int(parts[0]),
                'model': parts[1],
                '_idx': pid_to_extract_idx[pid]
            })
    lookup_df = pd.DataFrame(lookup_rows)
    lookup_df['disk_id'] = lookup_df['disk_id'].astype(int)
    lookup_df['model'] = lookup_df['model'].astype(str)

    usecols = ['disk_id', 'model'] + N_COLS

    # 逐天向量化提取
    print(f"  逐天提取 SMART 数据 (pandas merge 向量化)...")
    for di, date_str in enumerate(dates):
        fpath_csv = date_to_file[date_str]

        df = pd.read_csv(fpath_csv, usecols=usecols)
        df['disk_id'] = df['disk_id'].astype(int)
        df['model'] = df['model'].astype(str)
        df[N_COLS] = df[N_COLS].fillna(0).astype(np.float32)

        # 向量化 merge：只保留目标盘，C 层执行
        merged = df.merge(lookup_df, on=['disk_id', 'model'], how='inner')

        n_matched = len(merged)
        if n_matched > 0:
            day_arr = np.load(feat_files[di], mmap_mode='r+')
            day_arr[merged['_idx'].values] = merged[N_COLS].values
            day_arr.flush()
            del day_arr

        del df, merged
        gc.collect()

        if (di + 1) % 5 == 0 or di == 0:
            print(f"    {date_str}: {n_matched:,} 条匹配写入")

    print(f"  特征提取完成（按天分片 {n_dates} 个文件）")
    return dates, extract_pids, pid_to_extract_idx, feat_files


# ============================================================
# 4. FeatStore：按需加载按天分片特征，带 LRU 缓存
# ============================================================

class FeatStore:
    """
    按需加载按天分片的特征文件，使用 OrderedDict 实现 O(1) LRU 缓存。
    替代原来的 feat_tensor[pid_idx, date_indices, :] 三维索引。
    """
    def __init__(self, feat_files, pid_to_extract_idx, max_cache=30):
        self.feat_files = feat_files
        self.pid_to_extract_idx = pid_to_extract_idx
        self.cache = OrderedDict()   # LRU: 最近使用的在末尾
        self.max_cache = max_cache

    def get(self, pid, date_indices):
        """获取某盘在 date_indices 上的特征序列。返回 (len(date_indices), FEAT_DIM) 或 None"""
        idx = self.pid_to_extract_idx.get(pid)
        if idx is None:
            return None
        res = []
        for di in date_indices:
            arr = self._load_day(di)
            res.append(arr[idx])
        return np.stack(res, axis=0)

    def _load_day(self, di):
        if di not in self.cache:
            self.cache[di] = np.load(self.feat_files[di], mmap_mode='r')
            # LRU 淘汰最久未使用的
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        else:
            # 移动到末尾（标记为最近使用）
            self.cache.move_to_end(di)
        return self.cache[di]

    def clear(self):
        self.cache.clear()


# ============================================================
# 5. 样本生成 & 分片保存
# ============================================================

def _generate_and_save_samples(dates, disk_info, sampled_pids, neighbor_map,
                                extract_pids, pid_to_extract_idx, feat_files):
    windows = _build_window_list(dates)
    print(f"  窗口数: {len(windows)} (训练截止: {TRAIN_CUTOFF})")

    feat_store = FeatStore(feat_files, pid_to_extract_idx, max_cache=30)

    # ====== 第一遍扫描：per-disk 收集候选窗口 ======
    print("  第一遍扫描：收集候选窗口（per-disk）...")
    train_pos_candidates = defaultdict(list)
    test_pos_candidates = defaultdict(list)
    train_neg_candidates = defaultdict(list)
    test_neg_candidates = defaultdict(list)

    for wi, (w_end, w_idx, date_str) in enumerate(windows):
        window_end_dt = pd.to_datetime(date_str, format="%Y%m%d")
        is_train = date_str <= TRAIN_CUTOFF

        for pi, pid in enumerate(sampled_pids):
            # 检查盘是否在提取清单中（不访问 feat 数据）
            if pid not in pid_to_extract_idx:
                continue

            info = disk_info[pid]

            if info['is_failure'] and info['failure_time'] is not None:
                days = (info['failure_time'] - window_end_dt).days
                if is_train:
                    if 1 <= days <= L:
                        train_pos_candidates[pi].append((wi, pi, 1.0))
                else:
                    if 1 <= days <= TEST_LEAD_TIME:
                        test_pos_candidates[pi].append((wi, pi, 1.0))
            else:
                if is_train:
                    train_neg_candidates[pi].append((wi, pi, 0.0))
                else:
                    test_neg_candidates[pi].append((wi, pi, 0.0))

    # ====== 第二遍：per-disk 随机选取 ======
    print("  第二遍：per-disk 随机选取...")

    train_pos_entries = []
    for pi, cands in train_pos_candidates.items():
        train_pos_entries.extend(cands)

    test_pos_entries = []
    for pi, cands in test_pos_candidates.items():
        chosen = cands[RNG.randint(0, len(cands))]
        test_pos_entries.append(chosen)

    train_neg_entries = []
    for pi, cands in train_neg_candidates.items():
        chosen = cands[RNG.randint(0, len(cands))]
        train_neg_entries.append(chosen)

    test_neg_entries = []
    for pi, cands in test_neg_candidates.items():
        chosen = cands[RNG.randint(0, len(cands))]
        test_neg_entries.append(chosen)

    n_train_pos = len(train_pos_entries)
    n_train_neg = len(train_neg_entries)
    n_test_pos = len(test_pos_entries)
    n_test_neg = len(test_neg_entries)

    n_fail_disks = len(train_pos_candidates)
    n_healthy_disks = len(train_neg_candidates)
    print(f"  故障盘: {n_fail_disks:,} | 健康盘: {n_healthy_disks:,}")
    print(f"  训练集: 正 {n_train_pos:,} (TPS L={L}) | 负 {n_train_neg:,} "
          f"| 正负比 1:{n_train_neg / max(n_train_pos, 1):.0f}")
    print(f"  测试集: 正 {n_test_pos:,} (每盘 1 条) | 负 {n_test_neg:,} "
          f"| 正负比 1:{n_test_neg / max(n_test_pos, 1):.0f}")

    # ====== 训练集：合并 + 打乱 ======
    print(f"\n  训练集: 正样本全保留（TPS L={L}），负样本每盘 1 条")
    train_selected = train_pos_entries + train_neg_entries
    RNG.shuffle(train_selected)
    n_train_final = len(train_selected)
    print(f"  训练样本总计: {n_train_final:,} (正: {n_train_pos:,}, 负: {n_train_neg:,})")

    # ====== 测试集：全量保留 ======
    test_all = test_pos_entries + test_neg_entries
    RNG.shuffle(test_all)
    if MAX_TEST_SAMPLES > 0 and len(test_all) > MAX_TEST_SAMPLES:
        test_selected = test_all[:MAX_TEST_SAMPLES]
        n_test_final = len(test_selected)
        n_test_pos_final = sum(1 for _, _, label in test_selected if label == 1.0)
        print(f"\n  测试集: 随机采样至 {MAX_TEST_SAMPLES:,} (正: {n_test_pos_final})")
    else:
        test_selected = test_all
        n_test_final = len(test_selected)
        n_test_pos_final = n_test_pos
        print(f"\n  测试集: 全量保留 {n_test_final:,} (正: {n_test_pos_final})")

    # ====== 辅助函数 === ==
    def _copy_sample(wi, pi, label, s_tgt, n_tgt, m_tgt, l_tgt, counter):
        pid = sampled_pids[pi]
        w_end, w_idx, date_str = windows[wi]

        disk_seq = feat_store.get(pid, w_idx)
        if disk_seq is None:
            return counter

        neighbors = neighbor_map.get(pid, [])[:MAX_NEIGHBORS]
        neigh_seq_arr = np.zeros((MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM), dtype=np.float32)
        neigh_mask_arr = np.zeros(MAX_NEIGHBORS, dtype=np.bool_)
        for j, npid in enumerate(neighbors):
            nseq = feat_store.get(npid, w_idx)
            if nseq is None:
                continue
            if np.isnan(nseq).all() or np.all(nseq == 0):
                continue
            neigh_seq_arr[j] = nseq
            neigh_mask_arr[j] = True

        s_tgt[counter] = disk_seq
        n_tgt[counter] = neigh_seq_arr
        m_tgt[counter] = neigh_mask_arr
        l_tgt[counter] = label
        return counter + 1

    def _save_shards(source_indices, shard_pattern, n_shards, label_prefix):
        n_total = len(source_indices)
        if n_total == 0:
            print(f"    {label_prefix}: 0 样本，跳过")
            return 0

        shard_size = (n_total + n_shards - 1) // n_shards
        n_actual = min(n_shards, (n_total + shard_size - 1) // shard_size)

        shard_groups = [[] for _ in range(n_actual)]
        for idx, item in enumerate(source_indices):
            s = idx // shard_size
            if s >= n_actual:
                s = n_actual - 1
            shard_groups[s].append(item)

        for s in range(n_actual):
            group = shard_groups[s]
            n_shard = len(group)
            if n_shard == 0:
                continue

            tmp_prefix = shard_pattern.format(s)
            s_tmp = np.memmap(tmp_prefix + '.s.tmp', dtype=np.float32, mode='w+',
                              shape=(n_shard, SEQ_LEN, FEAT_DIM))
            n_tmp = np.memmap(tmp_prefix + '.n.tmp', dtype=np.float32, mode='w+',
                              shape=(n_shard, MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM))
            m_tmp = np.memmap(tmp_prefix + '.m.tmp', dtype=np.bool_, mode='w+',
                              shape=(n_shard, MAX_NEIGHBORS))
            l_tmp = np.memmap(tmp_prefix + '.l.tmp', dtype=np.float32, mode='w+',
                              shape=(n_shard, 1))

            for j, (wi, pi, label) in enumerate(group):
                _copy_sample(wi, pi, label, s_tmp, n_tmp, m_tmp, l_tmp, j)

            np.savez_compressed(shard_pattern.format(s),
                                s=np.array(s_tmp[:n_shard]),
                                n=np.array(n_tmp[:n_shard]),
                                m=np.array(m_tmp[:n_shard]),
                                l=np.array(l_tmp[:n_shard]))

            n_pos_s = int(l_tmp[:n_shard].sum())
            sp = shard_pattern.format(s)
            print(f"      {label_prefix}_shard_{s:02d}: {n_shard:,} "
                  f"(正:{n_pos_s}, 负:{n_shard - n_pos_s}) "
                  f"→ {os.path.getsize(sp) / 1024**2:.0f} MB")

            del s_tmp, n_tmp, m_tmp, l_tmp
            for ext in ['.s.tmp', '.n.tmp', '.m.tmp', '.l.tmp']:
                try:
                    os.remove(tmp_prefix + ext)
                except:
                    pass

        return n_actual

    print("\n  保存训练分片...")
    n_train_shards = _save_shards(train_selected, TRAIN_SHARD_PATTERN, TRAIN_SHARDS, "train")

    print("  保存测试分片...")
    n_test_shards = _save_shards(test_selected, TEST_SHARD_PATTERN, TEST_SHARDS, "test")

    feat_store.clear()
    gc.collect()

    if n_train_shards == 0:
        raise RuntimeError("没有生成任何训练样本！请检查数据或配置")

    return n_train_shards, n_test_shards


# ============================================================
# 6. 主入口
# ============================================================

def build_and_save_samples():
    print("=" * 60)
    print("[Phase 1] 构建样本数据集")

    disk_info = _load_disk_info()
    neighbor_map = _load_neighbor_map(disk_info)
    sampled_pids = _get_all_pids(disk_info)

    dates, extract_pids, pid_to_extract_idx, feat_files = _extract_and_build_feat(
        disk_info, sampled_pids, neighbor_map)

    n_train, n_test = _generate_and_save_samples(
        dates, disk_info, sampled_pids, neighbor_map,
        extract_pids, pid_to_extract_idx, feat_files)

    print("=" * 60)
    return n_train, n_test


def _numpy_to_dataloader(s_arr, n_arr, m_arr, l_arr, shuffle, batch_size):
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.asarray(s_arr)),
        torch.FloatTensor(np.asarray(n_arr)),
        torch.BoolTensor(np.asarray(m_arr)),
        torch.FloatTensor(np.asarray(l_arr)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_data():
    if not os.path.exists(TRAIN_SHARD_PATTERN.format(0)) and not os.path.exists(TEST_SHARD_PATTERN.format(0)):
        return build_and_save_samples()
    else:
        print("\n[data_utils] 找到已有分片文件，跳过构建...")
        n_train = get_num_train_shards()
        n_test = get_num_test_shards()
        print(f"  训练分片: {n_train}, 测试分片: {n_test}")
        return n_train, n_test


def load_train_shard(shard_id):
    shard_path = TRAIN_SHARD_PATTERN.format(shard_id)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"训练分片不存在: {shard_path}")
    data = np.load(shard_path)
    n_samples = len(data['l'])
    loader = _numpy_to_dataloader(np.asarray(data['s']), np.asarray(data['n']),
                                  np.asarray(data['m']), np.asarray(data['l']),
                                  shuffle=True, batch_size=BATCH_SIZE)
    print(f"  [Train Shard {shard_id:02d}] {n_samples} 样本, {len(loader)} batches")
    return loader


def load_test_shard(shard_id):
    shard_path = TEST_SHARD_PATTERN.format(shard_id)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"测试分片不存在: {shard_path}")
    data = np.load(shard_path)
    n_samples = len(data['l'])
    loader = _numpy_to_dataloader(np.asarray(data['s']), np.asarray(data['n']),
                                  np.asarray(data['m']), np.asarray(data['l']),
                                  shuffle=False, batch_size=BATCH_SIZE * 4)
    print(f"  [Test Shard {shard_id:02d}] {n_samples} 样本, {len(loader)} batches")
    return loader


def get_num_train_shards():
    count = 0
    for shard_id in range(TRAIN_SHARDS):
        if os.path.exists(TRAIN_SHARD_PATTERN.format(shard_id)):
            count += 1
        else:
            break
    return count


def get_num_test_shards():
    count = 0
    for shard_id in range(TEST_SHARDS):
        if os.path.exists(TEST_SHARD_PATTERN.format(shard_id)):
            count += 1
        else:
            break
    return count