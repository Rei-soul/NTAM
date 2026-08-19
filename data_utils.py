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
import glob
import json
from collections import defaultdict, OrderedDict
from config import *

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")
NEIGHBOR_MAP_FILE = os.path.join(PROCESSED_DIR, "neighbor_map.csv")
TRAIN_SHARD_PATTERN = os.path.join(PROCESSED_DIR, "train_shard_{:02d}.npz")
TEST_SHARD_PATTERN = os.path.join(PROCESSED_DIR, "test_shard_{:02d}.npz")
# ========== 分片切分 ==========
# 分片数由 config 的 TRAIN_SHARDS / TEST_SHARDS 决定（均匀切分）。
# 改 TRAIN_SHARDS / TEST_SHARDS 后需手动删除 datasets/processed 下对应 *_shard_*.npz 再重建。


# r_ 原始值列，经 Z-score 按 model 标准化后使用（由 build_feat_r.py 生成）
# 30 列 (30/3=10 整除 NUM_HEADS)
N_COLS = [f"r_{sid}" for sid in [
    5, 9, 12,
    170, 171, 172, 173, 174, 175,
    177,
    180, 181, 182, 183, 184,
    187, 188,
    190, 192, 194, 195, 196, 197, 198, 199,
    206,
    232, 233, 241, 242
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
    # 默认下MAX_DISKS=0,是全量模式
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
    支持 col_idx_map：当旧 feat_day 维度（如51）大于 FEAT_DIM（30）时自动切片。
    """
    def __init__(self, feat_files, pid_to_extract_idx, max_cache=30, col_idx_map=None):
        self.feat_files = feat_files
        self.pid_to_extract_idx = pid_to_extract_idx
        self.cache = OrderedDict()   # LRU: 最近使用的在末尾
        self.max_cache = max_cache
        self.col_idx_map = col_idx_map  # None 表示不需要切片

    def get(self, pid, date_indices):
        """获取某盘在 date_indices 上的特征序列。返回 (len(date_indices), FEAT_DIM) 或 None"""
        idx = self.pid_to_extract_idx.get(pid)
        if idx is None:
            return None
        res = []
        for di in date_indices:
            arr = self._load_day(di)
            row = arr[idx]
            if self.col_idx_map is not None:
                row = row[self.col_idx_map]  # (51,) → (30,)
            res.append(row)
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
                                extract_pids, pid_to_extract_idx, feat_files,
                                col_idx_map=None, sets=('train', 'test')):
    """
    论文 per-disk 锚定方式生成样本（非滑动窗口扫描）。

    故障盘（训练）: failure_time 前 l=1..L 天为窗口结束日，取前 h 天 → L 条正样本
    故障盘（测试）: failure_time 前 l=1..TEST_LEAD_TIME 内随机选 1 天 → 1 条正样本
    健康盘:       从合法窗口中随机选 1 条负样本

    sets: 需要生成并保存的集合（'train'/'test' 的子集）。
          只重建部分集合时（如仅改 TEST_LEAD_TIME），其他集合的分片文件保持不变。
    col_idx_map: 若不为 None，FeatStore 读取旧文件后按此索引切片
    """
    need_train = 'train' in sets
    need_test = 'test' in sets

    n_dates = len(dates)
    # date_str → date_index，用于 failure_time 快速查找
    date_to_di = {d: i for i, d in enumerate(dates)}
    # 训练/测试可用的窗口结束日索引列表
    train_end_di_list = [i for i, d in enumerate(dates)
                         if i >= SEQ_LEN - 1 and TRAIN_START <= d <= TRAIN_END]
    test_end_di_list = [i for i, d in enumerate(dates)
                        if i >= SEQ_LEN - 1 and TEST_START <= d <= TEST_END]

    print(f"  日期数: {n_dates} | 训练窗口池: {len(train_end_di_list)} "
          f"({TRAIN_START}~{TRAIN_END}) | 测试窗口池: {len(test_end_di_list)} "
          f"({TEST_START}~{TEST_END})")

    feat_store = FeatStore(feat_files, pid_to_extract_idx, max_cache=30, col_idx_map=col_idx_map)

    # ====== Per-disk 生成样本 ======
    print("  Per-disk 锚定生成样本...")
    train_pos_entries = []   # (w_idx_list, pi, label)
    train_neg_entries = []
    test_pos_entries = []
    test_neg_entries = []

    n_fail_disks = 0
    n_healthy_disks = 0

    for pi, pid in enumerate(sampled_pids):
        if pid not in pid_to_extract_idx:
            continue

        info = disk_info[pid]

        if info['is_failure'] and info['failure_time'] is not None:
            n_fail_disks += 1
            ft_str = info['failure_time'].strftime("%Y%m%d")
            ft_di = date_to_di.get(ft_str)
            if ft_di is None:
                continue  # failure_time 不在数据日期范围内

            # === 训练集 TPS: l = 1..L ===
            if need_train:
                for l in range(1, L + 1):
                    end_di = ft_di - l
                    if end_di < SEQ_LEN - 1:
                        break  # 窗口太小
                    if TRAIN_START <= dates[end_di] <= TRAIN_END:
                        w_idx = list(range(end_di - SEQ_LEN + 1, end_di + 1))
                        train_pos_entries.append((w_idx, pi, 1.0))

            # === 测试集: l=1..TEST_LEAD_TIME 中随机选 1 条 ===
            if need_test:
                test_cands = []
                for l in range(1, TEST_LEAD_TIME + 1):
                    end_di = ft_di - l
                    if end_di < SEQ_LEN - 1:
                        break
                    if TEST_START <= dates[end_di] <= TEST_END:
                        w_idx = list(range(end_di - SEQ_LEN + 1, end_di + 1))
                        test_cands.append(w_idx)
                if test_cands:
                    chosen = test_cands[RNG.randint(0, len(test_cands))]
                    test_pos_entries.append((chosen, pi, 1.0))
        else:
            # === 健康盘：训练/测试各随机选 1 条负样本 ===
            n_healthy_disks += 1

            if need_train and train_end_di_list:
                end_di = train_end_di_list[RNG.randint(0, len(train_end_di_list))]
                w_idx = list(range(end_di - SEQ_LEN + 1, end_di + 1))
                train_neg_entries.append((w_idx, pi, 0.0))

            if need_test and test_end_di_list:
                end_di = test_end_di_list[RNG.randint(0, len(test_end_di_list))]
                w_idx = list(range(end_di - SEQ_LEN + 1, end_di + 1))
                test_neg_entries.append((w_idx, pi, 0.0))

    n_train_pos = len(train_pos_entries)
    n_train_neg = len(train_neg_entries)
    n_test_pos = len(test_pos_entries)
    n_test_neg = len(test_neg_entries)

    print(f'  故障盘: {n_fail_disks:,} | 健康盘: {n_healthy_disks:,}')
    if need_train:
        print(f'  训练集: 正 {n_train_pos:,} (TPS L={L}) | 负 {n_train_neg:,} | 正负比 1:{n_train_neg / max(n_train_pos, 1):.0f}')
    if need_test:
        print(f'  测试集: 正 {n_test_pos:,} (每盘 1 条) | 负 {n_test_neg:,} | 正负比 1:{n_test_neg / max(n_test_pos, 1):.0f}')

    # ====== 辅助函数 ======
    def _copy_sample(w_idx, pi, label, s_tgt, n_tgt, m_tgt, l_tgt, counter):
        pid = sampled_pids[pi]
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

    def _save_shards(source_entries, shard_pattern, n_shards, label_prefix):
        """按 n_shards 均匀切分（片数由 config 的 TRAIN_SHARDS/TEST_SHARDS 决定）。
        改片数后需手动删除 datasets/processed 下对应 *_shard_*.npz 再重建。"""
        n_total = len(source_entries)
        if n_total == 0:
            print(f'    {label_prefix}: 0 样本，跳过')
            return 0

        shard_size = (n_total + n_shards - 1) // n_shards
        n_actual = min(n_shards, (n_total + shard_size - 1) // shard_size)

        print(f'    {label_prefix}: 共 {n_total:,} 样本，目标 {n_shards} 片 -> 实际 {n_actual} 片（每片约 {shard_size:,}）')

        for s in range(n_actual):
            group = source_entries[s * shard_size:(s + 1) * shard_size]
            n_shard = len(group)
            if n_shard == 0:
                continue
            tmp_prefix = shard_pattern.format(s)
            s_tmp = np.memmap(tmp_prefix + '.s.tmp', dtype=np.float32, mode='w+', shape=(n_shard, SEQ_LEN, FEAT_DIM))
            n_tmp = np.memmap(tmp_prefix + '.n.tmp', dtype=np.float32, mode='w+', shape=(n_shard, MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM))
            m_tmp = np.memmap(tmp_prefix + '.m.tmp', dtype=np.bool_, mode='w+', shape=(n_shard, MAX_NEIGHBORS))
            l_tmp = np.memmap(tmp_prefix + '.l.tmp', dtype=np.float32, mode='w+', shape=(n_shard, 1))
            for j, (w_idx, pi, label) in enumerate(group):
                _copy_sample(w_idx, pi, label, s_tmp, n_tmp, m_tmp, l_tmp, j)
            np.savez_compressed(shard_pattern.format(s),
                                s=np.array(s_tmp[:n_shard]),
                                n=np.array(n_tmp[:n_shard]),
                                m=np.array(m_tmp[:n_shard]),
                                l=np.array(l_tmp[:n_shard]))
            n_pos_s = int(l_tmp[:n_shard].sum())
            sp = shard_pattern.format(s)
            print(f'      {label_prefix}_shard_{s:02d}: {n_shard:,} (正:{n_pos_s}, 负:{n_shard - n_pos_s}) -> {os.path.getsize(sp) / 1024**2:.0f} MB')
            del s_tmp, n_tmp, m_tmp, l_tmp
            for ext in ['.s.tmp', '.n.tmp', '.m.tmp', '.l.tmp']:
                try:
                    os.remove(tmp_prefix + ext)
                except:
                    pass
        return n_actual

    n_train_shards = n_test_shards = 0

    # ====== 合并 + 按窗口排序（利用 FeatStore 缓存局部性，大幅加速分片保存） ======
    if need_train:
        print(f'\n  训练集: 正样本全保留（TPS L={L}），负样本每盘 1 条')
        train_selected = train_pos_entries + train_neg_entries
        train_selected.sort(key=lambda x: x[0][0])
        n_train_final = len(train_selected)
        print(f'  训练样本总计: {n_train_final:,} (正: {n_train_pos:,}, 负: {n_train_neg:,})')
        print('\n  保存训练分片...')
        n_train_shards = _save_shards(train_selected, TRAIN_SHARD_PATTERN, TRAIN_SHARDS, 'train')

    if need_test:
        test_all = test_pos_entries + test_neg_entries
        test_all.sort(key=lambda x: x[0][0])
        if MAX_TEST_SAMPLES > 0 and len(test_all) > MAX_TEST_SAMPLES:
            pos_entries = [e for e in test_all if e[2] == 1.0]
            neg_entries = [e for e in test_all if e[2] == 0.0]
            # 保持原始正负比例的分层随机采样；原实现取前 N 条会让测试集偏向最早时间段
            n_pos_sel = int(round(MAX_TEST_SAMPLES * len(pos_entries) / max(len(test_all), 1)))
            n_pos_sel = min(len(pos_entries), max(0, n_pos_sel))
            n_neg_sel = MAX_TEST_SAMPLES - n_pos_sel
            n_neg_sel = min(len(neg_entries), max(0, n_neg_sel))
            pos_idx = RNG.choice(len(pos_entries), size=n_pos_sel, replace=False) if n_pos_sel else []
            neg_idx = RNG.choice(len(neg_entries), size=n_neg_sel, replace=False) if n_neg_sel else []
            test_selected = [pos_entries[i] for i in pos_idx] + [neg_entries[i] for i in neg_idx]
            test_selected.sort(key=lambda x: x[0][0])
            n_test_final = len(test_selected)
            n_test_pos_final = sum(1 for _, _, label in test_selected if label == 1.0)
            print(f'\n  测试集: 随机采样至 {MAX_TEST_SAMPLES:,} (正: {n_test_pos_final})')
        else:
            test_selected = test_all
            n_test_final = len(test_selected)
            n_test_pos_final = n_test_pos
            print(f'\n  测试集: 全量保留 {n_test_final:,} (正: {n_test_pos_final})')
        print('\n  保存测试分片...')
        n_test_shards = _save_shards(test_selected, TEST_SHARD_PATTERN, TEST_SHARDS, 'test')

    feat_store.clear()
    gc.collect()

    if need_train and n_train_shards == 0:
        raise RuntimeError('没有生成任何训练样本！请检查数据或配置')

    return n_train_shards, n_test_shards

def build_and_save_samples(sets=('train', 'test')):
    sets_str = '+'.join(sets)
    print('=' * 60)
    print(f"[Phase 1] 构建样本数据集 ({sets_str})")

    disk_info = _load_disk_info()
    neighbor_map = _load_neighbor_map(disk_info)
    sampled_pids = _get_all_pids(disk_info)

    # 检测 feat_day 文件是否已存在
    feat_day_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "feat_day_*.npy")))

    if feat_day_files:
        # 已有 feat_day 文件（由 build_feat_r.py 生成），跳过特征提取
        print(f"  检测到 {len(feat_day_files)} 个已有 feat_day_*.npy 文件，跳过特征提取")

        dates, _ = _scan_csv_dates()
        # 重建 pid_to_extract_idx（与 _extract_and_build_feat 中 sorted(all_needed) 一致）
        all_needed = set(sampled_pids)
        for pid in sampled_pids:
            all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
        extract_pids = sorted(all_needed)
        pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
        # 防止 feat_day 与当前 FEAT_DIM / MAX_NEIGHBORS / MAX_DISKS 不匹配导致静默错位
        _probe = np.load(feat_day_files[0], mmap_mode='r')
        if _probe.shape[1] != FEAT_DIM:
            raise RuntimeError(
                f"feat_day 特征维度 ({_probe.shape[1]}) 与当前 FEAT_DIM ({FEAT_DIM}) 不一致。"
                f"请删除 datasets/processed/feat_day_*.npy 和 r_stats.json 后重跑 build_feat_r.py")
        if _probe.shape[0] != len(extract_pids):
            raise RuntimeError(
                f"feat_day 磁盘行数 ({_probe.shape[0]}) 小于当前需要的磁盘数 ({len(extract_pids)})。"
                f"请删除 feat_day_*.npy 后重跑 build_feat_r.py")
        npy_date_count = len(feat_day_files)
        if npy_date_count != len(dates):
            print(f"  ⚠️ 警告: feat_day 文件数 ({npy_date_count}) 与日期数 ({len(dates)}) 不一致")
            print("     将使用 feat_day 文件数限制样本生成的日期范围")
            dates = dates[:npy_date_count]
        # feat_day 由 build_feat_r.py 标准化；训练范围变化时统计量也必须重建
        _r_stats_path = os.path.join(PROCESSED_DIR, "r_stats.json")
        if os.path.exists(_r_stats_path):
            try:
                with open(_r_stats_path, 'r', encoding='utf-8') as _f:
                    _r_meta = json.load(_f)
            except Exception:
                _r_meta = {}
            if (_r_meta.get('train_range') != [TRAIN_START, TRAIN_END]
                    or _r_meta.get('cols') != N_COLS):
                raise RuntimeError(
                    "r_stats.json 与当前 TRAIN_START/TRAIN_END 或特征列不一致。"
                    "请先运行 python code/build_feat_r.py 重建特征，再运行 train.py")
        else:
            print("  ⚠️ 未找到 r_stats.json：feat_day 可能是未标准化的原始值，"
                  "建议先运行 python code/build_feat_r.py")
    else:
        dates, extract_pids, pid_to_extract_idx, feat_day_files = _extract_and_build_feat(
            disk_info, sampled_pids, neighbor_map)

    n_train, n_test = _generate_and_save_samples(
        dates, disk_info, sampled_pids, neighbor_map,
        extract_pids, pid_to_extract_idx, feat_day_files, sets=sets)


    return n_train, n_test

def _numpy_to_dataloader(s_arr, n_arr, m_arr, l_arr, shuffle, batch_size):
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.asarray(s_arr)),
        torch.FloatTensor(np.asarray(n_arr)),
        torch.BoolTensor(np.asarray(m_arr)),
        torch.FloatTensor(np.asarray(l_arr)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _set_shards_exist(set_name):
    pattern = TRAIN_SHARD_PATTERN if set_name == 'train' else TEST_SHARD_PATTERN
    return os.path.exists(pattern.format(0))


def load_data():
    # 不自动重建：分片由用户手动管理。改任何 config 参数都不会触发重建，
    # 需要重建时手动删除 datasets/processed 下对应 *_shard_*.npz 文件即可。
    # 这里只检查分片是否存在，缺失的集合才构建。
    missing = []
    for s in ('train', 'test'):
        if not _set_shards_exist(s):
            missing.append(s)

    if missing:
        sets_str = '+'.join(missing)
        print(f"[data_utils] 检测到分片缺失: {sets_str}，开始构建...")
        build_and_save_samples(tuple(missing))
    else:
        print("[data_utils] 找到训练/测试分片文件，跳过构建...")

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


def get_train_shard_ids():
    """返回实际参与训练的分片 ID 列表。TRAIN_SHARD_IDS 非空时按它过滤，否则取前 TRAIN_SHARDS 片（0=全部）。"""
    if TRAIN_SHARD_IDS:
        ids = [i for i in TRAIN_SHARD_IDS if os.path.exists(TRAIN_SHARD_PATTERN.format(i))]
        if ids:
            return ids
    ids = []
    limit = TRAIN_SHARDS if TRAIN_SHARDS > 0 else float('inf')
    i = 0
    while i < limit and os.path.exists(TRAIN_SHARD_PATTERN.format(i)):
        ids.append(i)
        i += 1
    return ids

def get_test_shard_ids():
    """返回实际参与评估的测试分片 ID 列表。TEST_SHARD_IDS 非空时按它过滤，否则取前 TEST_SHARDS 片（0=全部）。"""
    if TEST_SHARD_IDS:
        ids = [i for i in TEST_SHARD_IDS if os.path.exists(TEST_SHARD_PATTERN.format(i))]
        if ids:
            return ids
    ids = []
    limit = TEST_SHARDS if TEST_SHARDS > 0 else float('inf')
    i = 0
    while i < limit and os.path.exists(TEST_SHARD_PATTERN.format(i)):
        ids.append(i)
        i += 1
    return ids

def get_num_train_shards():
    return len(get_train_shard_ids())


def get_num_test_shards():
    return len(get_test_shard_ids())