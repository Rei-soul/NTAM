# data_utils.py
# 样本生成器：时间划分 → 滑动窗口 + TPS 打标签 → 加载邻居 → 提取 SMART → 分片保存
#
# 流程：
#   1. 加载 target_disks.csv（磁盘清单）和 neighbor_map.csv（邻居关系长表）
#   2. 扫描日期，按 TRAIN_CUTOFF 划分训练/测试窗口
#   3. 滑动窗口 + TPS 为每条 (disk, window) 打标签
#      - 故障盘 + 窗口结束日到故障日 ∈ [1, L]      → 正样本 (训练)
#      - 故障盘 + 窗口结束日到故障日 ∈ [1, TEST_LEAD_TIME] → 正样本 (测试)
#      - 健康盘 + 所有窗口                           → 负样本（全保留，不下采样）
#   4. 按需从原始 CSV 提取磁盘及其邻居的 SMART 特征 → memmap
#   5. 构建 (自身特征, 邻居特征, mask, label) 四元组 → 分片压缩为 .npz
#
# 论文设计原则（Section 3.6 TPS）：
#   "TPS not only retains all the characteristics of healthy disks,
#    but also brings more failure patterns."
#   → 健康盘负样本全保留，不下采样；故障盘通过 TPS 扩增正样本

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import os
import gc
from collections import defaultdict
from config import *

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")
NEIGHBOR_MAP_FILE = os.path.join(PROCESSED_DIR, "neighbor_map.csv")
FEAT_NPY = os.path.join(PROCESSED_DIR, "feat_tensor.npy")
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
    """扫描原始 CSV 文件，返回有序日期列表和日期→文件路径映射"""
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
    """生成所有合法滑动窗口清单 [(窗口结束日索引, 窗口内日期索引列表, 窗口结束日期字符串), ...]"""
    windows = []
    for i in range(SEQ_LEN - 1, len(dates)):
        w_idx = list(range(i - SEQ_LEN + 1, i + 1))
        windows.append((i, w_idx, dates[i]))
    return windows


# ============================================================
# 2. 磁盘清单 & 邻居加载
# ============================================================

def _load_disk_info():
    """读取 target_disks.csv，返回 {pair_id: {model, is_failure, failure_time, node_id}}"""
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
    """
    从 neighbor_map.csv（长表格式）加载邻居映射。

    neighbor_map.csv 列:
        pair_id          - 当前磁盘唯一标识
        neighbor_pair_id - 邻居磁盘唯一标识
        node_id          - 共享的服务器节点 ID

    返回:
        neighbor_map: {pair_id: [neighbor_pair_id, ...]}
            邻居列表按原始 CSV 出现顺序排列（由 groupby 保证），
            在模型训练时按 MAX_NEIGHBORS 截断。
    """
    print("  读取 neighbor_map.csv...")
    if not os.path.exists(NEIGHBOR_MAP_FILE):
        raise FileNotFoundError(
            f"邻居映射文件不存在: {NEIGHBOR_MAP_FILE}\n"
            f"请先运行 build_neighbors.py 生成该文件"
        )

    neighbor_df = pd.read_csv(NEIGHBOR_MAP_FILE)
    neighbor_df['pair_id'] = neighbor_df['pair_id'].astype(str)
    neighbor_df['neighbor_pair_id'] = neighbor_df['neighbor_pair_id'].astype(str)

    # 长表 → {pair_id: [neighbor_pair_id, ...]}
    neighbor_map = (
        neighbor_df.groupby('pair_id')['neighbor_pair_id']
        .apply(list)
        .to_dict()
    )

    # 为所有磁盘初始化（无邻居的磁盘给空列表）
    for pid in disk_info:
        if pid not in neighbor_map:
            neighbor_map[pid] = []

    # 统计
    n_with_neighbors = sum(1 for v in neighbor_map.values() if len(v) > 0)
    n_no_neighbors = len(disk_info) - n_with_neighbors
    max_n = max((len(v) for v in neighbor_map.values()), default=0)
    print(f"    有邻居的磁盘: {n_with_neighbors:,} | 无邻居的磁盘: {n_no_neighbors:,}")
    print(f"    最大邻居数: {max_n} | MAX_NEIGHBORS 截断: {MAX_NEIGHBORS}")

    return neighbor_map


def _get_all_pids(disk_info):
    """全量模式：所有磁盘参与（论文设计：健康盘不下采样）"""
    all_pids = sorted(disk_info.keys())
    n_fail = sum(1 for p in all_pids if disk_info[p]['is_failure'])
    print(f"  全量模式: {len(all_pids):,} 个磁盘全部参与 (故障: {n_fail:,})")
    return all_pids


# ============================================================
# 3. SMART 特征提取（从原始 CSV → memmap）
# ============================================================

def _extract_and_build_feat(disk_info, sampled_pids, neighbor_map):
    """
    从原始 CSV 逐天提取目标磁盘及其邻居的 SMART 特征，保存为 memmap。

    输出:
        feat_tensor: memmap (n_extract, n_dates, FEAT_DIM)
        dates, extract_pids, pid_to_extract_idx
    """
    dates, date_to_file = _scan_csv_dates()
    print(f"  日期数: {len(dates)} 天")

    # 收集需要提取的所有磁盘（采样盘 + 邻居）
    all_needed = set(sampled_pids)
    for pid in sampled_pids:
        for npid in neighbor_map.get(pid, [])[:MAX_NEIGHBORS]:
            all_needed.add(npid)# 从邻居清单中获得所有邻居的pid
    extract_pids = sorted(all_needed)
    pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
    print(f"  需提取数据的盘: {len(extract_pids):,} "
          f"(采样 {len(sampled_pids):,} + 邻居)")

    n_extract = len(extract_pids)
    n_dates = len(dates)

    # 创建 memmap
    print(f"  创建 feat_tensor memmap ({n_extract}×{n_dates}×{FEAT_DIM})...")
    feat_tensor = np.memmap(FEAT_NPY, dtype=np.float32, mode='w+',
                            shape=(n_extract, n_dates, FEAT_DIM))

    # 构建 (disk_id, model) → memmap 索引映射
    usecols = ['disk_id', 'model'] + N_COLS
    extract_key_to_idx = {}
    for pid in extract_pids:
        parts = pid.split('_', 1)
        if len(parts) == 2:
            did, model = int(parts[0]), parts[1]
            extract_key_to_idx[(did, model)] = pid_to_extract_idx[pid]

    # 逐天分块提取（小 chunksize + 即时回收，控制内存峰值）
    print(f"  逐天提取 SMART 数据 (chunksize=20000)...")
    for di, date_str in enumerate(dates):
        fpath = date_to_file[date_str]
        count = 0
        total_rows = 0
        for chunk in pd.read_csv(fpath, usecols=usecols, chunksize=20000):
            chunk['disk_id'] = chunk['disk_id'].astype(int)
            chunk['model'] = chunk['model'].astype(str)
            chunk_vals = chunk[N_COLS].fillna(0).values.astype(np.float32)
            chunk_dids = chunk['disk_id'].values
            chunk_models = chunk['model'].values
            total_rows += len(chunk)
            for j in range(len(chunk)):
                key = (int(chunk_dids[j]), str(chunk_models[j]))
                idx = extract_key_to_idx.get(key)
                if idx is not None:
                    feat_tensor[idx, di] = chunk_vals[j]
                    count += 1
            # 每个 chunk 后立即释放临时变量
            del chunk, chunk_vals, chunk_dids, chunk_models
            gc.collect()
        # 每天处理完后 flush memmap，确保写入磁盘
        feat_tensor.flush()
        if (di + 1) % 5 == 0 or di == 0:
            print(f"    {date_str}: {count:,} 条匹配 / {total_rows:,} 行")

    feat_tensor.flush()
    print(f"  → 已保存: {FEAT_NPY} ({os.path.getsize(FEAT_NPY)/1024**3:.1f} GB)")
    return dates, extract_pids, pid_to_extract_idx


# ============================================================
# 4. 样本生成 & 分片保存
# ============================================================

def _generate_and_save_samples(dates, disk_info, sampled_pids, neighbor_map,
                                extract_pids, pid_to_extract_idx):

    windows = _build_window_list(dates)
    print(f"  窗口数: {len(windows)} (训练截止: {TRAIN_CUTOFF})")

    n_extract = len(extract_pids)
    n_dates = len(dates)
    feat = np.memmap(FEAT_NPY, dtype=np.float32,
                     mode='r', shape=(n_extract, n_dates, FEAT_DIM))

    # ====== 第一遍扫描：per-disk 收集候选窗口 ======
    print("  第一遍扫描：收集候选窗口（per-disk）...")
    # 字典: pi (磁盘索引) → [(wi, pi, label), ...]
    train_pos_candidates = defaultdict(list)  # 故障盘训练正样本候选
    test_pos_candidates = defaultdict(list)   # 故障盘测试正样本候选
    train_neg_candidates = defaultdict(list)  # 健康盘训练负样本候选
    test_neg_candidates = defaultdict(list)   # 健康盘测试负样本候选

    for wi, (w_end, w_idx, date_str) in enumerate(windows):
        window_end_dt = pd.to_datetime(date_str, format="%Y%m%d")
        is_train = date_str <= TRAIN_CUTOFF

        for pi, pid in enumerate(sampled_pids):
            feat_idx = pid_to_extract_idx.get(pid)
            if feat_idx is None:
                continue

            info = disk_info[pid]

            if info['is_failure'] and info['failure_time'] is not None:
                # === 故障盘：只产出正样本（TPS 窗口），其余窗口丢弃 ===
                days = (info['failure_time'] - window_end_dt).days
                if is_train:
                    # 训练集 TPS: l = 1..L，全部保留
                    if 1 <= days <= L:
                        train_pos_candidates[pi].append((wi, pi, 1.0))
                else:
                    # 测试集: 固定 lead time，per-disk 随机选 1 条
                    if 1 <= days <= TEST_LEAD_TIME:
                        test_pos_candidates[pi].append((wi, pi, 1.0))
            else:
                # === 健康盘：per-disk 随机选 1 条负样本 ===
                if is_train:
                    train_neg_candidates[pi].append((wi, pi, 0.0))
                else:
                    test_neg_candidates[pi].append((wi, pi, 0.0))

    # ====== 第二遍：per-disk 随机选取 ======
    print("  第二遍：per-disk 随机选取...")

    # 训练集正样本：TPS 所有窗口全保留
    train_pos_entries = []
    for pi, cands in train_pos_candidates.items():
        train_pos_entries.extend(cands)  # TPS: L 条全保留

    # 测试集正样本：每块故障盘随机选 1 条
    test_pos_entries = []
    for pi, cands in test_pos_candidates.items():
        chosen = cands[RNG.randint(0, len(cands))]
        test_pos_entries.append(chosen)

    # 训练集负样本：每块健康盘随机选 1 条
    train_neg_entries = []
    for pi, cands in train_neg_candidates.items():
        chosen = cands[RNG.randint(0, len(cands))]
        train_neg_entries.append(chosen)

    # 测试集负样本：每块健康盘随机选 1 条
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
          f"| 正负比 1:{n_train_neg/max(n_train_pos,1):.0f}")
    print(f"  测试集: 正 {n_test_pos:,} (每盘 1 条) | 负 {n_test_neg:,} "
          f"| 正负比 1:{n_test_neg/max(n_test_pos,1):.0f}")

    # ====== 训练集：合并 + 打乱 ======
    print(f"\n  训练集: 正样本全保留（TPS L={L}），负样本每盘 1 条")
    train_selected = train_pos_entries + train_neg_entries
    RNG.shuffle(train_selected)
    n_train_final = len(train_selected)
    print(f"  训练样本总计: {n_train_final:,} (正: {n_train_pos:,}, 负: {n_train_neg:,})")

    # ====== 测试集：全量保留（如设置 MAX_TEST_SAMPLES 则随机采样） ======
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

    # ====== 辅助函数：将一个样本写入目标 memmap ======
    def _copy_sample(wi, pi, label, s_tgt, n_tgt, m_tgt, l_tgt, counter):
        pid = sampled_pids[pi]
        w_end, w_idx, date_str = windows[wi]
        feat_idx = pid_to_extract_idx[pid]

        # 自身特征
        disk_seq = feat[feat_idx, w_idx, :]  # [SEQ_LEN, FEAT_DIM]

        # 邻居特征
        neighbors = neighbor_map.get(pid, [])[:MAX_NEIGHBORS]
        neigh_seq_arr = np.zeros((MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM), dtype=np.float32)
        neigh_mask_arr = np.zeros(MAX_NEIGHBORS, dtype=np.bool_)
        for j, npid in enumerate(neighbors):
            nfi = pid_to_extract_idx.get(npid)
            if nfi is None:
                continue
            nseq = feat[nfi, w_idx, :]
            if np.all(nseq == 0):
                continue
            neigh_seq_arr[j] = nseq
            neigh_mask_arr[j] = True

        s_tgt[counter] = disk_seq
        n_tgt[counter] = neigh_seq_arr
        m_tgt[counter] = neigh_mask_arr
        l_tgt[counter] = label
        return counter + 1

    def _save_shards(source_indices, shard_pattern, n_shards, label_prefix):
        """通用分片保存：逐片处理，memmap 直写 + 压缩保存"""
        n_total = len(source_indices)
        if n_total == 0:
            print(f"    {label_prefix}: 0 样本，跳过")
            return 0

        shard_size = (n_total + n_shards - 1) // n_shards
        n_actual = min(n_shards, (n_total + shard_size - 1) // shard_size)

        # 预分组
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
                  f"→ {os.path.getsize(sp)/1024**2:.0f} MB")

            del s_tmp, n_tmp, m_tmp, l_tmp
            for ext in ['.s.tmp', '.n.tmp', '.m.tmp', '.l.tmp']:
                try: os.remove(tmp_prefix + ext)
                except: pass

        return n_actual

    # ====== 保存训练分片 ======
    print("\n  保存训练分片...")
    n_train_shards = _save_shards(train_selected, TRAIN_SHARD_PATTERN, TRAIN_SHARDS, "train")

    # ====== 保存测试分片 ======
    print("  保存测试分片...")
    n_test_shards = _save_shards(test_selected, TEST_SHARD_PATTERN, TEST_SHARDS, "test")

    # 清理 memmap
    feat._mmap.close()
    del feat

    if n_train_shards == 0:
        raise RuntimeError("没有生成任何训练样本！请检查数据或配置")

    return n_train_shards, n_test_shards


# ============================================================
# 5. 主入口
# ============================================================

def build_and_save_samples():
    """完整的数据集构建流水线"""
    print("=" * 60)
    print("[Phase 1] 构建样本数据集")

    # 1. 加载磁盘清单
    disk_info = _load_disk_info()

    # 2. 加载邻居映射（从 neighbor_map.csv）
    neighbor_map = _load_neighbor_map(disk_info) # "disk_A": ["disk_B", "disk_C"],   # A 的邻居是 B 和 C

    # 3. 获取参与磁盘列表
    sampled_pids = _get_all_pids(disk_info)

    # 4. 提取 SMART 特征
    dates, extract_pids, pid_to_extract_idx = _extract_and_build_feat(
        disk_info, sampled_pids, neighbor_map)

    # 5. 生成样本并分片保存
    n_train, n_test = _generate_and_save_samples(
        dates, disk_info, sampled_pids, neighbor_map, extract_pids, pid_to_extract_idx)

    print("=" * 60)
    return n_train, n_test


def _numpy_to_dataloader(s_arr, n_arr, m_arr, l_arr, shuffle, batch_size):
    """numpy 数组 → PyTorch DataLoader"""
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.asarray(s_arr)),
        torch.FloatTensor(np.asarray(n_arr)),
        torch.BoolTensor(np.asarray(m_arr)),
        torch.FloatTensor(np.asarray(l_arr)))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def load_data():
    """主入口：首次构建分片 npz，后续直接加载。返回 (训练分片数, 测试分片数)"""
    if not os.path.exists(TRAIN_SHARD_PATTERN.format(0)) and not os.path.exists(TEST_SHARD_PATTERN.format(0)):
        return build_and_save_samples()
    else:
        print("\n[data_utils] 找到已有分片文件，跳过构建...")
        n_train = get_num_train_shards()
        n_test = get_num_test_shards()
        print(f"  训练分片: {n_train}, 测试分片: {n_test}")
        return n_train, n_test


def load_train_shard(shard_id):
    """加载单个训练分片为 DataLoader"""
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
    """加载单个测试分片为 DataLoader"""
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
    """获取实际训练分片数量"""
    count = 0
    for shard_id in range(TRAIN_SHARDS):
        if os.path.exists(TRAIN_SHARD_PATTERN.format(shard_id)):
            count += 1
        else:
            break
    return count


def get_num_test_shards():
    """获取实际测试分片数量"""
    count = 0
    for shard_id in range(TEST_SHARDS):
        if os.path.exists(TEST_SHARD_PATTERN.format(shard_id)):
            count += 1
        else:
            break
    return count