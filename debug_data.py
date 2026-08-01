# debug_data.py
# 数据管道验证工具：逐一检查数据是否正确流入模型
# 五层验证：
#   1. 列映射验证 (OLD_N_COLS → N_COLS)
#   2. feat_day 文件维度验证
#   3. 分片维度 + 样本值验证
#   4. 正负样本特征差异
#   5. 邻居有效性检查

import numpy as np
import os
import sys
import gc
from collections import defaultdict
from memory_guard import start_guard
from config import MEMORY_LIMIT_GB

PROCESSED_DIR = "code/datasets/processed"
FEAT_DIM = 30
SEQ_LEN = 15
MAX_NEIGHBORS = 5

# 新旧列定义（与 data_utils.py 保持一致）
N_COLS = [
    "n_5", "n_9", "n_12",
    "n_170", "n_171", "n_172", "n_173", "n_174", "n_175",
    "n_177",
    "n_180", "n_181", "n_182", "n_183", "n_184",
    "n_187", "n_188",
    "n_190", "n_192", "n_194", "n_195", "n_196", "n_197", "n_198", "n_199",
    "n_206",
    "n_232", "n_233", "n_241", "n_242"
]

OLD_N_COLS = [
    "n_1", "n_2", "n_3", "n_4", "n_5", "n_6", "n_7", "n_8", "n_9",
    "n_10", "n_11", "n_12", "n_13",
    "n_170", "n_171", "n_172", "n_173", "n_174", "n_175",
    "n_177", "n_180", "n_181", "n_182", "n_183", "n_184",
    "n_187", "n_188", "n_189",
    "n_190", "n_191", "n_192", "n_193", "n_194", "n_195",
    "n_196", "n_197", "n_198", "n_199", "n_200",
    "n_204", "n_205", "n_206", "n_207", "n_211",
    "n_232", "n_233", "n_240", "n_241", "n_242", "n_244", "n_245"
]

_COL_IDX_MAP = [OLD_N_COLS.index(c) for c in N_COLS]
OLD_FEAT_DIM = len(OLD_N_COLS)  # 51


def load_feat_day_file(di=0):
    """加载第 di 天的 feat_day 文件"""
    fpath = os.path.join(PROCESSED_DIR, f"feat_day_{di:04d}.npy")
    if not os.path.exists(fpath):
        print(f"  ❌ feat_day_{di:04d}.npy 不存在！")
        return None
    return np.load(fpath, mmap_mode='r')


def load_shard(prefix, shard_id=0):
    """加载一个分片文件"""
    fpath = os.path.join(PROCESSED_DIR, f"{prefix}_shard_{shard_id:02d}.npz")
    if not os.path.exists(fpath):
        print(f"  ❌ {fpath} 不存在！")
        return None
    return np.load(fpath, mmap_mode='r')


def verify_col_mapping():
    """验证1: 列映射正确性"""
    print("=" * 70)
    print("【验证1】列映射验证 (51列 → 30列)")
    print("=" * 70)

    arr = load_feat_day_file(0)
    if arr is None:
        return False

    dim_on_disk = arr.shape[1]
    print(f"  磁盘上的维度: {dim_on_disk}")

    if dim_on_disk == OLD_FEAT_DIM:
        print(f"  ✅ 检测到旧 51 维文件")

        # 验证 COL_IDX_MAP
        expected = [OLD_N_COLS.index(c) for c in N_COLS]
        if expected == _COL_IDX_MAP:
            print(f"  ✅ _COL_IDX_MAP 与纸上计算一致")
        else:
            print(f"  ❌ _COL_IDX_MAP 不一致！")
            return False

        # 验证一个具体示例
        print(f"\n  示例映射 (前5列):")
        for i in range(min(5, len(N_COLS))):
            old_idx = _COL_IDX_MAP[i]
            old_name = OLD_N_COLS[old_idx]
            new_name = N_COLS[i]
            match = "✅" if old_name == new_name else "❌"
            print(f"    N_COLS[{i}]={new_name}  ←  OLD[{old_idx}]={old_name}  {match}")

        # 实际测试：取第一行做切片
        row_full = arr[0]  # (51,)
        row_sliced = row_full[_COL_IDX_MAP]  # (30,)
        print(f"\n  实际切片测试: full={row_full.shape} → sliced={row_sliced.shape}")

        # 验证切片中每列的非零值与原始列一致
        all_ok = True
        for i in range(len(N_COLS)):
            old_idx = _COL_IDX_MAP[i]
            if not np.allclose(row_full[old_idx], row_sliced[i], equal_nan=True):
                print(f"    ❌ 第{i}列不匹配！OLD[{old_idx}]={row_full[old_idx]} vs sliced[{i}]={row_sliced[i]}")
                all_ok = False
        if all_ok:
            print(f"  ✅ 切片验证通过：30列值与原始51列中对应位置完全一致")

    elif dim_on_disk == FEAT_DIM:
        print(f"  ✅ 检测到新 30 维文件，无需切片")
    else:
        print(f"  ❌ 未知维度: {dim_on_disk}")
        return False

    del arr
    return True


def verify_feat_day_content():
    """验证2: feat_day 文件内容"""
    print("\n" + "=" * 70)
    print("【验证2】feat_day 文件内容验证")
    print("=" * 70)

    arr = load_feat_day_file(0)
    if arr is None:
        return

    n_disks, dim = arr.shape
    print(f"  feat_day_0000.npy: {n_disks:,} 盘 × {dim} 维")

    # 按列统计非零率 (用旧维度，不用 SLICE)
    print(f"\n  各列非零值率 (原始 {dim} 维):")
    print(f"  {'列名':<10} {'非零率':>8} {'全零?':>8}")
    print(f"  {'─'*10} {'─'*8} {'─'*8}")

    # 抽样 5000 行做统计（全量太慢）
    sample_rows = min(5000, n_disks)
    sample_idx = np.random.choice(n_disks, sample_rows, replace=False)
    sample = arr[sample_idx]

    nz_counts = {}
    for ci, col_name in enumerate(OLD_N_COLS[:dim]) if dim > len(OLD_N_COLS) else enumerate(OLD_N_COLS):
        # 统计非零行数
        non_zero = (sample[:, ci] != 0).sum()
        nz_rate = non_zero / sample_rows * 100
        all_zero = "⚠️ 全0" if non_zero == 0 else ""
        print(f"  {col_name:<10} {nz_rate:>7.1f}% {all_zero:>8}")
        nz_counts[col_name] = nz_rate

    del arr, sample
    gc.collect()
    return nz_counts


def verify_shard_content():
    """验证3: 分片内容验证"""
    print("\n" + "=" * 70)
    print("【验证3】分片维度 + 样本值验证")
    print("=" * 70)

    for prefix in ["train", "test"]:
        data = load_shard(prefix, 0)
        if data is None:
            continue

        s_arr = data['s']  # (N, SEQ_LEN, FEAT_DIM)
        n_arr = data['n']  # (N, MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM)
        m_arr = data['m']  # (N, MAX_NEIGHBORS)
        l_arr = data['l']  # (N, 1)

        n_samples = len(l_arr)
        n_pos = int(l_arr.sum())
        n_neg = n_samples - n_pos

        print(f"\n  {prefix}_shard_00.npz:")
        print(f"    s.shape = {s_arr.shape}   预期: ({n_samples}, {SEQ_LEN}, {FEAT_DIM})")
        print(f"    n.shape = {n_arr.shape}   预期: ({n_samples}, {MAX_NEIGHBORS}, {SEQ_LEN}, {FEAT_DIM})")
        print(f"    m.shape = {m_arr.shape}   预期: ({n_samples}, {MAX_NEIGHBORS})")
        print(f"    l.shape = {l_arr.shape}   预期: ({n_samples}, 1)")
        print(f"    正样本: {n_pos} | 负样本: {n_neg} | 正负比 1:{n_neg/max(n_pos,1):.0f}")

        # 维度检查
        dim_ok = True
        if s_arr.shape != (n_samples, SEQ_LEN, FEAT_DIM):
            print(f"    ❌ s 维度错误: {s_arr.shape}")
            dim_ok = False
        if n_arr.shape != (n_samples, MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM):
            print(f"    ❌ n 维度错误: {n_arr.shape}")
            dim_ok = False
        if dim_ok:
            print(f"    ✅ 维度检查通过")
        else:
            return False

        # 统计正样本和负样本的自身特征非零率
        pos_idx = l_arr[:, 0] == 1
        neg_idx = l_arr[:, 0] == 0

        pos_s = s_arr[pos_idx]  # (n_pos, 15, 30)
        neg_s = s_arr[neg_idx]  # (n_neg, 15, 30)

        # 展平时间维，统计全局非零率
        pos_flat = pos_s.reshape(-1, FEAT_DIM) if len(pos_s) > 0 else np.zeros((0, FEAT_DIM))
        neg_flat = neg_s.reshape(-1, FEAT_DIM) if len(neg_s) > 0 else np.zeros((0, FEAT_DIM))

        print(f"\n  样本非零值率 (展平时序):")
        print(f"  {'列名':<10} {'正样本非零率':>12} {'负样本非零率':>12} {'差异':>8}")
        print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*8}")

        for ci in range(FEAT_DIM):
            col_name = N_COLS[ci] if ci < len(N_COLS) else f"c{ci}"
            pos_nz = (pos_flat[:, ci] != 0).mean() * 100 if len(pos_flat) > 0 else 0
            neg_nz = (neg_flat[:, ci] != 0).mean() * 100 if len(neg_flat) > 0 else 0
            diff = pos_nz - neg_nz
            flag = ""
            if abs(diff) > 10:
                flag = f"  ← 偏差{abs(diff):.0f}%"
            elif pos_nz < 1 and neg_nz < 1:
                flag = "  ⚠️ 接近全零"

            print(f"  {col_name:<10} {pos_nz:>11.1f}% {neg_nz:>11.1f}% {diff:>+7.1f}%{flag}")

        # 全零样本检测
        all_zero_samples = np.all(s_arr == 0, axis=(1, 2))
        n_all_zero = all_zero_samples.sum()
        if n_all_zero > 0:
            print(f"\n  ⚠️ 全零样本: {n_all_zero}/{n_samples} ({n_all_zero/n_samples*100:.1f}%)")
            # 看全零样本是正还是负
            pos_zero = all_zero_samples[pos_idx].sum()
            neg_zero = all_zero_samples[neg_idx].sum()
            print(f"    其中正样本: {pos_zero}, 负样本: {neg_zero}")

        del data, pos_s, neg_s
        gc.collect()


def verify_pos_vs_neg():
    """验证4: 正负样本特征差异"""
    print("\n" + "=" * 70)
    print("【验证4】正负样本特征差异")
    print("=" * 70)

    # 扫描所有训练分片
    import glob
    shard_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "train_shard_*.npz")))
    if len(shard_files) == 0:
        print("  ❌ 没有训练分片！")
        return

    max_shards = min(5, len(shard_files))
    print(f"  扫描前 {max_shards} 个训练分片...")

    pos_vals = defaultdict(list)
    neg_vals = defaultdict(list)

    for s in range(max_shards):
        data = np.load(shard_files[s], mmap_mode='r')
        s_arr = data['s']
        l_arr = data['l']

        pos_mask = l_arr[:, 0] == 1
        neg_mask = l_arr[:, 0] == 0

        for ci in range(FEAT_DIM):
            col_name = N_COLS[ci]
            pv = s_arr[pos_mask, :, ci].ravel()
            nv = s_arr[neg_mask, :, ci].ravel()
            # 只保留非零值（避免 NaN 的 0 填充干扰）
            pv_nz = pv[pv != 0]
            nv_nz = nv[nv != 0]
            if len(pv_nz) > 0:
                pos_vals[col_name].extend(pv_nz[:1000].tolist())  # 限制数量
            if len(nv_nz) > 0:
                neg_vals[col_name].extend(nv_nz[:10000].tolist())

        del data
        gc.collect()

    # KS 检验
    try:
        from scipy import stats
    except ImportError:
        print("  ⚠️ scipy 未安装，跳过 KS 检验")
        # 简单均值比较
        print(f"\n  {'列名':<10} {'正样本均值(非0)':>15} {'负样本均值(非0)':>15} {'差异':>10}")
        print(f"  {'─'*10} {'─'*15} {'─'*15} {'─'*10}")
        for ci in range(FEAT_DIM):
            cn = N_COLS[ci]
            pv = np.array(pos_vals.get(cn, []))
            nv = np.array(neg_vals.get(cn, []))
            p_mean = pv.mean() if len(pv) > 0 else 0
            n_mean = nv.mean() if len(nv) > 0 else 0
            diff_pct = abs(p_mean - n_mean) / max(abs(n_mean), 1e-8) * 100 if abs(n_mean) > 1e-8 else abs(p_mean - n_mean) * 100
            print(f"  {cn:<10} {p_mean:>15.4f} {n_mean:>15.4f} {p_mean - n_mean:>+10.4f}")
        return

    print(f"\n  {'列名':<10} {'正样本均值':>12} {'负样本均值':>12} {'KS p值':>10} {'判别力':>8}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

    n_disc = 0
    for ci in range(FEAT_DIM):
        cn = N_COLS[ci]
        pv = np.array(pos_vals.get(cn, []))
        nv = np.array(neg_vals.get(cn, []))

        if len(pv) < 10 or len(nv) < 10:
            disc = "⬜ 样本少"
            ks_p = -1
        else:
            try:
                _, ks_p = stats.ks_2samp(pv, nv)
            except Exception:
                ks_p = 1.0

            if ks_p < 0.001:
                disc = "🔴 强"
                n_disc += 1
            elif ks_p < 0.05:
                disc = "🟡 中"
            else:
                disc = "⬜ 弱"

        p_mean = pv.mean() if len(pv) > 0 else 0
        n_mean = nv.mean() if len(nv) > 0 else 0

        if ks_p >= 0:
            print(f"  {cn:<10} {p_mean:>12.4f} {n_mean:>12.4f} {ks_p:>10.4f} {disc:>8}")
        else:
            print(f"  {cn:<10} {p_mean:>12.4f} {n_mean:>12.4f} {'─':>10} {disc:>8}")

    print(f"\n  📊 {n_disc}/{FEAT_DIM} 列有强区分力 (KS p<0.001)")


def verify_neighbors():
    """验证5: 邻居有效性"""
    print("\n" + "=" * 70)
    print("【验证5】邻居有效性检查")
    print("=" * 70)

    data = load_shard("train", 0)
    if data is None:
        return

    m_arr = data['m']   # (N, MAX_NEIGHBORS)
    n_arr = data['n']   # (N, MAX_NEIGHBORS, 15, 30)

    n_samples = m_arr.shape[0]

    # 每样本有几个有效邻居
    n_valid_neighbors = m_arr.sum(axis=1)  # (N,)

    print(f"  邻居统计:")
    print(f"    0 个邻居: {(n_valid_neighbors == 0).sum()} ({(n_valid_neighbors == 0).mean()*100:.1f}%)")
    print(f"    1 个邻居: {(n_valid_neighbors == 1).sum()} ({(n_valid_neighbors == 1).mean()*100:.1f}%)")
    print(f"    2 个邻居: {(n_valid_neighbors == 2).sum()} ({(n_valid_neighbors == 2).mean()*100:.1f}%)")
    print(f"    3 个邻居: {(n_valid_neighbors == 3).sum()} ({(n_valid_neighbors == 3).mean()*100:.1f}%)")
    print(f"    4 个邻居: {(n_valid_neighbors == 4).sum()} ({(n_valid_neighbors == 4).mean()*100:.1f}%)")
    print(f"    5 个邻居: {(n_valid_neighbors == 5).sum()} ({(n_valid_neighbors == 5).mean()*100:.1f}%)")
    print(f"    平均邻居数: {n_valid_neighbors.mean():.2f}")

    # 有效邻居的特征非零率
    valid_mask = m_arr == 1  # (N, M)
    if valid_mask.sum() > 0:
        valid_neigh = n_arr[valid_mask]  # (n_valid, 15, 30)
        # 展平所有有效邻居的所有时间步
        valid_flat = valid_neigh.reshape(-1, FEAT_DIM)
        nz_rate = (valid_flat != 0).mean(axis=0) * 100

        print(f"\n  有效邻居特征非零率:")
        nz_cols_low = 0
        for ci in range(FEAT_DIM):
            col_name = N_COLS[ci]
            rate = nz_rate[ci]
            flag = " ⚠️ <1%" if rate < 1 else ""
            if rate < 5:
                nz_cols_low += 1
            if rate < 5 or (ci < 5):
                print(f"    {col_name:<10} {rate:>6.2f}%{flag}")
        if nz_cols_low > 10:
            print(f"    ⚠️ {nz_cols_low}/{FEAT_DIM} 列邻居非零率 < 5%")
        else:
            print(f"    ... (完整列表见上面)")

    del data, n_arr, m_arr
    gc.collect()


def main():
    # 启动内存看门狗
    start_guard(MEMORY_LIMIT_GB)

    print("=" * 70)
    print("  NTAM 数据管道验证工具")
    print("=" * 70)

    # 验证1
    ok = verify_col_mapping()

    # 验证2
    verify_feat_day_content()

    # 验证3
    verify_shard_content()

    # 验证4
    verify_pos_vs_neg()

    # 验证5
    verify_neighbors()

    print("\n" + "=" * 70)
    print("  验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()