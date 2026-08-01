# compare_r_vs_n.py
# 快速对比 r_ (原始值) vs n_ (归一化值) 在故障盘和健康盘上的区分力
# 不重建全量数据，只抽样几块盘做可视化对比

import numpy as np
import pandas as pd
import os
import gc
from config import *
from memory_guard import start_guard

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "code/datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")

# n_ 列 (当前使用的 30 列)
N_COLS = [f"n_{sid}" for sid in [
    5, 9, 12, 170, 171, 172, 173, 174, 175, 177,
    180, 181, 182, 183, 184, 187, 188, 190, 192,
    194, 195, 196, 197, 198, 199, 206, 232, 233, 241, 242
]]
# 对应的 r_ 列
R_COLS = [c.replace("n_", "r_") for c in N_COLS]


def main():
    # 启动内存看门狗
    start_guard(MEMORY_LIMIT_GB)

    print("=" * 70)
    print("  r_ vs n_ SMART 值区分力对比")
    print("=" * 70)

    # 1. 加载磁盘清单
    target_df = pd.read_csv(TARGET_FILE)
    target_df['pair_id'] = target_df['pair_id'].astype(str)

    fail_pids = target_df[target_df['is_failure'] == True]
    healthy_pids = target_df[target_df['is_failure'] == False]

    print(f"  故障盘: {len(fail_pids):,} | 健康盘: {len(healthy_pids):,}")

    # 随机选 3 个故障盘和 10 个健康盘
    np.random.seed(42)
    sample_fail = fail_pids.sample(min(3, len(fail_pids)))
    sample_healthy = healthy_pids.sample(10)

    all_samples = pd.concat([sample_fail, sample_healthy])
    sample_pids = set(all_samples['pair_id'].tolist())

    # 2. 扫描数据文件，收集这些盘的 r_ 和 n_ 值
    print(f"\n  扫描 CSV 收集 {len(sample_pids)} 块样本盘...")

    usecols = ['disk_id', 'model'] + N_COLS + R_COLS

    # 收集数据: {pid: {col: [values]}}
    fail_r = {c: [] for c in R_COLS}
    fail_n = {c: [] for c in N_COLS}
    healthy_r = {c: [] for c in R_COLS}
    healthy_n = {c: [] for c in N_COLS}

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith('2018')
        and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END
    ])

    # 只用训练集数据 (≤ TRAIN_CUTOFF)
    for fi, fname in enumerate(csv_files):
        date_str = fname[:8]
        if date_str > TRAIN_CUTOFF:
            break  # 只用训练集

        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception:
            continue

        df['disk_id'] = df['disk_id'].astype(int).astype(str)
        df['model'] = df['model'].astype(str)
        df['_pid'] = df['disk_id'] + '_' + df['model']

        df_sample = df[df['_pid'].isin(sample_pids)]
        if len(df_sample) == 0:
            del df
            continue

        for _, row in df_sample.iterrows():
            pid = row['_pid']
            is_fail = pid in set(sample_fail['pair_id'])

            for nc, rc in zip(N_COLS, R_COLS):
                nv = row.get(nc, np.nan)
                rv = row.get(rc, np.nan)
                if not pd.isna(nv) and nv != "":
                    if is_fail:
                        fail_n[nc].append(float(nv))
                    else:
                        healthy_n[nc].append(float(nv))
                if not pd.isna(rv) and rv != "":
                    if is_fail:
                        fail_r[rc].append(float(rv))
                    else:
                        healthy_r[rc].append(float(rv))

        del df, df_sample
        gc.collect()
        if (fi + 1) % 30 == 0:
            print(f"    [{fi+1}/{len(csv_files)}] 已扫描")

    # 3. 统计对比
    print(f"\n  {'='*70}")
    print(f"  故障盘 r_ 值统计 (3块盘):")
    print(f"  {'列名':<10} {'min':>12} {'mean':>12} {'max':>12} {'std':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for rc in R_COLS:
        vals = np.array(fail_r[rc])
        if len(vals) > 0:
            vals_nz = vals[vals != 0]  # 只看非零值（排除缺失填充）
            if len(vals_nz) > 0:
                print(f"  {rc:<10} {vals_nz.min():>12.2f} {vals_nz.mean():>12.2f} {vals_nz.max():>12.2f} {vals_nz.std():>12.2f}")
            else:
                print(f"  {rc:<10} {'(全零)':>12}")
        else:
            print(f"  {rc:<10} {'(无数据)':>12}")

    print(f"\n  健康盘 r_ 值统计 (10块盘):")
    print(f"  {'列名':<10} {'min':>12} {'mean':>12} {'max':>12} {'std':>12}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    for rc in R_COLS:
        vals = np.array(healthy_r[rc])
        if len(vals) > 0:
            vals_nz = vals[vals != 0]
            if len(vals_nz) > 0:
                print(f"  {rc:<10} {vals_nz.min():>12.2f} {vals_nz.mean():>12.2f} {vals_nz.max():>12.2f} {vals_nz.std():>12.2f}")
            else:
                print(f"  {rc:<10} {'(全零)':>12}")
        else:
            print(f"  {rc:<10} {'(无数据)':>12}")

    # 4. 逐列对比区分力
    print(f"\n  {'='*70}")
    print(f"  区分力对比 (故障 vs 健康):")
    print(f"  {'列名':<10} {'r_故障均值':>12} {'r_健康均值':>12} {'r_差异%':>10} "
          f"| {'n_故障均值':>12} {'n_健康均值':>12} {'n_差异%':>10} | {'建议':>10}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

    r_better = 0
    n_better = 0
    both_weak = 0

    for nc, rc in zip(N_COLS, R_COLS):
        fv_r = np.array(fail_r[rc])
        hv_r = np.array(healthy_r[rc])
        fv_n = np.array(fail_n[nc])
        hv_n = np.array(healthy_n[nc])

        # 仅统计非零值
        fv_r_nz = fv_r[fv_r != 0] if len(fv_r) > 0 else np.array([])
        hv_r_nz = hv_r[hv_r != 0] if len(hv_r) > 0 else np.array([])
        fv_n_nz = fv_n[fv_n != 0] if len(fv_n) > 0 else np.array([])
        hv_n_nz = hv_n[hv_n != 0] if len(hv_n) > 0 else np.array([])

        r_fail_mean = fv_r_nz.mean() if len(fv_r_nz) > 0 else 0
        r_healthy_mean = hv_r_nz.mean() if len(hv_r_nz) > 0 else 0
        n_fail_mean = fv_n_nz.mean() if len(fv_n_nz) > 0 else 0
        n_healthy_mean = hv_n_nz.mean() if len(hv_n_nz) > 0 else 0

        # 差异百分比
        r_base = max(abs(r_healthy_mean), 1e-8)
        n_base = max(abs(n_healthy_mean), 1e-8)
        r_diff = abs(r_fail_mean - r_healthy_mean) / r_base * 100
        n_diff = abs(n_fail_mean - n_healthy_mean) / n_base * 100

        if r_diff > n_diff * 1.5:
            suggest = "🔴 用r_"
            r_better += 1
        elif n_diff > r_diff * 1.5:
            suggest = "🟡 用n_"
            n_better += 1
        else:
            suggest = "⬜ 差不多"
            both_weak += 1

        print(f"  {rc.replace('r_','').ljust(8)[:8]:<10} "
              f"{r_fail_mean:>12.2f} {r_healthy_mean:>12.2f} {r_diff:>9.1f}% "
              f"| {n_fail_mean:>12.2f} {n_healthy_mean:>12.2f} {n_diff:>9.1f}% "
              f"| {suggest:>10}")

    print(f"\n  📊 汇总: r_ 更好: {r_better}列 | n_ 更好: {n_better}列 | 差不多: {both_weak}列")

    if r_better > n_better:
        print(f"\n  💡 建议: 切换到 r_ 原始值！r_ 在 {r_better}/{len(N_COLS)} 列上区分力更强")
    elif n_better > r_better:
        print(f"\n  💡 建议: 继续使用 n_ 归一化值")
    else:
        print(f"\n  💡 建议: 两者差异不大，r_ 值范围更大可能对梯度学习更友好")

    print("=" * 70)


if __name__ == "__main__":
    main()