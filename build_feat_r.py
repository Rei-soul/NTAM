# build_feat_r.py
# r_ 原始值 Z-score 标准化脚本
# 
# Pass 1: 按 model 统计 r_ 列的 mean/std (Welford 算法, 只用训练集)
# Pass 2: 向量化 Z-score 标准化 + 写入 feat_day_*.npy
#
# 内存安全：逐天读取 → 向量化处理 → 释放，峰值内存 ~300 MB

import numpy as np
import pandas as pd
import os
import gc
import json
from collections import defaultdict
from config import *
from memory_guard import start_guard

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "datasets/processed"

R_COLS = [f"r_{sid}" for sid in [
    5, 9, 12, 170, 171, 172, 173, 174, 175,
    177, 180, 181, 182, 183, 184, 187, 188,
    190, 192, 194, 195, 196, 197, 198, 199,
    206, 232, 233, 241, 242
]]


def main():
    start_guard(MEMORY_LIMIT_GB)

    print("=" * 70)
    print("  r_ 原始值 Z-score 标准化")
    print("=" * 70)

    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith('2018')
        and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END
    ])
    dates = [fname[:8] for fname in csv_files]
    date_to_file = {d: os.path.join(DATA_DIR, fname) for d, fname in zip(dates, csv_files)}
    n_dates = len(dates)
    usecols = ['disk_id', 'model'] + R_COLS
    n_cols = len(R_COLS)

    # ============================================================
    # Pass 1: Welford 统计
    # ============================================================
    print(f"\n  [Pass 1] 按 model 统计 r_ 值的 mean/std (仅训练集)...")

    # 从 Pass 1 的中断恢复
    r_stats_path = os.path.join(PROCESSED_DIR, "r_stats.json")
    if os.path.exists(r_stats_path):
        print(f"  检测到已有 r_stats.json，跳过 Pass 1")
        with open(r_stats_path, 'r') as f:
            saved = json.load(f)
        stats_final = {}
        for k, v in saved['per_model'].items():
            m, col = k.rsplit('_r_', 1)
            ci = R_COLS.index(f"r_{col}")
            stats_final[(m, ci)] = v
        global_stats = {R_COLS.index(c): saved['global'][c] for c in saved['global']}
    else:
        stats = {}
        train_count = 0
        for di, date_str in enumerate(dates):
            if not (TRAIN_START <= date_str <= TRAIN_END):
                continue
            fpath = date_to_file[date_str]
            try:
                df = pd.read_csv(fpath, usecols=usecols)
            except Exception:
                continue
            df['model'] = df['model'].astype(str)
            df[R_COLS] = df[R_COLS].replace("", np.nan).astype(np.float32)

            for model_name, group in df.groupby('model'):
                for ci in range(n_cols):
                    vals = group[R_COLS[ci]].dropna().values
                    if len(vals) == 0:
                        continue
                    key = (model_name, ci)
                    if key not in stats:
                        stats[key] = {'count': 0, 'mean': 0.0, 'M2': 0.0}
                    s = stats[key]
                    for v in vals:
                        s['count'] += 1
                        delta = float(v) - s['mean']
                        s['mean'] += delta / s['count']
                        delta2 = float(v) - s['mean']
                        s['M2'] += delta * delta2
            del df; gc.collect()
            train_count += 1
            if (di + 1) % 30 == 0:
                print(f"    [{di+1}] 已扫描({train_count}天)")

        stats_final = {}
        for key, s in stats.items():
            std = np.sqrt(s['M2'] / s['count']) if s['count'] >= 10 > 0 else 1.0
            if std < 1e-8:
                std = 1.0
            stats_final[key] = {'mean': s['mean'], 'std': std, 'count': s['count']}

        global_stats = {}
        for ci in range(n_cols):
            means = [s['mean'] for (m, i), s in stats_final.items() if i == ci and s['count'] >= 10]
            global_stats[ci] = {
                'mean': np.mean(means) if means else 0.0,
                'std': np.std(means) if means else 1.0
            }

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        with open(r_stats_path, 'w') as f:
            json.dump({
                'per_model': {f"{m}_r_{R_COLS[ci][2:]}": s for (m, ci), s in stats_final.items()},
                'global': {R_COLS[ci]: gs for ci, gs in global_stats.items()}
            }, f, indent=2)

        print(f"\n    训练集天数: {train_count} | model数: {len(set(k[0] for k in stats_final))}")
        print(f"    统计量已保存至 r_stats.json")

    # ============================================================
    # Pass 2: 向量化 Z-score + 写入 feat_day
    # ============================================================
    print(f"\n  [Pass 2] 向量化 Z-score + 写入 feat_day_*.npy...")

    from data_utils import _load_disk_info, _load_neighbor_map, _get_all_pids
    disk_info = _load_disk_info()
    neighbor_map = _load_neighbor_map(disk_info)
    sampled_pids = _get_all_pids(disk_info)

    all_needed = set(sampled_pids)
    for pid in sampled_pids:
        all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
    extract_pids = sorted(all_needed)
    pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
    n_extract = len(extract_pids)

    # lookup DataFrame
    lookup_rows = []
    for pid in extract_pids:
        parts = pid.split('_', 1)
        if len(parts) == 2:
            lookup_rows.append({'disk_id': int(parts[0]), 'model': parts[1],
                                '_idx': pid_to_extract_idx[pid]})
    lookup_df = pd.DataFrame(lookup_rows)
    lookup_df['disk_id'] = lookup_df['disk_id'].astype(int)
    lookup_df['model'] = lookup_df['model'].astype(str)

    # 预分配 feat_day
    feat_files = [os.path.join(PROCESSED_DIR, f"feat_day_{di:04d}.npy") for di in range(n_dates)]
    for fp in feat_files:
        np.save(fp, np.zeros((n_extract, n_cols), dtype=np.float32))

    # 构建向量化查找表: means/std 矩阵 (n_models, n_cols)
    all_models = sorted(set(k[0] for k in stats_final.keys()))
    model_to_midx = {m: i for i, m in enumerate(all_models)}
    n_models_total = len(all_models)
    means_arr = np.zeros((n_models_total, n_cols), dtype=np.float32)
    stds_arr = np.ones((n_models_total, n_cols), dtype=np.float32)
    for (m, ci), s in stats_final.items():
        midx = model_to_midx[m]
        if s['count'] >= 10:
            means_arr[midx, ci] = s['mean']
            stds_arr[midx, ci] = s['std']

    g_means = np.array([global_stats[ci]['mean'] for ci in range(n_cols)], dtype=np.float32)
    g_stds = np.array([global_stats[ci]['std'] for ci in range(n_cols)], dtype=np.float32)
    g_stds[g_stds < 1e-8] = 1.0

    print(f"  需提取盘数: {n_extract:,}, 日期数: {n_dates}, Model数: {n_models_total}")

    for di, date_str in enumerate(dates):
        fpath = date_to_file[date_str]
        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception:
            continue

        df['disk_id'] = df['disk_id'].astype(int)
        df['model'] = df['model'].astype(str)
        df['_pid'] = df['disk_id'].astype(str) + '_' + df['model']
        df[R_COLS] = df[R_COLS].replace("", np.nan).astype(np.float32)

        merged = df.merge(lookup_df[['disk_id', 'model', '_idx']], on=['disk_id', 'model'], how='inner')
        n_merged = len(merged)
        del df

        if n_merged == 0:
            continue

        vals = merged[R_COLS].fillna(0).values.astype(np.float32)
        midx_arr = np.array([model_to_midx.get(m, -1) for m in merged['model'].values], dtype=np.int64)
        idx_arr = merged['_idx'].values.astype(np.int64)

        # 向量化 Z-score
        valid = midx_arr >= 0
        z_vals = np.zeros((n_merged, n_cols), dtype=np.float32)

        if valid.sum() > 0:
            mv = midx_arr[valid]
            vv = vals[valid]
            z_vals[valid] = np.clip((vv - means_arr[mv]) / stds_arr[mv], -5.0, 5.0)

        invalid = ~valid
        if invalid.sum() > 0:
            z_vals[invalid] = np.clip((vals[invalid] - g_means) / g_stds, -5.0, 5.0)

        # 写入
        day_arr = np.load(feat_files[di], mmap_mode='r+')
        day_arr[idx_arr] = z_vals
        day_arr.flush()
        del day_arr, merged, vals, midx_arr, idx_arr, z_vals
        gc.collect()

        if (di + 1) % 30 == 0 or di == 0:
            print(f"    [{di+1}/{n_dates}] {date_str} ({n_merged:,}条)")

    print(f"\n  ✅ Z-score feat_day 文件已生成 ({n_dates} 个文件, {n_extract:,}×{n_cols})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()