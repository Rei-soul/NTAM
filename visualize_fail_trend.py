# visualize_fail_trend.py
# 验证故障盘的 SMART r_ 值在故障前是否有明显变化趋势
# 抽样几块盘，画时间序列曲线（文本格式），判断数据是否可用

import numpy as np
import pandas as pd
import os
import gc
from config import *
from memory_guard import start_guard

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "code/datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")

# 关键 SMART 属性（最可能与故障相关）
KEY_R_COLS = [
    "r_5",    # Reallocated Sectors Count - 坏扇区重分配
    "r_197",  # Current Pending Sector Count - 待处理坏扇区
    "r_187",  # Reported Uncorrectable Errors - 不可纠正错误
    "r_188",  # Command Timeout - 命令超时
    "r_173",  # SSD Wear Leveling Count - 磨损计数
    "r_9",    # Power-On Hours - 运行时间 (baseline)
    "r_194",  # Temperature - 温度
    "r_196",  # Reallocation Event Count - 重分配事件
    "r_241",  # Total LBAs Written - 写入量
    "r_242",  # Total LBAs Read - 读取量
]

# 对应 n_ 列做对比
KEY_N_COLS = [c.replace("r_", "n_") for c in KEY_R_COLS]


def main():
    start_guard(MEMORY_LIMIT_GB)

    print("=" * 70)
    print("  故障盘 SMART 趋势验证")
    print("=" * 70)

    # 1. 加载磁盘清单
    target_df = pd.read_csv(TARGET_FILE)
    target_df['pair_id'] = target_df['pair_id'].astype(str)
    target_df['failure_time'] = pd.to_datetime(target_df['failure_time'])

    fail_df = target_df[target_df['is_failure'] == True].copy()
    healthy_df = target_df[target_df['is_failure'] == False].copy()

    # 筛选：故障时间在训练集范围内且有足够的历史数据
    train_cutoff_dt = pd.to_datetime(TRAIN_CUTOFF)
    fail_df = fail_df[fail_df['failure_time'] <= train_cutoff_dt]

    print(f"  训练集中故障盘: {len(fail_df):,} | 健康盘: {len(healthy_df):,}")

    # 2. 抽样：选几块故障盘和健康盘
    np.random.seed(42)
    n_fail_sample = 3
    n_healthy_sample = 5
    sample_fail = fail_df.sample(min(n_fail_sample, len(fail_df)))
    sample_healthy = healthy_df.sample(n_healthy_sample)

    # 3. 扫描日期，建立日期索引
    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith('2018')
        and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END
    ])
    dates = []
    date_to_file = {}
    for fname in csv_files:
        date_str = fname[:8]
        dates.append(date_str)
        date_to_file[date_str] = os.path.join(DATA_DIR, fname)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    usecols = ['disk_id', 'model'] + KEY_R_COLS + KEY_N_COLS

    # 4. 对每块故障盘，收集故障前 30 天的数据
    TRACE_DAYS = 30

    print(f"\n{'=' * 70}")
    print(f"  【故障盘时间序列】故障前 {TRACE_DAYS} 天 r_ 值变化")
    print(f"{'=' * 70}")

    for _, disk_row in sample_fail.iterrows():
        pid = str(disk_row['pair_id'])
        ft = disk_row['failure_time']
        ft_str = ft.strftime("%Y%m%d")
        ft_di = date_to_idx.get(ft_str)

        if ft_di is None:
            print(f"\n  {pid}: failure_time {ft_str} 不在数据日期范围内，跳过")
            continue

        # 故障前 30 天的日期索引
        start_di = max(0, ft_di - TRACE_DAYS)
        trace_dates = dates[start_di:ft_di + 1]
        # 从后往前取，最后一天是故障日
        display_dates = trace_dates[-min(TRACE_DAYS, len(trace_dates)):]

        print(f"\n  {'─' * 65}")
        print(f"  磁盘: {pid}")
        print(f"  故障日: {ft_str} | 数据范围: {display_dates[0]} ~ {display_dates[-1]}")

        # 收集数据
        trace_r = {c: [] for c in KEY_R_COLS}
        trace_n = {c: [] for c in KEY_N_COLS}

        for date_str in display_dates:
            fpath = date_to_file.get(date_str)
            if fpath is None:
                continue
            try:
                df = pd.read_csv(fpath, usecols=usecols)
            except Exception:
                continue

            df['disk_id'] = df['disk_id'].astype(int).astype(str)
            df['model'] = df['model'].astype(str)
            df['_pid'] = df['disk_id'] + '_' + df['model']

            row = df[df['_pid'] == pid]
            if len(row) == 0:
                for c in KEY_R_COLS:
                    trace_r[c].append(np.nan)
                for c in KEY_N_COLS:
                    trace_n[c].append(np.nan)
                del df
                continue

            r = row.iloc[0]
            for c in KEY_R_COLS:
                val = r.get(c, np.nan)
                trace_r[c].append(float(val) if not pd.isna(val) and val != "" else np.nan)
            for c in KEY_N_COLS:
                val = r.get(c, np.nan)
                trace_n[c].append(float(val) if not pd.isna(val) and val != "" else np.nan)
            del df

        # 只显示有变化的列
        cols_with_change = []
        for c in KEY_R_COLS:
            vals = np.array(trace_r[c])
            vals_clean = vals[~np.isnan(vals)]
            if len(vals_clean) >= 2 and (vals_clean.max() - vals_clean.min()) > 0:
                cols_with_change.append(c)

        if not cols_with_change:
            cols_with_change = KEY_R_COLS[:5]  # 兜底

        # 打印 r_ 趋势
        print(f"\n  r_ 原始值 ({len(display_dates)} 天):")
        header = f"  {'Day':<12}"
        for c in cols_with_change:
            header += f" {c:<12}"
        print(header)
        print(f"  {'─'*12}{'─'*13 * len(cols_with_change)}")

        # 采样显示（每5天一个点，加首尾）
        sample_indices = set([0, len(display_dates) - 1])
        for i in range(0, len(display_dates), 5):
            sample_indices.add(i)
        sample_indices = sorted(sample_indices)

        for i in sample_indices:
            day_label = display_dates[i][-4:]  # MMDD
            if i == len(display_dates) - 1:
                day_label += "(X)"
            line = f"  {day_label:<12}"
            for c in cols_with_change:
                val = trace_r[c][i]
                if np.isnan(val):
                    line += f" {'NaN':<12}"
                elif abs(val) > 1e9:
                    line += f" {val:>11.1f}G" if abs(val) > 1e10 else f" {val:>11.0f}M"
                elif abs(val) > 1000:
                    line += f" {val:>11.0f} "
                else:
                    line += f" {val:>11.1f} "
            print(line)

        # n_ 对比
        print(f"\n  n_ 归一化值 (对比):")
        header = f"  {'Day':<12}"
        nc_map = {c.replace('r_', 'n_'): c for c in cols_with_change}
        nc_show = list(nc_map.keys())
        for c in nc_show:
            header += f" {c:<12}"
        print(header)
        print(f"  {'─'*12}{'─'*13 * len(nc_show)}")
        for i in sample_indices:
            day_label = display_dates[i][-4:]
            if i == len(display_dates) - 1:
                day_label += "(X)"
            line = f"  {day_label:<12}"
            for c in nc_show:
                val = trace_n[c][i] if c in trace_n else np.nan
                if np.isnan(val):
                    line += f" {'NaN':<12}"
                else:
                    line += f" {val:>11.2f} "
            print(line)

    gc.collect()

    # 5. 健康盘对比
    print(f"\n{'=' * 70}")
    print(f"  【健康盘时间序列】随机 {TRACE_DAYS} 天 r_ 值变化")
    print(f"{'=' * 70}")

    for _, disk_row in sample_healthy.iterrows():
        pid = str(disk_row['pair_id'])

        # 随机选一个 30 天窗口
        if len(dates) <= TRACE_DAYS:
            continue
        start_di = np.random.randint(0, len(dates) - TRACE_DAYS)
        display_dates = dates[start_di:start_di + TRACE_DAYS]

        print(f"\n  {'─' * 65}")
        print(f"  磁盘: {pid}")
        print(f"  数据范围: {display_dates[0]} ~ {display_dates[-1]}")

        trace_r = {c: [] for c in KEY_R_COLS}
        for date_str in display_dates:
            fpath = date_to_file.get(date_str)
            if fpath is None:
                continue
            try:
                df = pd.read_csv(fpath, usecols=usecols)
            except Exception:
                continue
            df['disk_id'] = df['disk_id'].astype(int).astype(str)
            df['model'] = df['model'].astype(str)
            df['_pid'] = df['disk_id'] + '_' + df['model']

            row = df[df['_pid'] == pid]
            if len(row) == 0:
                for c in KEY_R_COLS:
                    trace_r[c].append(np.nan)
                del df
                continue

            r = row.iloc[0]
            for c in KEY_R_COLS:
                val = r.get(c, np.nan)
                trace_r[c].append(float(val) if not pd.isna(val) and val != "" else np.nan)
            del df

        # 只显示有值的列
        cols_with_data = []
        for c in KEY_R_COLS:
            vals = np.array(trace_r[c])
            vals_clean = vals[~np.isnan(vals)]
            if len(vals_clean) > 0:
                cols_with_data.append(c)

        if not cols_with_data:
            print("  (该盘无任何 SMART 数据)")
            continue

        header = f"  {'Day':<12}"
        for c in cols_with_data[:8]:
            header += f" {c:<12}"
        print(f"\n  r_ 原始值 ({len(display_dates)} 天):")
        print(header)
        print(f"  {'─'*12}{'─'*13 * min(len(cols_with_data), 8)}")

        sample_indices = set([0, len(display_dates) - 1])
        for i in range(0, len(display_dates), 5):
            sample_indices.add(i)
        sample_indices = sorted(sample_indices)

        for i in sample_indices:
            day_label = display_dates[i][-4:]
            line = f"  {day_label:<12}"
            for c in cols_with_data[:8]:
                val = trace_r[c][i]
                if np.isnan(val):
                    line += f" {'NaN':<12}"
                elif abs(val) > 1e9:
                    line += f" {val:>11.1f}G"
                elif abs(val) > 1000:
                    line += f" {val:>11.0f} "
                else:
                    line += f" {val:>11.1f} "
            print(line)

    gc.collect()

    print(f"\n{'=' * 70}")
    print("  验证完成")
    print("  结论:")
    print("  - 如果故障盘 r_5/r_197/r_187 在故障前有明显上升趋势 → 数据可用")
    print("  - 如果故障盘和健康盘曲线几乎一样 → SMART 数据本身信号太弱")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()