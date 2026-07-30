# analyze_smart.py
# SMART 数据质量分析工具
# 六层报告：
#   1. 列级 NaN / 合法 0 统计
#   2. 磁盘（pair_id）级缺失分析
#   3. Model 级缺失分析
#   4. 故障盘 vs 健康盘 分布偏移
#   5. 时间序列偏移
#   6. 总结 & 建议
#
# 核心：区分「真缺失（NaN/空）」和「合法值 0」

import numpy as np
import pandas as pd
import os
import gc
import json
from collections import defaultdict
from config import *

DATA_DIR = "D:/2018Datasets"
PROCESSED_DIR = "datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")

# SMART 属性 ID → 名称映射（常见 SMART 属性）
SMART_NAME_MAP = {
    1: "Read Error Rate",
    2: "Throughput Performance",
    3: "Spin-Up Time",
    4: "Start/Stop Count",
    5: "Reallocated Sectors Count",
    6: "Read Channel Margin",
    7: "Seek Error Rate",
    8: "Seek Time Performance",
    9: "Power-On Hours",
    10: "Spin Retry Count",
    11: "Recalibration Retries",
    12: "Power Cycle Count",
    13: "Soft Read Error Rate",
    170: "Available Reserved Space",
    171: "SSD Program Fail Count",
    172: "SSD Erase Fail Count",
    173: "SSD Wear Leveling Count",
    174: "Unexpected Power Loss Count",
    175: "Power Loss Protection Failure",
    177: "Wear Range Delta",
    180: "Unused Reserved Block Count Total",
    181: "Program Fail Count Total",
    182: "Erase Fail Count",
    183: "SATA Downshift Error Count",
    184: "End-to-End Error",
    187: "Reported Uncorrectable Errors",
    188: "Command Timeout",
    189: "High Fly Writes",
    190: "Airflow Temperature",
    191: "G-Sense Error Rate",
    192: "Power-Off Retract Cycle",
    193: "Load/Unload Cycle Count",
    194: "Temperature",
    195: "Hardware ECC Recovered",
    196: "Reallocation Event Count",
    197: "Current Pending Sector Count",
    198: "Offline Scan Uncorrectable Count",
    199: "UltraDMA CRC Error Count",
    200: "Write Error Rate / Multi-Zone Error Rate",
    204: "Soft ECC Correction",
    205: "Thermal Asperity Rate",
    206: "Flying Height",
    207: "Spin High Current",
    211: "Multi-zone Error Rate",
    232: "Endurance Remaining",
    233: "Media Wearout Indicator",
    240: "Head Flying Hours / Transfer Rate",
    241: "Total LBAs Written",
    242: "Total LBAs Read",
    244: "Temperature Difference",
    245: "New Attribute"
}

N_COLS = [f"n_{sid}" for sid in [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    170, 171, 172, 173, 174, 175,
    177, 180, 181, 182, 183, 184, 187, 188, 189,
    190, 191, 192, 193, 194, 195, 196, 197, 198,
    199, 200, 204, 205, 206, 207, 211,
    232, 233, 240, 241, 242, 244, 245
]]


def col_desc(col_name):
    """返回列的可读描述"""
    sid = int(col_name.split('_')[1])
    name = SMART_NAME_MAP.get(sid, f"Unknown({sid})")
    return f"{col_name} ({name})"


# ============================================================
# 第一层：列级 NaN / 合法 0 统计
# ============================================================

def analyze_columns():
    print("=" * 70)
    print("【第一层】列级 NaN / 合法 0 统计")
    print("=" * 70)

    # 扫描所有 CSV 文件
    csv_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.csv') and f.startswith('2018')
        and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END
    ])

    if not csv_files:
        print("  ❌ 未找到任何 CSV 文件！请检查 DATA_DIR 和 DATA_MONTH 配置")
        return None

    print(f"  找到 {len(csv_files)} 个 CSV 文件")
    print(f"  数据月份范围: {DATA_MONTH_START} ~ {DATA_MONTH_END}")

    # 累积统计
    col_total = {c: 0 for c in N_COLS}       # 总行数
    col_nan = {c: 0 for c in N_COLS}           # NaN 数
    col_zero = {c: 0 for c in N_COLS}          # 合法 0 数
    monthly_stats = {}                          # 每月统计
    monthly_nan = {}                            # 每月每列 NaN 率（第五层用）

    usecols = ['model'] + N_COLS

    for fi, fname in enumerate(csv_files):
        date_str = fname[:8]
        fpath = os.path.join(DATA_DIR, fname)
        print(f"  [{fi+1}/{len(csv_files)}] {date_str} ...", end=" ")

        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception as e:
            print(f"跳过 (读取失败: {e})")
            continue

        n_rows = len(df)
        print(f"{n_rows:,} 行")

        # 逐列统计
        month_nan_rate = {}
        for c in N_COLS:
            if c in df.columns:
                nan_count = df[c].isna().sum()
                # 空字符串也视为缺失
                empty_count = (df[c] == "").sum() if df[c].dtype == object else 0
                total_missing = nan_count + empty_count
                zero_count = (df[c] == 0).sum()  # 合法 0

                col_total[c] += n_rows
                col_nan[c] += total_missing
                col_zero[c] += zero_count
                month_nan_rate[c] = total_missing / max(n_rows, 1) * 100
            else:
                col_total[c] += n_rows
                col_nan[c] += n_rows  # 列不存在 = 全缺失
                month_nan_rate[c] = 100.0

        monthly_stats[date_str] = n_rows
        monthly_nan[date_str] = month_nan_rate

        del df
        gc.collect()

    # ---- 输出表格 ----
    print(f"\n  {'列名':<10} {'总行数':>10} {'NaN数':>10} {'NaN率':>8} {'合法0数':>10} {'合法0率':>8} {'建议':>12}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*10} {'─'*8} {'─'*12}")

    col_analysis = {}  # 供后续层使用
    for c in N_COLS:
        total = col_total[c]
        nan_count = col_nan[c]
        zero_count = col_zero[c]
        nan_rate = nan_count / max(total, 1) * 100
        zero_rate = zero_count / max(total, 1) * 100

        if nan_rate >= 90:
            suggestion = "🔴 剔除"
        elif nan_rate >= 50:
            suggestion = "🟡 保留但关注"
        else:
            suggestion = "🟢 保留"

        print(f"  {c:<10} {total:>10,} {nan_count:>10,} {nan_rate:>7.1f}% {zero_count:>10,} {zero_rate:>7.1f}% {suggestion:>12}")

        col_analysis[c] = {
            'total': int(total),
            'nan_count': int(nan_count),
            'nan_rate': round(nan_rate, 2),
            'zero_count': int(zero_count),
            'zero_rate': round(zero_rate, 2),
            'suggestion': suggestion
        }

    # 汇总
    discard = [c for c in N_COLS if col_analysis[c]['nan_rate'] >= 90]
    keep_attention = [c for c in N_COLS if 50 <= col_analysis[c]['nan_rate'] < 90]
    keep_good = [c for c in N_COLS if col_analysis[c]['nan_rate'] < 50]

    print(f"\n  📊 汇总:")
    print(f"    🔴 建议剔除 (NaN率≥90%): {len(discard)} 列 → {', '.join(discard) if discard else '无'}")
    print(f"    🟡 保留但关注 (50%≤NaN率<90%): {len(keep_attention)} 列")
    print(f"    🟢 保留 (NaN率<50%): {len(keep_good)} 列")
    print(f"    有效特征维度: {len(N_COLS)} → {len(keep_good) + len(keep_attention)} (剔除 {len(discard)} 列)")

    return {
        'col_analysis': col_analysis,
        'discard_cols': discard,
        'keep_attention_cols': keep_attention,
        'keep_good_cols': keep_good,
        'monthly_nan': monthly_nan,
        'csv_files': csv_files
    }


# ============================================================
# 第二层：磁盘（pair_id）级缺失分析
# ============================================================

def analyze_disks(col_result):
    print("\n" + "=" * 70)
    print("【第二层】磁盘（pair_id）级缺失分析")
    print("=" * 70)

    if not os.path.exists(TARGET_FILE):
        print("  ⚠️ target_disks.csv 不存在，跳过磁盘级分析")
        return None

    target_df = pd.read_csv(TARGET_FILE)
    target_df['pair_id'] = target_df['pair_id'].astype(str)
    target_pids = set(target_df['pair_id'].tolist())
    print(f"  target_disks.csv 中有 {len(target_pids):,} 块盘")

    # 构建 model lookup
    pid_to_model = {}
    pid_to_is_failure = {}
    for _, row in target_df.iterrows():
        pid = str(row['pair_id'])
        pid_to_model[pid] = str(row['model'])
        pid_to_is_failure[pid] = bool(row['is_failure'])

    csv_files = col_result['csv_files']
    usecols = ['disk_id', 'model'] + N_COLS

    # 每块盘的累积统计
    disk_nan = defaultdict(lambda: {'rows': 0, 'col_nan': defaultdict(int), 'col_total': defaultdict(int)})

    print(f"  扫描 {len(csv_files)} 个 CSV 查找目标盘...")
    for fi, fname in enumerate(csv_files):
        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception:
            continue

        df['disk_id'] = df['disk_id'].astype(int).astype(str)
        df['model'] = df['model'].astype(str)
        df['_pid'] = df['disk_id'] + '_' + df['model']

        # 只保留 target 中的盘
        df_target = df[df['_pid'].isin(target_pids)]
        if len(df_target) == 0:
            del df
            continue

        for _, row in df_target.iterrows():
            pid = row['_pid']
            disk_nan[pid]['rows'] += 1
            for c in N_COLS:
                val = row.get(c)
                is_missing = pd.isna(val) or val == ""
                if is_missing:
                    disk_nan[pid]['col_nan'][c] += 1
                disk_nan[pid]['col_total'][c] += 1

        del df, df_target
        gc.collect()
        if (fi + 1) % 3 == 0:
            print(f"    [{fi+1}/{len(csv_files)}] 已处理, 找到 {len(disk_nan):,} 块目标盘")

    print(f"  共找到 {len(disk_nan):,} 块目标盘")

    # 计算每块盘的统计量
    disk_stats = []
    for pid, stats in disk_nan.items():
        total_rows = stats['rows']
        nan_cols_count = 0
        total_nan = 0
        total_cells = 0
        col_nan_rates = {}
        for c in N_COLS:
            cn = stats['col_nan'][c]
            ct = stats['col_total'][c]
            total_nan += cn
            total_cells += ct
            if ct > 0 and cn / ct >= 0.9:
                nan_cols_count += 1
            col_nan_rates[c] = cn / max(ct, 1) * 100

        overall_nan_rate = total_nan / max(total_cells, 1) * 100
        is_failure = pid_to_is_failure.get(pid, False)

        disk_stats.append({
            'pid': pid,
            'model': pid_to_model.get(pid, 'Unknown'),
            'is_failure': is_failure,
            'rows': total_rows,
            'nan_cols_count': nan_cols_count,
            'overall_nan_rate': round(overall_nan_rate, 2),
            'col_nan_rates': col_nan_rates
        })

    # 排序：按 NaN 率从高到低
    disk_stats.sort(key=lambda x: x['overall_nan_rate'], reverse=True)

    # ---- 全局统计 ----
    nan_rates = [d['overall_nan_rate'] for d in disk_stats]
    nan_cols_counts = [d['nan_cols_count'] for d in disk_stats]

    print(f"\n  📊 磁盘 NaN 率分布:")
    print(f"    均值: {np.mean(nan_rates):.1f}%")
    print(f"    中位数: {np.median(nan_rates):.1f}%")
    print(f"    最小: {np.min(nan_rates):.1f}% | 最大: {np.max(nan_rates):.1f}%")

    # 分桶
    buckets = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 100)]
    print(f"\n  📊 磁盘 NaN 率分段分布:")
    for lo, hi in buckets:
        count = sum(1 for r in nan_rates if lo <= r < hi)
        pct = count / max(len(nan_rates), 1) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lo:>3}%-{hi:>3}%: {count:>6,} 块盘 ({pct:>5.1f}%) {bar}")

    print(f"\n  📊 磁盘 NaN 列数分布 (共 {len(N_COLS)} 列):")
    print(f"    均值: {np.mean(nan_cols_counts):.1f} 列 | 中位数: {np.median(nan_cols_counts):.1f} 列")
    print(f"    最多: {np.max(nan_cols_counts)} 列 | 最少: {np.min(nan_cols_counts)} 列")

    # ---- 缺失最严重的前 20 块盘 ----
    print(f"\n  🔴 缺失最严重的前 20 块盘:")
    print(f"    {'pair_id':<25} {'Model':<12} {'故障':>4} {'行数':>6} {'NaN列':>6} {'NaN率':>8}")
    print(f"    {'─'*25} {'─'*12} {'─'*4} {'─'*6} {'─'*6} {'─'*8}")
    for d in disk_stats[:20]:
        fail_mark = "✓" if d['is_failure'] else ""
        print(f"    {d['pid']:<25} {d['model']:<12} {fail_mark:>4} {d['rows']:>6,} {d['nan_cols_count']:>6} {d['overall_nan_rate']:>7.1f}%")

    # ---- 数据最完整的前 20 块盘 ----
    print(f"\n  🟢 数据最完整的前 20 块盘:")
    print(f"    {'pair_id':<25} {'Model':<12} {'故障':>4} {'行数':>6} {'NaN列':>6} {'NaN率':>8}")
    print(f"    {'─'*25} {'─'*12} {'─'*4} {'─'*6} {'─'*6} {'─'*8}")
    for d in disk_stats[-20:]:
        fail_mark = "✓" if d['is_failure'] else ""
        print(f"    {d['pid']:<25} {d['model']:<12} {fail_mark:>4} {d['rows']:>6,} {d['nan_cols_count']:>6} {d['overall_nan_rate']:>7.1f}%")

    # ---- 故障盘 vs 健康盘缺失对比 ----
    fail_disks = [d for d in disk_stats if d['is_failure']]
    healthy_disks = [d for d in disk_stats if not d['is_failure']]
    print(f"\n  📊 故障盘 vs 健康盘 缺失对比:")
    print(f"    故障盘: {len(fail_disks):,} 块, 平均NaN率 {np.mean([d['overall_nan_rate'] for d in fail_disks]):.1f}%")
    print(f"    健康盘: {len(healthy_disks):,} 块, 平均NaN率 {np.mean([d['overall_nan_rate'] for d in healthy_disks]):.1f}%")

    return disk_stats


# ============================================================
# 第三层：Model 级缺失分析
# ============================================================

def analyze_models(disk_stats, col_result):
    print("\n" + "=" * 70)
    print("【第三层】Model 级缺失分析")
    print("=" * 70)

    if disk_stats is None:
        print("  ⚠️ 无磁盘级数据，跳过 Model 级分析")
        return None

    # 按 model 分组
    model_groups = defaultdict(list)
    for d in disk_stats:
        model_groups[d['model']].append(d)

    print(f"  共 {len(model_groups)} 种 Model")

    # 计算每种 model 的统计
    model_stats = []
    for model, disks in model_groups.items():
        n_disks = len(disks)
        n_fail = sum(1 for d in disks if d['is_failure'])
        fail_rate = n_fail / n_disks * 100

        avg_nan_rate = np.mean([d['overall_nan_rate'] for d in disks])
        avg_nan_cols = np.mean([d['nan_cols_count'] for d in disks])

        # 每列在该 model 下的平均 NaN 率
        col_nan_model = {}
        nan90_cols = []
        for c in N_COLS:
            rates = [d['col_nan_rates'].get(c, 0) for d in disks]
            avg_rate = np.mean(rates) if rates else 0
            col_nan_model[c] = round(avg_rate, 2)
            if avg_rate >= 90:
                nan90_cols.append(c)

        model_stats.append({
            'model': model,
            'n_disks': n_disks,
            'n_fail': n_fail,
            'fail_rate': round(fail_rate, 2),
            'avg_nan_rate': round(avg_nan_rate, 2),
            'avg_nan_cols': round(avg_nan_cols, 1),
            'nan90_cols': nan90_cols,
            'col_nan_model': col_nan_model
        })

    # 按盘数排序
    model_stats.sort(key=lambda x: x['n_disks'], reverse=True)

    # ---- 输出表格 ----
    print(f"\n  {'Model':<20} {'盘数':>8} {'故障盘':>8} {'故障率':>8} {'平均NaN率':>10} {'平均NaN列':>10} {'NaN≥90%列数':>12}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*12}")

    for ms in model_stats[:30]:  # 只显示前 30 种 model
        print(f"  {ms['model']:<20} {ms['n_disks']:>8,} {ms['n_fail']:>8,} {ms['fail_rate']:>7.1f}% "
              f"{ms['avg_nan_rate']:>9.1f}% {ms['avg_nan_cols']:>9.1f} {len(ms['nan90_cols']):>12}")

    if len(model_stats) > 30:
        print(f"  ... 还有 {len(model_stats) - 30} 种 model 未显示")

    # ---- NaN 列最多的 Model Top 10 ----
    model_by_nan = sorted(model_stats, key=lambda x: len(x['nan90_cols']), reverse=True)
    print(f"\n  🔴 NaN≥90%列数最多的 Model Top 10:")
    for ms in model_by_nan[:10]:
        cols_str = ', '.join(ms['nan90_cols'][:5])
        if len(ms['nan90_cols']) > 5:
            cols_str += f" ... +{len(ms['nan90_cols']) - 5}列"
        print(f"    {ms['model']:<20} NaN≥90%列: {len(ms['nan90_cols'])} → {cols_str}")

    # ---- Model 间 NaN 率差异最大的列 ----
    print(f"\n  📊 Model 间 NaN 率差异最大的列 (标准差):")
    col_std_across_models = {}
    for c in N_COLS:
        rates = [ms['col_nan_model'].get(c, 0) for ms in model_stats]
        col_std_across_models[c] = np.std(rates)
    top_diff_cols = sorted(col_std_across_models.items(), key=lambda x: x[1], reverse=True)

    for c, std_val in top_diff_cols[:10]:
        rates = [ms['col_nan_model'].get(c, 0) for ms in model_stats]
        print(f"    {col_desc(c):<45} std={std_val:.1f}%  "
              f"(min={np.min(rates):.0f}%, max={np.max(rates):.0f}%)")

    # ---- 故障率最高的 Model ----
    model_by_fail = sorted(model_stats, key=lambda x: x['fail_rate'], reverse=True)
    print(f"\n  🔴 故障率最高的 Model Top 10 (盘数≥10):")
    count = 0
    for ms in model_by_fail:
        if ms['n_disks'] >= 10:
            print(f"    {ms['model']:<20} {ms['n_disks']:>6,}块 故障率 {ms['fail_rate']:>6.1f}%  "
                  f"NaN率 {ms['avg_nan_rate']:>5.1f}%")
            count += 1
            if count >= 10:
                break

    return model_stats


# ============================================================
# 第四层：故障盘 vs 健康盘 分布偏移
# ============================================================

def analyze_failure_vs_healthy(col_result):
    print("\n" + "=" * 70)
    print("【第四层】故障盘 vs 健康盘 分布偏移")
    print("=" * 70)

    if not os.path.exists(TARGET_FILE):
        print("  ⚠️ target_disks.csv 不存在，跳过分布偏移分析")
        return None

    target_df = pd.read_csv(TARGET_FILE)
    target_df['pair_id'] = target_df['pair_id'].astype(str)

    fail_pids = set(target_df[target_df['is_failure'] == True]['pair_id'].tolist())
    healthy_pids = set(target_df[target_df['is_failure'] == False]['pair_id'].tolist())
    print(f"  故障盘: {len(fail_pids):,} | 健康盘: {len(healthy_pids):,}")

    csv_files = col_result['csv_files']
    usecols = ['disk_id', 'model'] + N_COLS

    # 只用训练集时间范围的数据（≤ TRAIN_CUTOFF）做分布分析
    # 收集故障盘和健康盘的所有非缺失值
    fail_values = {c: [] for c in N_COLS}
    healthy_values = {c: [] for c in N_COLS}

    print(f"  扫描 CSV 收集 SMART 值...")
    for fi, fname in enumerate(csv_files):
        date_str = fname[:8]
        if date_str > TRAIN_CUTOFF:
            continue  # 只用训练集数据

        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath, usecols=usecols)
        except Exception:
            continue

        df['disk_id'] = df['disk_id'].astype(int).astype(str)
        df['model'] = df['model'].astype(str)
        df['_pid'] = df['disk_id'] + '_' + df['model']

        df_fail = df[df['_pid'].isin(fail_pids)]
        df_healthy = df[df['_pid'].isin(healthy_pids)]

        for c in N_COLS:
            if c in df.columns:
                fv = df_fail[c].dropna().values
                hv = df_healthy[c].dropna().values
                # 过滤掉空字符串
                fv = fv[fv != ""].astype(float) if len(fv) > 0 else np.array([])
                hv = hv[hv != ""].astype(float) if len(hv) > 0 else np.array([])
                fail_values[c].extend(fv.tolist())
                healthy_values[c].extend(hv.tolist())

        del df, df_fail, df_healthy
        gc.collect()
        if (fi + 1) % 3 == 0:
            print(f"    [{fi+1}/{len(csv_files)}] 已处理")

    # 对每列做 KS 检验和 Cohen's d
    from scipy import stats

    shift_results = []
    print(f"\n  {'列名':<25} {'健康样本':>10} {'故障样本':>10} {'健康0值率':>10} {'故障0值率':>10} {'KS p值':>8} {'判别力':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

    for c in N_COLS:
        fv = np.array(fail_values[c])
        hv = np.array(healthy_values[c])

        if len(fv) < 10 or len(hv) < 10:
            shift_results.append({'col': c, 'discriminative': '⬜ 样本不足'})
            print(f"  {col_desc(c):<25} {len(hv):>10,} {len(fv):>10,} {'─':>10} {'─':>10} {'─':>8} {'⬜ 样本不足':>8}")
            continue

        # 0 值率
        fv_zero_rate = (fv == 0).sum() / len(fv) * 100
        hv_zero_rate = (hv == 0).sum() / len(hv) * 100

        # 过滤掉 0 值后做 KS 检验（0 值太多会掩盖分布差异）
        fv_nz = fv[fv != 0] if len(fv[fv != 0]) > 0 else fv
        hv_nz = hv[hv != 0] if len(hv[hv != 0]) > 0 else hv

        try:
            ks_stat, ks_p = stats.ks_2samp(fv_nz, hv_nz)
        except Exception:
            ks_p = 1.0

        # 判别力判断
        if ks_p < 0.001:
            disc = "🔴 强"
        elif ks_p < 0.05:
            disc = "🟡 中等"
        else:
            disc = "⬜ 无区分力"

        shift_results.append({
            'col': c,
            'healthy_n': len(hv),
            'fail_n': len(fv),
            'healthy_zero_rate': round(hv_zero_rate, 2),
            'fail_zero_rate': round(fv_zero_rate, 2),
            'ks_p': round(ks_p, 6),
            'discriminative': disc
        })

        print(f"  {col_desc(c):<25} {len(hv):>10,} {len(fv):>10,} "
              f"{hv_zero_rate:>9.1f}% {fv_zero_rate:>9.1f}% {ks_p:>8.4f} {disc:>8}")

    # 汇总
    strong = sum(1 for r in shift_results if r.get('discriminative') == '🔴 强')
    medium = sum(1 for r in shift_results if r.get('discriminative') == '🟡 中等')
    none_disc = sum(1 for r in shift_results if r.get('discriminative') == '⬜ 无区分力')
    print(f"\n  📊 汇总: 🔴强区分力 {strong}列 | 🟡中等 {medium}列 | ⬜无区分力 {none_disc}列")

    return shift_results


# ============================================================
# 第五层：时间序列偏移
# ============================================================

def analyze_temporal(col_result):
    print("\n" + "=" * 70)
    print("【第五层】时间序列偏移")
    print("=" * 70)

    monthly_nan = col_result['monthly_nan']

    # ---- 每月 NaN 率变化 ----
    dates_sorted = sorted(monthly_nan.keys())
    print(f"\n  📊 每月 NaN 率变化 (选取 NaN 率最高的 5 列):")

    # 找出全局 NaN 率最高的 5 列
    avg_nan_across_months = {}
    for c in N_COLS:
        rates = [monthly_nan[d].get(c, 100) for d in dates_sorted]
        avg_nan_across_months[c] = np.mean(rates)
    top5_cols = sorted(avg_nan_across_months, key=avg_nan_across_months.get, reverse=True)[:5]

    print(f"    {'月份':<10} " + " ".join([f"{col_desc(c):<25}" for c in top5_cols]))
    print(f"    {'─'*10} " + " ".join(["─"*25 for _ in top5_cols]))
    for d in dates_sorted:
        rates_str = " ".join([f"{monthly_nan[d].get(c, 100):>24.1f}%" for c in top5_cols])
        print(f"    {d:<10} {rates_str}")

    # ---- NaN 率漂移检测 ----
    print(f"\n  📊 时间漂移检测 (每月平均 NaN 率):")
    for d in dates_sorted:
        rates = [monthly_nan[d].get(c, 100) for c in N_COLS]
        avg_rate = np.mean(rates)
        bar = "█" * int(avg_rate / 2)
        print(f"    {d}: {avg_rate:>5.1f}% {bar}")

    # 判断是否有明显漂移
    first_month_avg = np.mean([monthly_nan[dates_sorted[0]].get(c, 100) for c in N_COLS])
    last_month_avg = np.mean([monthly_nan[dates_sorted[-1]].get(c, 100) for c in N_COLS])
    drift = last_month_avg - first_month_avg
    print(f"\n    首月平均 NaN: {first_month_avg:.1f}% → 末月平均 NaN: {last_month_avg:.1f}%")
    if abs(drift) > 5:
        direction = "上升" if drift > 0 else "下降"
        print(f"    ⚠️ 存在明显时间漂移 ({direction} {abs(drift):.1f}%)，建议检查数据源变化")


# ============================================================
# 第六层：总结 & 建议
# ============================================================

def summarize(col_result, disk_stats, model_stats, shift_results):
    print("\n" + "=" * 70)
    print("【第六层】总结 & 建议")
    print("=" * 70)

    col_analysis = col_result['col_analysis']
    discard_cols = col_result['discard_cols']
    keep_attention_cols = col_result['keep_attention_cols']
    keep_good_cols = col_result['keep_good_cols']

    # ---- 推荐特征列 ----
    recommended = keep_good_cols + keep_attention_cols
    print(f"\n  📋 推荐特征列 ({len(recommended)}/{len(N_COLS)}):")
    print(f"     ❌ 剔除 (NaN≥90%): {len(discard_cols)} 列")
    for c in discard_cols:
        print(f"        {col_desc(c)}: NaN率 {col_analysis[c]['nan_rate']:.1f}%")
    print(f"     🟢 保留 ({len(recommended)} 列): FEAT_DIM = {len(recommended)}")
    print(f"        RECOMMENDED_COLS = {recommended}")

    # 如果做了分布偏移分析，进一步筛选
    if shift_results:
        no_disc = [r['col'] for r in shift_results if r.get('discriminative') == '⬜ 无区分力']
        print(f"\n     ⬜ 无区分力列 (KS p>0.05): {len(no_disc)} 列")
        for c in no_disc:
            r = next(x for x in shift_results if x['col'] == c)
            print(f"        {col_desc(c)}: KS p={r['ks_p']:.4f}")

        # 强区分力列
        strong_disc = [r['col'] for r in shift_results if r.get('discriminative') == '🔴 强']
        print(f"\n     🔴 强区分力列 (KS p<0.001, 故障vs健康差异显著): {len(strong_disc)} 列")
        for c in strong_disc:
            r = next(x for x in shift_results if x['col'] == c)
            print(f"        {col_desc(c)}: 健康0值率 {r['healthy_zero_rate']:.1f}% vs 故障0值率 {r['fail_zero_rate']:.1f}%")

    # ---- 数据质量警告 ----
    print(f"\n  ⚠️ 数据质量警告:")

    if disk_stats:
        high_nan_disks = [d for d in disk_stats if d['overall_nan_rate'] > 50]
        if high_nan_disks:
            print(f"    • {len(high_nan_disks)} 块盘 NaN 率 > 50%")
            # 按 model 分布
            hnd_models = defaultdict(int)
            for d in high_nan_disks:
                hnd_models[d['model']] += 1
            for m, cnt in sorted(hnd_models.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {m}: {cnt} 块")

        # 全空盘
        all_nan_disks = [d for d in disk_stats if d['overall_nan_rate'] > 95]
        if all_nan_disks:
            print(f"    • {len(all_nan_disks)} 块盘几乎全空 (NaN率 > 95%)，建议从训练集中移除")

    # ---- 模型训练建议 ----
    print(f"\n  💡 模型训练建议:")

    # 特征维度
    print(f"    1. FEAT_DIM 从 51 调整为 {len(recommended)}")
    print(f"       在 config.py 中修改: FEAT_DIM = {len(recommended)}")
    print(f"       在 data_utils.py 中修改 N_COLS 为推荐列列表")

    # NaN 处理策略
    nan_cols_present = len(discard_cols) > 0
    if nan_cols_present:
        print(f"    2. 剔除 {len(discard_cols)} 列高 NaN 列后，剩余的 NaN 可能来自:")
        print(f"       - 部分 model 不支持某些 SMART 属性 → 按 model 做缺失值填充策略")
        print(f"       - 偶发数据采集失败 → fillna(0) 可以接受")

    # pos_weight 建议
    if disk_stats:
        n_fail = sum(1 for d in disk_stats if d['is_failure'])
        n_healthy = sum(1 for d in disk_stats if not d['is_failure'])
        if n_fail > 0:
            imbalance_ratio = n_healthy / n_fail
            print(f"    3. 正负样本不平衡比约 1:{imbalance_ratio:.0f}")
            print(f"       建议在 BCEWithLogitsLoss 中设置 pos_weight={imbalance_ratio:.1f}")

    # 时间漂移
    if abs(col_result.get('drift', 0)) > 5:
        print(f"    4. ⚠️ 数据存在时间漂移，考虑按月份做特征归一化或使用更近期的数据训练")

    # 保存结果
    print(f"\n  📁 分析结果已保存至: {os.path.join(PROCESSED_DIR, 'smart_analysis.json')}")
    print(f"  📁 可读报告已保存至: {os.path.join(PROCESSED_DIR, 'smart_analysis.txt')}")


# ============================================================
# 主入口
# ============================================================

def save_results(col_result, disk_stats, model_stats, shift_results):
    """保存 JSON 和 TXT 格式的分析结果"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    json_path = os.path.join(PROCESSED_DIR, "smart_analysis.json")
    txt_path = os.path.join(PROCESSED_DIR, "smart_analysis.txt")

    # 构建 JSON 输出
    output = {}

    # 列分析
    output['column_analysis'] = col_result['col_analysis']
    output['discard_cols'] = col_result['discard_cols']
    output['keep_attention_cols'] = col_result['keep_attention_cols']
    output['keep_good_cols'] = col_result['keep_good_cols']
    output['recommended_cols'] = col_result['keep_good_cols'] + col_result['keep_attention_cols']
    output['recommended_feat_dim'] = len(output['recommended_cols'])

    # 磁盘分析（只保存摘要）
    if disk_stats:
        output['disk_summary'] = {
            'total_disks': len(disk_stats),
            'avg_nan_rate': round(np.mean([d['overall_nan_rate'] for d in disk_stats]), 2),
            'median_nan_rate': round(np.median([d['overall_nan_rate'] for d in disk_stats]), 2),
            'high_nan_disks': len([d for d in disk_stats if d['overall_nan_rate'] > 50]),
            'all_nan_disks': len([d for d in disk_stats if d['overall_nan_rate'] > 95]),
            'top20_worst': [{'pid': d['pid'], 'model': d['model'], 'nan_rate': d['overall_nan_rate'],
                             'nan_cols': d['nan_cols_count'], 'is_failure': d['is_failure']}
                            for d in disk_stats[:20]]
        }

    # Model 分析
    if model_stats:
        output['model_analysis'] = [
            {'model': ms['model'], 'n_disks': ms['n_disks'], 'n_fail': ms['n_fail'],
             'fail_rate': ms['fail_rate'], 'avg_nan_rate': ms['avg_nan_rate'],
             'avg_nan_cols': ms['avg_nan_cols'], 'nan90_cols': ms['nan90_cols']}
            for ms in model_stats
        ]

    # 分布偏移
    if shift_results:
        output['failure_vs_healthy'] = shift_results

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✅ JSON 已保存: {json_path}")

    # # 保存 TXT 可读报告（包含推荐列列表，方便直接复制）
    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     f.write("=" * 70 + "\n")
    #     f.write("SMART 数据分析报告\n")
    #     f.write("=" * 70 + "\n\n")

    #     f.write(f"推荐特征维度: {output['recommended_feat_dim']}\n\n")
    #     f.write("推荐特征列列表 (可直接复制到 config.py 和 data_utils.py):\n")
    #     f.write(f"RECOMMENDED_COLS = {output['recommended_cols']}\n\n")
    #     f.write(f"FEAT_DIM = {output['recommended_feat_dim']}\n\n")

    #     f.write(f"剔除列 (NaN率≥90%): {output['discard_cols']}\n")
    #     f.write(f"保留但关注列 (50%≤NaN率<90%): {output['keep_attention_cols']}\n")
    #     f.write(f"保留列 (NaN率<50%): {output['keep_good_cols']}\n")

    # print(f"  ✅ TXT 已保存: {txt_path}")


def main():
    print("=" * 70)
    print("  SMART 数据质量分析工具")
    print("  数据目录: " + DATA_DIR)
    print("  特征列数: " + str(len(N_COLS)))
    print("=" * 70 + "\n")

    import time
    t_start = time.time()

    # 第一层
    col_result = analyze_columns()
    if col_result is None:
        return

    # # 第二层
    # disk_stats = analyze_disks(col_result)

    # # 第三层
    # model_stats = analyze_models(disk_stats, col_result)

    # 第四层
    shift_results = analyze_failure_vs_healthy(col_result)

    # 第五层
    analyze_temporal(col_result)

    # 计算时间漂移（汇总用）
    monthly_nan = col_result['monthly_nan']
    dates_sorted = sorted(monthly_nan.keys())
    if len(dates_sorted) >= 2:
        first_avg = np.mean([monthly_nan[dates_sorted[0]].get(c, 100) for c in N_COLS])
        last_avg = np.mean([monthly_nan[dates_sorted[-1]].get(c, 100) for c in N_COLS])
        col_result['drift'] = last_avg - first_avg
    else:
        col_result['drift'] = 0

    # # 第六层
    # summarize(col_result, disk_stats, model_stats, shift_results)

    # # 保存
    # save_results(col_result, disk_stats, model_stats, shift_results)

    t_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  分析完成！耗时 {t_elapsed:.0f} 秒 ({t_elapsed/60:.1f} 分钟)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()