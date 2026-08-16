# analyze_lead_time.py
# 故障信号强度分析：测量"距离故障第 l 天"的信号强度曲线
#
# 目的：
#   1. 用数据确定最优 L（TPS 扩增范围）：信号衰减到健康盘噪音水平的那个天数
#   2. 用数据确定最优 SEQ_LEN（窗口长度）：信号覆盖的时间范围
#   3. 输出特征重要性：哪些 r_ 列在故障前变化最大
#
# 方法：
#   对每个故障盘：
#     - 健康基线 = 故障前 50~30 天的特征均值（该盘"正常时期"的状态）
#     - 对 l=1..40：计算故障前第 l 天特征 vs 基线的平均绝对偏差
#   对健康盘（对照组）：
#     - 随机取时间点作为"伪故障日"，同样计算偏差曲线（自然波动水平）
#   信号强度(l) = 故障盘偏差均值(l) − 健康盘偏差均值(l)
#
# 用法：在项目根目录运行  python code/analyze_lead_time.py

import numpy as np
import pandas as pd
import os
import glob
from collections import OrderedDict
from config import *
from data_utils import (_scan_csv_dates, _load_disk_info, _load_neighbor_map,
                        _get_all_pids, PROCESSED_DIR, N_COLS)


# ============================================================
# 1. 构建 pid → 索引 映射（与 data_utils 完全一致）
# ============================================================

def build_pid_mapping(disk_info, neighbor_map, sampled_pids):
    all_needed = set(sampled_pids)
    for pid in sampled_pids:
        all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
    extract_pids = sorted(all_needed)
    pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
    return extract_pids, pid_to_extract_idx


# ============================================================
# 2. 按天特征加载（带 LRU 缓存，避免频繁 mmap 打开）
# ============================================================

class DayFeatCache:
    def __init__(self, feat_files, max_cache=20):
        self.feat_files = feat_files
        self.cache = OrderedDict()
        self.max_cache = max_cache

    def get(self, di):
        if di not in self.cache:
            self.cache[di] = np.load(self.feat_files[di], mmap_mode='r')
            if len(self.cache) > self.max_cache:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(di)
        return self.cache[di]


# ============================================================
# 3. 计算单盘偏差曲线
# ============================================================

def compute_deviation_curve(feat_cache, pid_to_idx, pid, ft_di, n_days=40, base_start=50, base_end=30):
    """
    计算某盘故障前 n_days 天相对健康基线的逐天偏差。
    返回 (l_values, deviations) 或 None（数据不足）。
    """
    idx = pid_to_idx.get(pid)
    if idx is None:
        return None

    # 健康基线：故障前 base_start~base_end 天的特征均值
    base_indices = list(range(ft_di - base_start, ft_di - base_end))
    if min(base_indices) < 0:
        return None
    base_rows = []
    for di in base_indices:
        arr = feat_cache.get(di)
        row = np.array(arr[idx], dtype=np.float32)
        base_rows.append(row)
    baseline = np.mean(np.stack(base_rows, axis=0), axis=0)  # (FEAT_DIM,)

    # 逐天偏差
    deviations = []
    l_values = []
    for l in range(1, n_days + 1):
        di = ft_di - l
        if di < 0:
            break
        arr = feat_cache.get(di)
        row = np.array(arr[idx], dtype=np.float32)

        # 全 0 行 = 该盘这天没有数据，跳过
        if np.all(row == 0):
            continue

        # 平均绝对偏差（按特征维度平均）
        dev = float(np.mean(np.abs(row - baseline)))
        deviations.append(dev)
        l_values.append(l)

    if len(deviations) < n_days // 2:
        return None
    return l_values, deviations


# ============================================================
# 4. 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  故障信号强度分析（确定最优 L 和 SEQ_LEN）")
    print("=" * 70)

    # --- 加载元数据 ---
    disk_info = _load_disk_info()
    neighbor_map = _load_neighbor_map(disk_info)
    sampled_pids = _get_all_pids(disk_info)
    extract_pids, pid_to_idx = build_pid_mapping(disk_info, neighbor_map, sampled_pids)

    # --- 加载特征文件 ---
    feat_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "feat_day_*.npy")))
    n_dates = len(feat_files)
    print(f"  日期数: {n_dates} | 提取盘数: {len(extract_pids):,} | 特征维度: {FEAT_DIM}")

    # 日期 → 索引
    date_strs = [os.path.basename(f).replace('feat_day_', '').replace('.npy', '') for f in feat_files]
    # 从 _scan_csv_dates 拿真实日期字符串
    dates, _ = _scan_csv_dates()
    dates = dates[:n_dates]
    date_to_di = {d: i for i, d in enumerate(dates)}

    feat_cache = DayFeatCache(feat_files, max_cache=20)

    # --- 收集故障盘 ---
    fail_pids = [p for p in sampled_pids if disk_info[p]['is_failure'] and disk_info[p]['failure_time'] is not None]
    print(f"  故障盘: {len(fail_pids):,}")

    # --- 计算故障盘偏差曲线 ---
    print("\n  计算故障盘偏差曲线...")
    fail_curves = []  # 每个元素是 dict {l: dev}
    n_skipped = 0
    for i, pid in enumerate(fail_pids):
        ft_str = disk_info[pid]['failure_time'].strftime("%Y%m%d")
        ft_di = date_to_di.get(ft_str)
        if ft_di is None:
            n_skipped += 1
            continue
        # 需要 ft_di 至少在前 60 天之后才有基线
        if ft_di < 60:
            n_skipped += 1
            continue

        result = compute_deviation_curve(feat_cache, pid_to_idx, pid, ft_di)
        if result is None:
            n_skipped += 1
            continue
        l_vals, devs = result
        fail_curves.append(dict(zip(l_vals, devs)))

        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(fail_pids)}] 有效曲线: {len(fail_curves)}")

    print(f"  有效故障曲线: {len(fail_curves):,} | 跳过(无数据): {n_skipped}")

    # --- 计算健康盘对照曲线 ---
    print("\n  计算健康盘对照曲线（伪故障日）...")
    healthy_pids = [p for p in sampled_pids if not disk_info[p]['is_failure']]
    rng = np.random.RandomState(42)
    # 随机抽样（全量健康盘太多，取 500 个足够估算自然波动）
    n_healthy_sample = min(500, len(healthy_pids))
    sampled_healthy = rng.choice(healthy_pids, size=n_healthy_sample, replace=False)

    healthy_curves = []
    for pid in sampled_healthy:
        # 随机选一个"伪故障日"（在数据范围后 60 天起）
        pass_di = rng.randint(60, max(61, n_dates - 1))
        result = compute_deviation_curve(feat_cache, pid_to_idx, pid, pass_di)
        if result is None:
            continue
        l_vals, devs = result
        healthy_curves.append(dict(zip(l_vals, devs)))

    print(f"  有效健康对照曲线: {len(healthy_curves)}")

    # --- 汇总输出 ---
    print(f"\n{'=' * 90}")
    print("  信号强度曲线 (故障前第 l 天):")
    print(f"  {'l':>4} | {'故障均值':>8} | {'故障中位':>8} | {'健康均值':>8} | {'信号强度':>8} | 显著性")
    print("-" * 90)

    results = []
    for l in range(1, 41):
        fail_devs = [c[l] for c in fail_curves if l in c]
        healthy_devs = [c[l] for c in healthy_curves if l in c]
        if len(fail_devs) < 50 or len(healthy_devs) < 20:
            continue

        fail_mean = float(np.mean(fail_devs))
        fail_med = float(np.median(fail_devs))
        healthy_mean = float(np.mean(healthy_devs))
        healthy_std = float(np.std(healthy_devs))
        signal = fail_mean - healthy_mean

        # 显著性判断：信号 > 2 倍健康盘标准差
        significant = "██ 强" if signal > 2 * healthy_std else (
            "▓ 中" if signal > 1 * healthy_std else "░ 弱")
        marker = " ← 信号消失" if signal < healthy_std else ""

        results.append({
            'l': l, 'fail_mean': fail_mean, 'fail_med': fail_med,
            'healthy_mean': healthy_mean, 'signal': signal,
            'healthy_std': healthy_std, 'significant': significant
        })
        bar_len = max(1, int(signal / max(healthy_std, 1e-8) * 5))
        bar = "█" * min(bar_len, 30)
        print(f"  {l:>4} | {fail_mean:>8.3f} | {fail_med:>8.3f} | "
              f"{healthy_mean:>8.3f} | {signal:>+8.3f} | {significant:>6}{bar}{marker}")

    print("=" * 90)

    # --- 推荐 L ---
    if results:
        # 找到信号开始弱于 1 倍健康标准差的最小 l
        weak_l = None
        for r in results:
            if r['signal'] < r['healthy_std']:
                weak_l = r['l']
                break
        # 找到信号开始弱于 2 倍健康标准差的最小 l
        weak_l_2 = None
        for r in results:
            if r['signal'] < 2 * r['healthy_std']:
                weak_l_2 = r['l']
                break

        print("\n  📌 推荐结论:")
        print(f"     - 信号完全消失（< 1σ）于 l={weak_l}")
        print(f"     - 信号变弱（< 2σ）于 l={weak_l_2}")
        if weak_l_2:
            print(f"     → 建议 L 取 {max(1, weak_l_2 - 1)} ~ {max(1, weak_l_2)}（在信号变弱之前）")
            print(f"     → 建议 SEQ_LEN 取 {min(15, max(5, weak_l_2 + 5))} ~ {weak_l_2 + 10}（覆盖信号全程）")
        else:
            print(f"     → 40 天内信号都较强，可能需要延长分析范围")

    # --- 特征重要性 ---
    print(f"\n{'=' * 80}")
    print("  特征重要性（故障前 7 天，各特征平均绝对偏差 vs 健康基线）:")
    print("-" * 80)

    # 收集所有故障盘的 (基线, 故障前7天) 特征
    feat_importance = np.zeros(FEAT_DIM, dtype=np.float64)
    n_feat_discs = 0
    for pid in fail_pids[:2000]:  # 取前 2000 个故障盘足够
        ft_str = disk_info[pid]['failure_time'].strftime("%Y%m%d")
        ft_di = date_to_di.get(ft_str)
        if ft_di is None or ft_di < 60:
            continue
        idx = pid_to_idx.get(pid)
        if idx is None:
            continue

        # 基线
        base_rows = []
        for di in range(ft_di - 50, ft_di - 30):
            arr = feat_cache.get(di)
            base_rows.append(np.array(arr[idx], dtype=np.float32))
        baseline = np.mean(np.stack(base_rows, axis=0), axis=0)

        # 故障前 7 天平均特征
        fail_rows = []
        for di in range(ft_di - 7, ft_di):
            arr = feat_cache.get(di)
            fail_rows.append(np.array(arr[idx], dtype=np.float32))
        fail_avg = np.mean(np.stack(fail_rows, axis=0), axis=0)

        feat_importance += np.abs(fail_avg - baseline)
        n_feat_discs += 1

    if n_feat_discs > 0:
        feat_importance /= n_feat_discs
        # 排序输出前 15
        order = np.argsort(-feat_importance)
        for rank, ci in enumerate(order[:15]):
            sid = N_COLS[ci]
            print(f"    #{rank+1:>2}  {sid:<8}  偏差 {feat_importance[ci]:.4f}")
    print("=" * 80)

    print("\n  ✅ 分析完成！根据上表确定最优 L 和 SEQ_LEN")


if __name__ == "__main__":
    main()