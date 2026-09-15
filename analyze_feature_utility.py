#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_feature_utility.py —— 逐 SMART 列的"效用画像"（纯只读分析，无需训练）

回答的问题:
  Q1  30 列里哪些列真的对"故障判别"有贡献? 哪些是噪声/常量/冗余?
  Q2  "时间/累计量"类列(9/12/241/242 ...) 单独看有没有用?
      (直觉: 单独判别力低, 但它们是别的特征的基准 —— 比率/速率的分母)
  Q3  哪些列可能只反映"机架/节点共模"而不是盘级故障信号?
      (→ 这类列会污染邻域注意力的 Q/K 点积, 让 logit_spread 变小 = 注意力更平)

数据源: <PROCESSED_DIR>/{test,train}_shard_*.npz
        s [N,T,F]   n [N,T,M,F](新序) 或 [N,M,T,F](旧序, 自动识别)
        m [N,M]     l [N]
前提:   分片生成与 USE_NEIGHBORHOOD 无关(data_utils 总是保存邻居) → 现成分片即可分析

指标(逐列 c):
  auc_raw     窗口均值                 vs 标签 的 AUC
  auc_delta   (后 k 天 - 前 T-k 天)     的 AUC        <- 故障的本质是"变化"
  auc_slope   窗口内线性斜率            的 AUC
  auc_lastk   最后 3 天均值             的 AUC
  auc_vol     窗口内标准差              的 AUC
  auc_nb      同一列在"邻居"上的 delta AUC             <- 若 ≈ auc_delta → 只反映机架共模
  mwu_p       Mann-Whitney U 检验 p 值(基于 auc_delta)
  mi          与标签的互信息(10 分箱)
  zero_rate   零值占比
  col_std     逐列标准差(尺度健康度)
  gt2_rate    |z|>2 占比
  maxcorr     与其它列的最大 |Pearson|(delta 空间, 冗余度)
  maxcorr_with 最大相关列

用法:
  python analyze_feature_utility.py                       # 默认用测试分片
  python analyze_feature_utility.py --kind train --max-samples 20000
  python analyze_feature_utility.py --dir /path/to/processed_data --out feature_utility.json
  python analyze_feature_utility.py --max-shards 2 --last-k 7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

EPS = 1e-8
BLOCK = 10000            # 每次处理的样本数(邻居张量 [B,T,M,F] 内存友好)

# ============================================================
# SMART ID → 常见语义(SATA/SSD SMART 规范的通用映射, 仅用于分组与可读性)
# ============================================================
SMART_NAME = {
    5: 'Reallocated_Sector_Ct',
    9: 'Power_On_Hours',
    12: 'Power_Cycle_Count',
    170: 'Available_Reservd_Space/Grown_Bad_Blk',
    171: 'Program_Fail_Count',
    172: 'Erase_Fail_Count',
    173: 'Wear_Leveling_Count',
    174: 'Unexpect_Power_Loss_Ct',
    175: 'Program_Fail_Count_Chip',
    177: 'Wear_Leveling_Count(alt)',
    180: 'Unused_Rsvd_Blk_Cnt_Tot',
    181: 'Program_Fail_Cnt_Total',
    182: 'Erase_Fail_Count_Total',
    183: 'Runtime_Bad_Block',
    184: 'End-to-End_Error',
    187: 'Reported_Uncorrect',
    188: 'Command_Timeout',
    190: 'Airflow_Temperature_Cel',
    192: 'Power-off_Retract_Count',
    194: 'Temperature_Celsius',
    195: 'Hardware_ECC_Recovered',
    196: 'Reallocated_Event_Count',
    197: 'Current_Pending_Sector',
    198: 'Offline_Uncorrectable',
    199: 'UDMA_CRC_Error_Count',
    206: 'Flying_Height/Write_Error_Rate',
    232: 'Available_Reservd_Space',
    233: 'Media_Wearout_Indicator',
    241: 'Total_LBAs_Written',
    242: 'Total_LBAs_Read',
}

# 语义分组: 用于组级汇总 + 规划"分组消融"(leave-one-group-out)
SMART_GROUP = {
    '重映射/坏块':     [5, 170, 183, 196, 197, 198, 199],
    '程序/擦除错误':   [171, 172, 175, 181, 182, 184],
    '磨损/寿命':       [173, 177, 232, 233],
    '错误计数/超时':   [187, 188, 195],
    '温度':            [190, 194],
    '时间/负载/事件':  [9, 12, 174, 192, 241, 242],
    '预留空间':        [180],
    '其他':            [206],
}


# ============================================================
# 0. 参数 / 路径 / 列名
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description='逐 SMART 列的效用画像(无需训练)')
    ap.add_argument('--dir', default=None, help='分片目录(默认 data_utils.PROCESSED_DIR)')
    ap.add_argument('--kind', default='test', choices=['test', 'train'], help='用测试集还是训练集分片')
    ap.add_argument('--max-shards', type=int, default=0, help='最多用几个分片(0=全部)')
    ap.add_argument('--max-samples', type=int, default=0, help='每个分片最多采样多少样本(0=全部)')
    ap.add_argument('--last-k', type=int, default=7, help='delta 定义的"近端"天数(默认 7, 与 TEST_LEAD_TIME 对齐)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='feature_utility.json')
    return ap.parse_args()


def resolve_paths(args):
    """返回 (shard_dir, col_ids)

    优先复用 data_utils 的 PROCESSED_DIR / N_COLS, 保证与训练脚本同源;
    导入失败(缺 torch/pandas 等)时回退到 --dir + 占位列名, 便于离线试跑。
    """
    col_ids, shard_dir = None, args.dir
    try:
        import data_utils as du
        if shard_dir is None:
            shard_dir = du.PROCESSED_DIR
        col_ids = [int(c.split('_', 1)[1]) for c in du.N_COLS]
    except Exception as e:
        print('  [warn] 无法从 data_utils 读取列名/目录(%s)' % e)
        if shard_dir is None:
            raise SystemExit('请用 --dir 显式指定分片目录')
    if not os.path.isdir(shard_dir):
        raise SystemExit('分片目录不存在: %s' % shard_dir)
    return shard_dir, col_ids


def list_shards(d, kind):
    fs = sorted(glob.glob(os.path.join(d, kind + '_shard_*.npz')))
    if not fs:
        other = 'train' if kind == 'test' else 'test'
        print('  [warn] 目录下没有 %s_shard_*.npz, 改试 %s 分片' % (kind, other))
        fs = sorted(glob.glob(os.path.join(d, other + '_shard_*.npz')))
        kind = other
    return fs, kind


def load_shard(path, max_samples, rng):
    z = np.load(path)
    s = np.asarray(z['s'], dtype=np.float32)
    n = np.asarray(z['n'], dtype=np.float32)
    m = np.asarray(z['m']).astype(bool)
    l = np.asarray(z['l'], dtype=np.float32).reshape(-1)
    if max_samples and s.shape[0] > max_samples:
        idx = rng.choice(s.shape[0], size=max_samples, replace=False)
        s, n, m, l = s[idx], n[idx], m[idx], l[idx]
    return s, n, m, l


def infer_layout(s, n):
    """返回 'TMF'(新序 [N,T,M,F]) 或 'MTF'(旧序 [N,M,T,F])"""
    N, T, F = s.shape
    if n.ndim != 4 or n.shape[0] != N:
        raise ValueError('邻居张量形状异常: s=%s n=%s' % (s.shape, n.shape))
    if n.shape[1] == T and n.shape[3] == F:
        return 'TMF'
    if n.shape[2] == T and n.shape[3] == F:
        return 'MTF'
    raise ValueError('无法识别轴序: s=%s n=%s' % (s.shape, n.shape))


# ============================================================
# 1. 分块特征提取(时间维 → 每样本一个 F 维向量)
# ============================================================
def block_features(s, n, m, layout, last_k):
    """返回 (raw, delta, slope, lastk, vol, nb_delta, nb_has)

    说明: 这里直接用全部 T 步(不做缺失掩码), 因为 per-(model,col) Z-score 后
          "0" 既可能代表"该天无记录"也可能代表"该列本身就是 0", 两者不可区分。
          列级 zero_rate 会在报告中单独给出, 便于评估该局限。
    """
    B, T, F = s.shape
    x = np.nan_to_num(s).astype(np.float32)
    k = int(max(1, min(last_k, T - 1)))

    t = np.arange(T, dtype=np.float32)
    t = t - t.mean()
    denom = float((t * t).sum()) or 1.0
    slope = (x * t[None, :, None]).sum(1) / denom        # [B,F] 线性趋势
    raw = x.mean(1)                                      # [B,F]
    delta = x[:, -k:, :].mean(1) - x[:, :-k, :].mean(1)  # [B,F] 近端 - 远端
    lastk = x[:, -3:, :].mean(1) if T >= 3 else x.mean(1)
    vol = x.std(1)                                       # [B,F] 波动性

    nn = n
    if layout == 'MTF':
        nn = np.transpose(n, (0, 2, 1, 3))               # [B,M,T,F] → [B,T,M,F]
    nn = np.nan_to_num(nn).astype(np.float32)
    nb_d = nn[:, -k:, :, :].mean(1) - nn[:, :-k, :, :].mean(1)   # [B,M,F]
    w = m[:, :, None].astype(np.float32)                          # [B,M,1]
    nb_delta = (nb_d * w).sum(1) / np.maximum(w.sum(1), 1)        # [B,F]
    nb_has = m.sum(1).astype(np.float32)
    return raw, delta, slope, lastk, vol, nb_delta, nb_has


# ============================================================
# 2. 单列判别力工具
# ============================================================
def auc_p(y, v):
    """返回 (AUC, Mann-Whitney p)。AUC>0.5 表示数值越大越像故障。"""
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(v)
    if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
        return float('nan'), float('nan')
    a = float(roc_auc_score(y[ok], v[ok]))
    try:
        p = float(mannwhitneyu(v[ok & (y > 0.5)], v[ok & (y < 0.5)],
                               alternative='two-sided').pvalue)
    except Exception:
        p = float('nan')
    return a, p


def auc_by_shard(y, v, shard_id):
    """逐分片 AUC(看稳健性) → (mean, std, n_shards)"""
    from sklearn.metrics import roc_auc_score
    vals = []
    for sid in np.unique(shard_id):
        msk = shard_id == sid
        if msk.sum() < 50 or len(np.unique(y[msk])) < 2:
            continue
        vals.append(float(roc_auc_score(y[msk], v[msk])))
    if not vals:
        return float('nan'), float('nan'), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def mutual_info_binned(y, v, bins=10):
    from sklearn.metrics import mutual_info_score
    ok = np.isfinite(v)
    if ok.sum() < 100:
        return float('nan')
    q = np.unique(np.quantile(v[ok], np.linspace(0.0, 1.0, bins + 1)))
    if len(q) < 3:
        return 0.0
    b = np.digitize(v[ok], q[1:-1])
    return float(mutual_info_score(y[ok].astype(np.int64), b))


def pearson_matrix(X):
    """X [N,F] → |corr| 矩阵 [F,F](常量列置 0, 避免 NaN)"""
    Xc = X - X.mean(0, keepdims=True)
    sd = Xc.std(0)
    good = sd > 1e-9
    Z = np.zeros_like(Xc)
    Z[:, good] = Xc[:, good] / sd[good]
    C = (Z.T @ Z) / max(len(Xc) - 1, 1)
    C = np.abs(np.nan_to_num(C))
    np.fill_diagonal(C, 0.0)
    return C


# ============================================================
# 3. 分类 / 分组 / 冗余聚类
# ============================================================
def classify(m, corr_hi=0.95):
    """给单列打标签(可多标签)

    strong / medium / weak : 由 |auc_delta - 0.5| 判定该列的"变化量"判别力
    rack_noise             : 自身无判别力, 但"邻居"上却有 → 只反映机架共模, 疑似噪声
    shared_with_rack       : 自身有判别力, 但邻居同样强(>=70%) → 机架/节点共同漂移
    redundant              : 与其它列 |corr| >= 0.95 → 信息重复
    degenerate             : 常量列 / 99% 以上为 0
    """
    if (m['col_std'] < 1e-6) or (m['zero_rate'] > 0.99):
        return ['degenerate']
    ad, an = m['auc_delta'], m['auc_nb']
    s_ad = abs(ad - 0.5) if np.isfinite(ad) else 0.0
    s_an = abs(an - 0.5) if np.isfinite(an) else 0.0
    tags = ['strong' if s_ad >= 0.08 else ('medium' if s_ad >= 0.03 else 'weak')]
    if s_an > 0.05 and s_ad < 0.03:
        tags.append('rack_noise')
    elif s_ad >= 0.03 and s_an >= 0.7 * s_ad:
        tags.append('shared_with_rack')
    if m['maxcorr'] >= corr_hi:
        tags.append('redundant')
    return tags


def group_summary(rows, col_ids):
    """按 SMART_GROUP 汇总(只统计确实存在于 col_ids 的列)"""
    pos = {cid: i for i, cid in enumerate(col_ids)}
    out = {}
    for gname, ids in SMART_GROUP.items():
        sel = [pos[i] for i in ids if i in pos]
        if not sel:
            continue
        ad = np.array([rows[i]['auc_delta'] for i in sel], dtype=np.float64)
        strength = np.abs(ad - 0.5)
        out[gname] = {
            'n_cols': len(sel),
            'ids': [col_ids[i] for i in sel],
            'auc_delta_mean': float(np.nanmean(ad)),
            'strength_mean': float(np.nanmean(strength)),
            'strength_max': float(np.nanmax(strength)),
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['strength_max']))


def redundancy_clusters(C, col_ids, thr=0.90):
    """基于 |corr| 的并查集聚类: |corr(i,j)| >= thr 即同簇

    不用 scipy.hierarchy —— 阈值语义更直观(严格按 |corr| 连边), 也少一层依赖。
    """
    n = len(col_ids)
    if n < 2:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] >= thr:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    buckets = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(col_ids[i])
    out = [{'size': len(v), 'members': v} for _, v in sorted(buckets.items()) if len(v) > 1]
    out.sort(key=lambda d: (-d['size'], d['members']))
    return out


def fmt_name(cid, col_ids):
    return SMART_NAME.get(cid, 'col%d' % cid) if col_ids is not None else 'col%d' % cid


def _ids(rs):
    """提取一列行的 SMART ID 列表(用于打印)"""
    return [r['col'] for r in rs]


def print_table(rows, col_ids, top=None):
    print('  排名  ID    名称                                auc_raw  auc_del  slope   lastk    vol    auc_nb   |dAUC|  maxcorr  标签')
    print('  ' + '-' * 128)
    order = sorted(range(len(rows)), key=lambda i: -abs(rows[i]['auc_delta'] - 0.5))
    for rank, i in enumerate(order, 1):
        if top and rank > top:
            break
        m = rows[i]
        cid = col_ids[i] if col_ids is not None else -1
        tags = ','.join(m['tags'])
        print('  %3d  %-4s  %-34s  %7.3f  %7.3f  %6.3f  %6.3f  %6.3f  %7.3f  %6.3f  %7.3f  %s'
              % (rank, cid, fmt_name(cid, col_ids)[:34], m['auc_raw'], m['auc_delta'],
                 m['auc_slope'], m['auc_lastk'], m['auc_vol'], m['auc_nb'],
                 abs(m['auc_delta'] - 0.5), m['maxcorr'], tags))


# ============================================================
# 4. 主流程
# ============================================================
def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    d, col_ids = resolve_paths(args)
    shards, kind = list_shards(d, args.kind)
    if not shards:
        raise SystemExit('未找到分片文件: %s' % d)
    if args.max_shards:
        shards = shards[:args.max_shards]

    print('=' * 128)
    print('  逐 SMART 列效用画像   (kind=%s, 分片=%d, dir=%s)' % (kind, len(shards), d))
    print('=' * 128)

    acc = {k: [] for k in ('raw', 'delta', 'slope', 'lastk', 'vol', 'nb_delta')}
    elem = {'zero': [], 'gt2': [], 'sq': [], 'lin': []}
    ys, sids = [], []
    layout, T, Fd, M, n_elem = None, None, None, None, 0

    for si, path in enumerate(shards):
        s, n, m, l = load_shard(path, args.max_samples, rng)
        if layout is None:
            layout = infer_layout(s, n)
            T, Fd, M = s.shape[1], s.shape[2], m.shape[1]
        for b0 in range(0, s.shape[0], BLOCK):
            b1 = min(b0 + BLOCK, s.shape[0])
            out = block_features(s[b0:b1], n[b0:b1], m[b0:b1], layout, args.last_k)
            for k, v in zip(acc.keys(), out):
                acc[k].append(v)
            z = np.nan_to_num(s[b0:b1]).astype(np.float64)
            elem['zero'].append((z == 0).sum(axis=(0, 1)))
            elem['gt2'].append((np.abs(z) > 2.0).sum(axis=(0, 1)))
            # ★ float64 累积: 用 Σx²/n - (Σx/n)² 求方差存在"灾难性抵消"，
            #   float32 会误把低方差列判成常量列（实测比两遍法多出 1 个假常量列）
            elem['sq'].append((z ** 2).sum(axis=(0, 1)))
            elem['lin'].append(z.sum(axis=(0, 1)))
            n_elem += z.shape[0] * z.shape[1]
        ys.append(l)
        sids.append(np.full(l.shape[0], si, dtype=np.int32))
        print('  [%2d/%2d] %-22s N=%6d  pos=%5d  layout=%s'
              % (si + 1, len(shards), os.path.basename(path), l.shape[0],
                 int((l > 0.5).sum()), layout))

    for k in acc:
        acc[k] = np.concatenate(acc[k], axis=0)
    for k in elem:
        elem[k] = np.sum(elem[k], axis=0)      # ★ 每块返回 [F]，是可加性累计量 → 按列求和
    y = np.concatenate(ys, axis=0)
    sid = np.concatenate(sids, axis=0)
    n_pos = int((y > 0.5).sum())
    print('  → 合计样本 %d | 正样本 %d (%.2f%%) | T=%d F=%d M=%d | 正样本每列 0 值占比见下表'
          % (len(y), n_pos, 100.0 * n_pos / max(len(y), 1), T, Fd, M))

    # ---------- 逐列指标 ----------
    C = pearson_matrix(acc['delta'])
    lin_m = elem['lin'] / n_elem
    col_std = np.sqrt(np.maximum(elem['sq'] / n_elem - lin_m ** 2, 0.0))
    rows = []
    for i in range(Fd):
        a_del, p_del = auc_p(y, acc['delta'][:, i])
        a_nb, _ = auc_p(y, acc['nb_delta'][:, i])
        sm, ss, ns = auc_by_shard(y, acc['delta'][:, i], sid)
        ci = int(np.argmax(C[i]))
        colv = acc['delta'][:, i]
        m = {
            'col': int(col_ids[i]) if col_ids else i,
            'name': fmt_name(col_ids[i] if col_ids else i, col_ids),
            'auc_raw': auc_p(y, acc['raw'][:, i])[0],
            'auc_delta': a_del, 'p_delta': p_del,
            'auc_slope': auc_p(y, acc['slope'][:, i])[0],
            'auc_lastk': auc_p(y, acc['lastk'][:, i])[0],
            'auc_vol': auc_p(y, acc['vol'][:, i])[0],
            'auc_nb': a_nb,
            'auc_delta_shard_mean': sm, 'auc_delta_shard_std': ss, 'n_shards_used': ns,
            'delta_mean_fail': float(np.nanmean(colv[y > 0.5])) if n_pos else float('nan'),
            'delta_mean_ok': float(np.nanmean(colv[y < 0.5])) if (len(y) - n_pos) else float('nan'),
            'mi_delta': mutual_info_binned(y, colv),
            'zero_rate': float(elem['zero'][i] / max(n_elem, 1)),
            'gt2_rate': float(elem['gt2'][i] / max(n_elem, 1)),
            'col_std': float(col_std[i]),
            'maxcorr': float(C[i, ci]),
            'maxcorr_with': int(col_ids[ci]) if col_ids else ci,
        }
        m['direction'] = ('higher_in_fail' if (m['delta_mean_fail'] - m['delta_mean_ok']) > 0
                          else 'lower_in_fail')
        m['tags'] = classify(m)
        rows.append(m)

    gs = group_summary(rows, col_ids if col_ids else list(range(Fd)))
    clusters = redundancy_clusters(C, col_ids if col_ids else list(range(Fd)))
    return report(args, d, kind, layout, T, Fd, M, len(y), n_pos, rows, gs, clusters,
                  col_ids, col_std, elem, n_elem, sid, y, acc)


# ============================================================
# 5. 报告 / 落盘
# ============================================================
def report(args, d, kind, layout, T, Fd, M, N, n_pos, rows, gs, clusters,
           col_ids, col_std, elem, n_elem, sid, y, acc):
    zr = np.array([r['zero_rate'] for r in rows])
    # ★ 全局 E[x²]: Σ_c Σx² ÷ (每列元素数 × 列数)。elem['sq'] 是每列的平方和，
    #   n_elem 是"每列"的元素数 → 必须再除以 Fd，否则结果会大 F 倍（实测差 30 倍）。
    ex2 = float(elem['sq'].sum() / max(n_elem * Fd, 1))
    strength = np.array([abs(r['auc_delta'] - 0.5) for r in rows])
    std_shard = np.array([r['auc_delta_shard_std'] for r in rows])

    # ---------- 输入尺度自检 ----------
    print('\n' + '=' * 128)
    print('  输入尺度自检（决定注意力是否"有话说"）')
    print('=' * 128)
    print('  E[x^2] = %.4f    [per-(model,col) Z-score 若尺度健康应 ≈ 1.0]' % ex2)
    nzc = col_std[col_std > 1e-6]
    print('  逐列 std 中位 %.3f | 范围 [%.4f, %.4f]（含常量列）'
          % (float(np.median(col_std)), float(col_std.min()), float(col_std.max())))
    print('  非零列 std 范围 [%.4f, %.4f] | std_max/std_min = %.1fx  （排除 %d 个常量列）'
          % (float(nzc.min()) if nzc.size else float('nan'),
             float(nzc.max()) if nzc.size else float('nan'),
             float(nzc.max() / nzc.min()) if nzc.size else float('nan'),
             int((col_std <= 1e-6).sum())))
    print('  逐列 0 值占比 中位 %.1f%% | 最大 %.1f%%   (含"该天无记录"与"该列本就是 0"，不可区分)'
          % (100 * float(np.median(zr)), 100 * float(zr.max())))
    print('  逐列 |z|>2 占比 中位 %.2f%%' % (100 * float(np.median([r['gt2_rate'] for r in rows]))))
    if ex2 < 0.6:
        print('  [!] E[x^2]=%.3f 远小于 1 → 输入尺度整体偏小 → Q/K 点积小 → logit_spread 小 →'
              ' softmax 近乎均匀(等权平均) → 邻域分支退化为"平均池化"，会稀释异常。' % ex2)
        print('      这与 attention_saturation.json 的 ‖q‖≈0.87 / logit_spread≈0.036 互相印证。')
        print('      对策优先级: ①输入 LayerNorm(最省, 实测 spread ×9) ②稳健标准化(median/MAD) 之后再谈 1/√F')
    if float(zr.max()) > 0.5:
        print('  [!] 存在 0 值占比 > 50% 的列 → 这些列的"变化量"信号极易被 0 淹没，建议列入观察名单。')

    # ---------- 逐列表 ----------
    print('\n' + '=' * 128)
    print('  逐列效用表（按 |auc_delta-0.5| 排序；分片 AUC std 中位 %.3f 可作为"差异是否可信"的参照）'
          % float(np.nanmedian(std_shard)))
    print('=' * 128)
    print_table(rows, col_ids)
    print('  列说明: auc_del=变化量判别力(近%d天-远端)  slope=斜率  lastk=最后3天  vol=波动性  '
          'auc_nb=同一列在"邻居"上的变化量判别力' % args.last_k)

    # ---------- 分类摘要 ----------
    print('\n' + '=' * 128)
    print('  分类摘要')
    print('=' * 128)

    strong = [r for r in rows if 'strong' in r['tags']]
    weak = [r for r in rows if 'weak' in r['tags']]
    degen = [r for r in rows if 'degenerate' in r['tags']]
    rnoise = [r for r in rows if 'rack_noise' in r['tags']]
    shared = [r for r in rows if 'shared_with_rack' in r['tags']]
    redun = [r for r in rows if 'redundant' in r['tags']]
    print('  strong  (|dAUC|>=0.08)      : %2d 列  %s' % (len(strong), _ids(strong)))
    print('  weak    (|dAUC|<0.03)       : %2d 列  %s' % (len(weak), _ids(weak)))
    print('  degenerate(常量/近常量)     : %2d 列  %s' % (len(degen), _ids(degen)))
    print('  rack_noise(自身无,邻居有)   : %2d 列  %s' % (len(rnoise), _ids(rnoise)))
    print('  shared_with_rack(邻居同样强): %2d 列  %s' % (len(shared), _ids(shared)))
    print('  redundant(|corr|>=0.95)     : %2d 列  %s' % (len(redun), _ids(redun)))

    # ---------- 语义分组 ----------
    print('\n' + '=' * 128)
    print('  语义分组汇总（用于规划"分组消融"：leave-one-group-out）')
    print('=' * 128)
    for gname, g in gs.items():
        print('  %-16s %2d 列 | auc_delta 均值 %.3f | 判别强度 mean %.3f max %.3f | ids=%s'
              % (gname, g['n_cols'], g['auc_delta_mean'], g['strength_mean'],
                 g['strength_max'], g['ids']))

    # ---------- 冗余簇 ----------
    print('\n' + '=' * 128)
    print('  高冗余簇（|corr|>=0.90，簇内可只留 1 列或用 PCA）')
    print('=' * 128)
    if not clusters:
        print('  （无 >=2 列的高冗余簇）')
    for cl in clusters:
        best = max(cl['members'],
                   key=lambda cid: strength[col_ids.index(cid)] if col_ids else 0.0)
        print('  size=%d  成员=%s   建议保留: %d (%s)'
              % (cl['size'], cl['members'], best, fmt_name(best, col_ids)))
    return save(args, d, kind, layout, T, Fd, M, N, n_pos, rows, gs, clusters,
                col_ids, col_std, ex2, zr, strength, std_shard, acc, y)


def save(args, d, kind, layout, T, Fd, M, N, n_pos, rows, gs, clusters,
         col_ids, col_std, ex2, zr, strength, std_shard, acc, y):
    order = list(np.argsort(-strength))
    print('\n' + '=' * 128)
    print('  结论')
    print('=' * 128)
    print('  ① 最有效列 top-8（按"变化量"判别力排序）:')
    for i in order[:8]:
        r = rows[i]
        print('     %-4d %-34s auc_delta=%.3f (%s) auc_raw=%.3f slope=%.3f lastk=%.3f vol=%.3f nb=%.3f mi=%.4f'
              % (r['col'], r['name'], r['auc_delta'], r['direction'], r['auc_raw'],
                 r['auc_slope'], r['auc_lastk'], r['auc_vol'], r['auc_nb'], r['mi_delta']))

    gain = sorted(((r['col'], r['name'],
                    abs(r['auc_delta'] - 0.5) - abs(r['auc_raw'] - 0.5)) for r in rows),
                  key=lambda t: -t[2])
    print('\n  ② "变化量比原值更有用"的列 top-5（说明模型必须看到时间趋势，不能只看窗口均值）:')
    for cid, name, g in gain[:5]:
        print('     %-4d %-34s 判别力增量 +%.3f' % (cid, name, g))

    weak_cols = _ids([r for r in rows if 'weak' in r['tags']])
    degen = _ids([r for r in rows if 'degenerate' in r['tags']])
    print('\n  ③ 低贡献/退化列（不建议直接删，见 ④）:')
    print('     weak(|dAUC|<0.03): %s' % weak_cols)
    print('     degenerate       : %s' % degen)

    tgrp = gs.get('时间/负载/事件')
    if tgrp:
        print('\n  ④ 时间/负载/事件类列（%s）定位:' % tgrp['ids'])
        print('     auc_delta 均值 %.3f | 判别强度 mean %.3f max %.3f'
              % (tgrp['auc_delta_mean'], tgrp['strength_mean'], tgrp['strength_max']))
        print('     → 单独判别力弱是正常的（健康盘与故障盘的通电时长分布高度重叠）。它们的价值是"当分母":')
        print('        5/9(重映射速率)  241/9(写入速率)  232/9(寿命消耗速率)  194 与 241 的联合解读')
        print('     → 处理建议: 不要删, 改为"派生比率通道"参与（原值通道可保留或降权）。')

    shared = _ids([r for r in rows if 'shared_with_rack' in r['tags']])
    rnoise = _ids([r for r in rows if 'rack_noise' in r['tags']])
    if shared or rnoise:
        print('\n  ⑤ 疑似"机架共模"列（会污染邻域注意力的 Q/K 点积 → logit_spread 变小 → 更平）:')
        print('     rack_noise(自身无判别力、邻居反而有)      : %s' % rnoise)
        print('     shared_with_rack(自身有、邻居同样强>=70%%) : %s' % shared)
        print('     → 与 neighborhood_analysis 的 cos≈-0.94 / shrink≈0.45 是同一现象的两个侧面。')
        print('     → 对策: 门控 r=self+g*c，或后置差分 concat[s_self, s_nb, s_self-s_nb]，'
              '不要让模型从"均匀平均"里自己学。')

    print('\n  提示: 单列 AUC 高 ≠ 模型里一定重要（列间可互相替代）；反之单列弱也可能因组合而有用。')
    print('        本报告用于【筛掉确定无用的列 / 找出机架共模列 / 规划派生特征】，'
          '最终裁决请用分组消融或置换重要性。')

    nzc = col_std[col_std > 1e-6]
    payload = {
        'shard_dir': d, 'kind': kind, 'layout': layout,
        'n_samples': int(N), 'n_pos': int(n_pos), 'pos_ratio': float(n_pos / max(N, 1)),
        'seq_len': int(T), 'feat_dim': int(Fd), 'max_neighbors': int(M),
        'last_k': int(args.last_k),
        'input_scale': {
            'E_x2': ex2,
            'col_std_median': float(np.median(col_std)),
            'col_std_min': float(col_std.min()), 'col_std_max': float(col_std.max()),
            'col_std_list': [float(v) for v in col_std],
            'col_std_min_nonzero': float(nzc.min()) if nzc.size else float('nan'),
            'n_constant_cols': int((col_std <= 1e-6).sum()),
            'zero_rate_median': float(np.median(zr)), 'zero_rate_max': float(zr.max()),
        },
        'auc_delta_shard_std_median': float(np.nanmedian(std_shard)),
        'cols': rows,
        'groups': gs,
        'redundancy_clusters': clusters,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('\n  -> 结果已保存: %s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
