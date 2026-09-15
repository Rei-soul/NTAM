#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_neighborhood_value.py —— 用【分片数据】回答两个问题（无需重训）

  Q1  邻域开对模型有害吗?  → 稀释效应检验（论文式 r = self + c 是否把异常拉回正常）
  Q2  到底需不需要邻域?    → 增量判别力 ΔAUC：仅自身 vs +真邻居 vs +随机盘（关键对照）

数据源: <PROCESSED_DIR>/{train,test}_shard_*.npz
        s [N,T,F]   n [N,T,M,F](新序) 或 [N,M,T,F](旧序, 自动识别)
        m [N,M]     l [N,1]
前提:  分片生成与 USE_NEIGHBORHOOD 无关（data_utils 总是保存邻居）→ 现成分片即可分析

用法:
  python analyze_neighborhood_value.py
  python analyze_neighborhood_value.py --dir /mnt/newdisk/qhmiao/disk_failure_prediction/processed_data
  python analyze_neighborhood_value.py --max-shards 2 --max-samples 20000   # 快速试跑
  python analyze_neighborhood_value.py --no-fake
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

EPS = 1e-8
BLOCK = 20000

SELF_KEYS = ['self_absmean', 'self_absmax', 'self_gt2', 'self_seq_absmean',
             'self_seq_std', 'self_slope_absmean', 'self_last_absmean', 'self_t_valid']
NB_KEYS = ['nb_absmean', 'nb_absmax', 'nb_gt2', 'nb_minus_self_absmean',
           'nb_anom_cos', 'nb_valid', 'nb_has']


# ============================================================
# 0. 参数
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description='用分片数据评估邻域信息的价值')
    ap.add_argument('--dir', default=None, help='分片目录（默认 data_utils.PROCESSED_DIR）')
    ap.add_argument('--max-shards', type=int, default=0, help='train/test 各最多用几个分片（0=全部）')
    ap.add_argument('--max-samples', type=int, default=0, help='每个分片最多采样多少样本（0=全部）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-fake', action='store_true', help='跳过“随机盘”对照组')
    ap.add_argument('--out', default='neighborhood_analysis.json')
    return ap.parse_args()


# ============================================================
# 1. 分片 IO / 轴序识别
# ============================================================
def list_shards(d, kind):
    return sorted(glob.glob(os.path.join(d, kind + '_shard_*.npz')))


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
# 2. 向量聚合（T×M → 每样本一个 F 向量），全 0/NaN 视为缺失
# ============================================================
def self_vector(s):
    ok = ~np.all(s == 0, axis=-1)
    ok &= ~np.isnan(s).any(axis=-1)
    cnt = ok.sum(1)
    vec = np.where(ok[..., None], np.nan_to_num(s), 0.0).sum(1) / np.maximum(cnt, 1)[:, None]
    return vec.astype(np.float32), cnt.astype(np.float32)


def neighbor_vectors(n, m, layout):
    if layout == 'MTF':
        n = np.transpose(n, (0, 2, 1, 3))                  # [N,M,T,F] → [N,T,M,F]
    ok_t = ~np.all(n == 0, axis=-1)
    ok_t &= ~np.isnan(n).any(axis=-1)
    w = (m[:, None, :] & ok_t).astype(np.float32)          # [N,T,M]
    num = (np.nan_to_num(n) * w[..., None]).sum((1, 2))
    den = w.sum((1, 2))
    nb = num / np.maximum(den, 1)[:, None]
    return nb.astype(np.float32), m.sum(1).astype(np.float32)


# ============================================================
# 3. 特征构造（全部在逐列标准化 z 空间）
# ============================================================
def _cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return ((a * b).sum(1) / (na * nb + EPS)).astype(np.float32)


def _proj(d, delta):
    return ((d * delta).sum(1) / ((d * d).sum(1) + EPS)).astype(np.float32)


def build_feature_blocks(s, n, m, mu, sd, center, layout, rng, use_fake):
    """返回 (X_self[N,8], X_nb[N,7], cos[N], proj[N], shrink[N], mag[N], n_valid[N])

    use_fake=True 时用“随机盘的自身向量”充当邻居（关键对照组）
    shrink = 1 - ||z_nb - center|| / ||z_self - center|| ：融合后“异常度”的收缩率
             >0 表示邻居把目标往正常方向拉（稀释），<0 表示放大
    mag    = ||z_self - center|| ：目标自身偏离正常的幅度（用于分层，小幅度样本无异常可稀释）
    """
    N, T, F = s.shape
    Xs = np.zeros((N, len(SELF_KEYS)), dtype=np.float32)
    Xn = np.zeros((N, len(NB_KEYS)), dtype=np.float32)
    cos_all = np.zeros(N, dtype=np.float32)
    proj_all = np.zeros(N, dtype=np.float32)
    shrink_all = np.zeros(N, dtype=np.float32)
    mag_all = np.zeros(N, dtype=np.float32)
    nv_all = np.zeros(N, dtype=np.float32)

    t = np.arange(T, dtype=np.float32)
    t = t - t.mean()
    denom = float((t * t).sum()) or 1.0

    for b0 in range(0, N, BLOCK):
        b1 = min(b0 + BLOCK, N)
        sb, nb_b, mb = s[b0:b1], n[b0:b1], m[b0:b1]

        z_s = (np.nan_to_num(sb) - mu) / sd
        sv, t_valid = self_vector(sb)
        z_self = (sv - mu) / sd
        slope = (z_s * t[None, :, None]).sum(1) / denom

        if use_fake:
            perm = rng.permutation(b1 - b0)
            z_nb = z_self[perm]
            nvalid = mb.sum(1).astype(np.float32)
        else:
            nbv, nvalid = neighbor_vectors(nb_b, mb, layout)
            z_nb = (nbv - mu) / sd

        d = z_self - center
        delta = z_nb - z_self
        mag = np.linalg.norm(d, axis=1)
        mag_n = np.linalg.norm(z_nb - center, axis=1)
        cos_all[b0:b1] = _cos(d, delta)
        proj_all[b0:b1] = _proj(d, delta)
        shrink_all[b0:b1] = 1.0 - mag_n / (mag + EPS)
        mag_all[b0:b1] = mag
        nv_all[b0:b1] = nvalid

        Xn[b0:b1] = np.stack([
            np.abs(z_nb).mean(1),
            np.abs(z_nb).max(1),
            (np.abs(z_nb) > 2).mean(1),
            np.abs(z_nb - z_self).mean(1),
            _cos(d, z_nb - center),
            nvalid,
            (nvalid > 0).astype(np.float32),
        ], axis=1)

        Xs[b0:b1] = np.stack([
            np.abs(z_self).mean(1),
            np.abs(z_self).max(1),
            (np.abs(z_self) > 2).mean(1),
            np.abs(z_s).mean((1, 2)),
            z_s.std(1).mean(1),
            np.abs(slope).mean(1),
            np.abs(z_s[:, -1, :]).mean(1),
            t_valid,
        ], axis=1)

    return Xs, Xn, cos_all, proj_all, shrink_all, mag_all, nv_all


# ============================================================
# 4. 评估 / 统计
# ============================================================
def evaluate(Xtr, ytr, Xte, yte, shard_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight='balanced'))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    per = {}
    for sid in np.unique(shard_te):
        msk = shard_te == sid
        if len(np.unique(yte[msk])) < 2:
            continue
        per[int(sid)] = float(roc_auc_score(yte[msk], p[msk]))
    vals = list(per.values())
    return {
        'auc': float(roc_auc_score(yte, p)),
        'pr_auc': float(average_precision_score(yte, p)),
        'per_shard_auc': per,
        'auc_shard_mean': float(np.mean(vals)) if vals else float('nan'),
        'auc_shard_std': float(np.std(vals)) if vals else float('nan'),
    }


def dilution_stats(cos, proj, shrink, mag, y, nvalid):
    """稀释效应统计。

    shrink = 1 - ||z_nb - center|| / ||z_self - center||
      > 0 → 融合后异常度被收缩（邻居把目标往正常方向拉 = 稀释）
      < 0 → 异常被放大
    由于 mag 很小的样本本来就“没有异常可稀释”，额外给出 highmag（mag 大于中位数）子集统计。
    """
    from scipy.stats import mannwhitneyu
    has = nvalid > 0
    out = {'note': 'shrink>0 表示邻居把目标异常度拉向正常(稀释); cos<0 表示方向与异常方向相反'}
    if not bool(has.any()):
        return out
    big = has & (mag > float(np.median(mag[has])))
    groups = (('fail', (y > 0.5) & has), ('healthy', (y < 0.5) & has),
              ('fail_highmag', (y > 0.5) & big), ('healthy_highmag', (y < 0.5) & big))
    for grp, msk in groups:
        if int(msk.sum()) < 10:
            continue
        out[grp] = {
            'n': int(msk.sum()),
            'mag_median': float(np.median(mag[msk])),
            'cos_median': float(np.median(cos[msk])),
            'cos_neg_ratio': float((cos[msk] < 0).mean()),
            'proj_median': float(np.median(proj[msk])),
            'shrink_median': float(np.median(shrink[msk])),
            'shrink_mean': float(shrink[msk].mean()),
            'shrink_pos_ratio': float((shrink[msk] > 0).mean()),
        }
    for a, b, tag in (('fail', 'healthy', 'all'), ('fail_highmag', 'healthy_highmag', 'highmag')):
        if a in out and b in out:
            msk_a = (y > 0.5) & (big if tag == 'highmag' else has)
            msk_b = (y < 0.5) & (big if tag == 'highmag' else has)
            try:
                _, p_cos = mannwhitneyu(cos[msk_a], cos[msk_b], alternative='two-sided')
                _, p_shr = mannwhitneyu(shrink[msk_a], shrink[msk_b], alternative='two-sided')
            except Exception:
                p_cos = p_shr = None
            out['gap_cos_neg_ratio_' + tag] = out[a]['cos_neg_ratio'] - out[b]['cos_neg_ratio']
            out['gap_shrink_median_' + tag] = out[a]['shrink_median'] - out[b]['shrink_median']
            out['mwu_p_shrink_' + tag] = None if p_shr is None else float(p_shr)
            out['mwu_p_cos_' + tag] = None if p_cos is None else float(p_cos)
    return out


# ============================================================
# 5. 主流程
# ============================================================
def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    d = args.dir
    if d is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import data_utils
            d = data_utils.PROCESSED_DIR
        except Exception as e:
            print('[错误] 无法导入 data_utils 获取 PROCESSED_DIR:', e)
            return 1

    print('=' * 78)
    print('  邻域价值分析（基于分片数据）')
    print('=' * 78)
    print('  分片目录:', d)

    tr = list_shards(d, 'train')
    te = list_shards(d, 'test')
    if args.max_shards:
        tr, te = tr[:args.max_shards], te[:args.max_shards]
    if not tr:
        print('[错误] 未找到 train_shard_*.npz')
        return 1
    if not te:
        print('  [警告] 未找到 test 分片，改用最后一个 train 分片作评估集（与拟合集不重叠）')
        te, tr = tr[-1:], tr[:-1]
        if not tr:
            print('[错误] 训练分片不足，无法划分')
            return 1
    print('  训练分片 %d 个 | 评估分片 %d 个' % (len(tr), len(te)))

    # ---------- 体检 ----------
    s0, n0, m0, l0 = load_shard(tr[0], args.max_samples, rng)
    layout = infer_layout(s0, n0)
    N0, T, F = s0.shape
    M = n0.shape[2] if layout == 'TMF' else n0.shape[1]
    sv0, tv0 = self_vector(s0)
    nvc0 = m0.sum(1)
    print('\n[体检]  <- 这一步同时告诉你“实际训练用的是几天窗口”')
    print('  s=%s  n=%s  m=%s  l=%s' % (s0.shape, n0.shape, m0.shape, l0.shape))
    print('  -> 实际 SEQ_LEN = %d 天（以分片为准，不要信 config.py）' % T)
    print('  -> 邻居轴序 = %s  %s' % (layout, '[N,T,M,F] 新序' if layout == 'TMF' else '[N,M,T,F] 旧序'))
    print('  -> 特征维 F=%d, 最大邻居 M=%d' % (F, M))
    print('  -> 目标样本缺测(全0)占比 %.3f%%' % (100 * float((tv0 == 0).mean())))
    print('  -> 有效邻居数 mean %.2f / median %.0f / 有邻居占比 %.1f%%'
          % (nvc0.mean(), np.median(nvc0), 100 * float((nvc0 > 0).mean())))
    print('  -> 标签正例占比 %.2f%% (%d/%d)' % (100 * float(l0.mean()), int(l0.sum()), len(l0)))

    # ---------- Pass 1: 标准化参数（仅训练分片） ----------
    print('\n[Pass 1] 估计逐列 mu/sigma 与 z 空间中心（仅用训练分片）...')
    S = np.zeros(F, dtype=np.float64)
    S2 = np.zeros(F, dtype=np.float64)
    C = 0
    for p in tr:
        s = np.asarray(np.load(p)['s'], dtype=np.float32)
        sv, tv = self_vector(s)
        ok = tv > 0
        if int(ok.sum()) == 0:
            continue
        S += sv[ok].sum(0)
        S2 += (sv[ok].astype(np.float64) ** 2).sum(0)
        C += int(ok.sum())
    mu = (S / max(C, 1)).astype(np.float32)
    var = np.maximum(S2 / max(C, 1) - mu.astype(np.float64) ** 2, 0.0)
    sd = np.sqrt(var).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    Zsum = np.zeros(F, dtype=np.float64)
    Zc = 0
    for p in tr:
        s = np.asarray(np.load(p)['s'], dtype=np.float32)
        sv, tv = self_vector(s)
        ok = tv > 0
        if int(ok.sum()) == 0:
            continue
        Zsum += ((sv[ok] - mu) / sd).sum(0)
        Zc += int(ok.sum())
    center = (Zsum / max(Zc, 1)).astype(np.float32)

    ratio = float(sd.max() / max(float(sd.min()), 1e-12))
    print('  -> mu 范围 [%.4g, %.4g]   sigma 范围 [%.4g, %.4g]   sigma_max/sigma_min = %.1f x'
          % (mu.min(), mu.max(), sd.min(), sd.max(), ratio))
    print('  -> 提示: 该比值越大，说明各列尺度差异越大，越有必要做 Z-score 归一化')

    # ---------- 特征收集 ----------
    def collect(shard_paths, use_fake):
        Xs, Xn, Xf, ys = [], [], [], []
        cos_l, proj_l, shr_l, mag_l, nv_l, sid_l = [], [], [], [], [], []
        for sid, p in enumerate(shard_paths):
            s, n, m, l = load_shard(p, args.max_samples, rng)
            if infer_layout(s, n) != layout:
                print('  [警告] 分片 %s 轴序不一致，已跳过' % os.path.basename(p))
                continue
            a_xs, a_xn, a_cos, a_proj, a_shr, a_mag, a_nv = build_feature_blocks(
                s, n, m, mu, sd, center, layout, rng, use_fake=False)
            Xs.append(a_xs)
            Xn.append(a_xn)
            cos_l.append(a_cos)
            proj_l.append(a_proj)
            shr_l.append(a_shr)
            mag_l.append(a_mag)
            nv_l.append(a_nv)
            ys.append((l > 0.5).astype(np.int8))
            sid_l.append(np.full(len(l), sid, dtype=np.int16))
            if use_fake:
                _, f_xn, _, _, _, _, _ = build_feature_blocks(
                    s, n, m, mu, sd, center, layout, rng, use_fake=True)
                Xf.append(f_xn)
        if not Xs:
            raise RuntimeError('没有任何可用分片（轴序全部不一致？）')
        return (np.concatenate(Xs), np.concatenate(Xn),
                np.concatenate(Xf) if Xf else None,
                np.concatenate(ys), np.concatenate(cos_l), np.concatenate(proj_l),
                np.concatenate(shr_l), np.concatenate(mag_l),
                np.concatenate(nv_l), np.concatenate(sid_l))

    print('\n[Pass 2] 构造训练特征 ...')
    Xs_tr, Xn_tr, Xf_tr, y_tr, cos_tr, proj_tr, shr_tr, mag_tr, nv_tr, _ = collect(tr, not args.no_fake)
    print('  训练样本 %d（正 %d）' % (len(y_tr), int(y_tr.sum())))
    print('[Pass 3] 构造评估特征 ...')
    Xs_te, Xn_te, Xf_te, y_te, cos_te, proj_te, shr_te, mag_te, nv_te, sid_te = collect(te, not args.no_fake)
    print('  评估样本 %d（正 %d）' % (len(y_te), int(y_te.sum())))


    # ---------- 分析 A: 增量判别力 ----------
    print('\n[分析 A] 增量判别力 Delta-AUC')
    res = {}
    res['A'] = evaluate(Xs_tr, y_tr, Xs_te, y_te, sid_te)
    res['B'] = evaluate(np.hstack([Xs_tr, Xn_tr]), y_tr,
                        np.hstack([Xs_te, Xn_te]), y_te, sid_te)
    print('  A 仅自身            : AUC %.4f  PR-AUC %.4f  (分片 %.4f±%.4f)'
          % (res['A']['auc'], res['A']['pr_auc'],
             res['A']['auc_shard_mean'], res['A']['auc_shard_std']))
    print('  B 自身+真邻居       : AUC %.4f  PR-AUC %.4f  (分片 %.4f±%.4f)'
          % (res['B']['auc'], res['B']['pr_auc'],
             res['B']['auc_shard_mean'], res['B']['auc_shard_std']))
    if Xf_tr is not None:
        res['C'] = evaluate(np.hstack([Xs_tr, Xf_tr]), y_tr,
                            np.hstack([Xs_te, Xf_te]), y_te, sid_te)
        print('  C 自身+随机盘(对照) : AUC %.4f  PR-AUC %.4f  (分片 %.4f±%.4f)'
              % (res['C']['auc'], res['C']['pr_auc'],
                 res['C']['auc_shard_mean'], res['C']['auc_shard_std']))
    res['dAUC_B_minus_A'] = res['B']['auc'] - res['A']['auc']
    if 'C' in res:
        res['dAUC_B_minus_C'] = res['B']['auc'] - res['C']['auc']

    # ---------- 分析 B: 稀释效应 ----------
    print('\n[分析 B] 稀释效应（r = self + c）')
    dil_tr = dilution_stats(cos_tr, proj_tr, shr_tr, mag_tr, y_tr, nv_tr)
    dil_te = dilution_stats(cos_te, proj_te, shr_te, mag_te, y_te, nv_te)
    for tag, dd in (('训练集', dil_tr), ('评估集', dil_te)):
        if 'fail' not in dd:
            continue
        f = dd['fail']
        h = dd.get('healthy')
        print('  [%s] 故障样本: 异常幅度 mag 中位数 %.2f | 收缩率 shrink 中位数 %+.3f'
              '（被稀释占比 %.1f%%, n=%d）'
              % (tag, f['mag_median'], f['shrink_median'], 100 * f['shrink_pos_ratio'], f['n']))
        if h:
            print('         健康样本: mag 中位数 %.2f | shrink 中位数 %+.3f'
                  '（被稀释占比 %.1f%%, n=%d）'
                  % (h['mag_median'], h['shrink_median'], 100 * h['shrink_pos_ratio'], h['n']))
            print('         gap_shrink(全部样本) = %+.3f,  MWU p = %.3g'
                  % (dd['gap_shrink_median_all'], dd['mwu_p_shrink_all']))
        fh = dd.get('fail_highmag')
        hh = dd.get('healthy_highmag')
        if fh and hh:
            print('         [mag 高于中位数的子集] 故障 shrink %+.3f vs 健康 %+.3f'
                  ' -> gap %+.3f  (MWU p = %.3g)'
                  % (fh['shrink_median'], hh['shrink_median'],
                     dd['gap_shrink_median_highmag'], dd['mwu_p_shrink_highmag']))

    # ---------- 结论 ----------
    print('\n' + '=' * 78)
    print('  结论')
    print('=' * 78)
    dA = res['dAUC_B_minus_A']
    dC = res.get('dAUC_B_minus_C')
    if dA <= 0.002:
        print('  [x] 邻域没有增量判别力: Delta-AUC(B-A) = %+.4f <= 0.002' % dA)
    elif dC is not None and dC <= 0.002:
        print('  [x] 邻域无真实价值: Delta-AUC(B-A)=%+.4f 但与随机盘对照(B-C)=%+.4f 无差别'
              ' -> 增益来自噪声' % (dA, dC))
    else:
        dc_txt = ('%+.4f' % dC) if dC is not None else '未计算(--no-fake)'
        print('  [v] 邻域有真实增量价值: Delta-AUC(B-A)=%+.4f, Delta-AUC(B-C)=%s'
              % (dA, dc_txt))
    if 'fail_highmag' in dil_te and 'healthy_highmag' in dil_te:
        g = dil_te['gap_shrink_median_highmag']
        pv = dil_te['mwu_p_shrink_highmag']
        pv = 1.0 if pv is None else pv
        if g > 0.10 and pv < 0.05:
            print('  [!] 存在稀释效应: 高异常样本的 shrink 中位数 %+.3f vs 健康样本 %+.3f'
                  % (dil_te['fail_highmag']['shrink_median'],
                     dil_te['healthy_highmag']['shrink_median']))
            print('      （gap %+.3f, MWU p=%.3g）→ 邻居把目标的异常度拉向正常，'
                  '会稀释故障信号' % (g, pv))
            print('      建议: 改后置融合（先各自时序编码再融合）或加自适应门控（models_improved.py）')
        else:
            print('  [.] 未发现显著稀释效应 -> 若性能仍下降，原因更可能在输入尺度/注意力饱和或优化干扰')
    elif 'fail' in dil_te:
        print('  [.] 稀释效应样本不足，无法判定')
    print('  提示: 最终裁决请以“同配置仅切 USE_NEIGHBORHOOD”的消融为准（分片无需重建）')

    # ---------- 落盘 ----------
    payload = {
        'shard_dir': d,
        'layout': layout,
        'seq_len': int(T),
        'feat_dim': int(F),
        'max_neighbors': int(M),
        'n_train_shards': len(tr),
        'n_test_shards': len(te),
        'sigma_ratio_max_min': ratio,
        'mu': mu.tolist(),
        'sd': sd.tolist(),
        'center': center.tolist(),
        'auc': {'A_self': res['A'], 'B_self_plus_neighbor': res['B'],
                'C_self_plus_random': res.get('C')},
        'delta_auc': {'B_minus_A': res['dAUC_B_minus_A'],
                      'B_minus_C': res.get('dAUC_B_minus_C')},
        'dilution_train': dil_tr,
        'dilution_test': dil_te,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('\n  -> 结果已保存: %s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())



