#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_attention_saturation.py —— 邻域注意力是否“饱和”的体检

目的：用数据判断 NeighborhoodAttention 的 softmax 是否饱和（权重塌缩到单个邻居），
      并量化“加上 1/sqrt(F) 缩放 / 输入归一化”后能改善多少。

四个探针：
  P1 随机初始化 Q/K（结构层面）      : scale=1 与 scale=1/sqrt(F) 对比
  P2 ckpt 权重（训练后）             : 真实情况
  P3 缩放敏感性（同一 ckpt）         : scale=1 / 1/sqrt(F) / 1/sqrt(F)+输入LayerNorm
  P4 按有效邻居数 M 分层             : ★ 必须分层！M=1 时熵必然为 0，会误判饱和

关键指标（只在 M_eff >= 2 的样本上有意义）：
  H_norm = 熵 / log(M_eff)    → 1 表示权重均匀，接近 0 表示塌缩（饱和）
  maxw   = 最大注意力权重     → 接近 1 表示“只挑一个邻居”
  |logit| = mask 内点积的绝对值 → 无缩放时随 FEAT_DIM 增大而增大

用法：
  python diagnose_attention_saturation.py                    # 自动找 saved_models/ntam_best.pt
  python diagnose_attention_saturation.py --max-samples 20000
  python diagnose_attention_saturation.py --ckpt /path/to/ntam_best.pt
  python diagnose_attention_saturation.py --dir /path/to/processed_data --out attn_check.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import *  # noqa: E402
from models import NeighborhoodAttention  # noqa: E402
from data_utils import TEST_SHARD_PATTERN, get_num_test_shards, PROCESSED_DIR  # noqa: E402

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

EPS = 1e-12
BLOCK = 200000          # 每次处理的 (B*T) 行数


# ============================================================
# 0. 参数
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(description='邻域注意力饱和体检')
    ap.add_argument('--dir', default=None, help='分片目录（默认 data_utils.PROCESSED_DIR）')
    ap.add_argument('--ckpt', default=None,
                    help='模型 checkpoint（默认 <cwd>/saved_models/ntam_best.pt；不存在则用随机权重）')
    ap.add_argument('--shards', type=int, default=1, help='用几个测试分片（默认 1）')
    ap.add_argument('--max-samples', type=int, default=20000, help='最多用多少样本（默认 20000）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='attention_saturation.json')
    return ap.parse_args()


# ============================================================
# 1. 分片加载 / 轴序识别
# ============================================================
def list_test_shards(d):
    fs = sorted(glob.glob(os.path.join(d, 'test_shard_*.npz')))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, 'train_shard_*.npz')))
    return fs


def load_shard_np(path, max_samples, rng):
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
    N, T, Fd = s.shape
    if n.ndim != 4 or n.shape[0] != N:
        raise ValueError('邻居张量形状异常: s=%s n=%s' % (s.shape, n.shape))
    if n.shape[1] == T and n.shape[3] == Fd:
        return 'TMF'          # [N,T,M,F] 新轴序
    if n.shape[2] == T and n.shape[3] == Fd:
        return 'MTF'          # [N,M,T,F] 旧轴序
    raise ValueError('无法识别轴序: s=%s n=%s' % (s.shape, n.shape))


# ============================================================
# 2. 从 ckpt 取 neighborhood.Q / neighborhood.K
# ============================================================
def try_load_qk(ckpt_path, feat_dim):
    """返回 (Q, K, info)；失败返回 (None, None, info)"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None, None, {'loaded': False, 'reason': 'ckpt 不存在: %s' % ckpt_path}
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        return None, None, {'loaded': False, 'reason': 'torch.load 失败: %r' % e}

    meta = ck if isinstance(ck, dict) else {}
    sd = None
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        sd = ck['model_state_dict']
    elif isinstance(ck, dict) and 'state_dict' in ck:
        sd = ck['state_dict']
    elif isinstance(ck, dict) and any(k.endswith('Q.weight') for k in ck.keys()):
        sd = ck

    info = {'loaded': False, 'ckpt': ckpt_path}
    if 'config' in meta:
        info['ckpt_config'] = meta['config']
        info['ckpt_use_neighborhood'] = meta['config'].get('use_neighborhood')
        info['ckpt_seq_len'] = meta['config'].get('seq_len')
    if sd is None:
        info['reason'] = 'checkpoint 里没有 state_dict（且键名不像是裸 state_dict）'
        return None, None, info

    qw = kw = None
    for k in sd:
        if k.endswith('neighborhood.Q.weight'):
            qw = sd[k]
        elif k.endswith('neighborhood.K.weight'):
            kw = sd[k]
    if qw is None or kw is None:
        info['reason'] = ('checkpoint 中没有 neighborhood.Q/K 权重 '
                          '（use_neighborhood 很可能为 False —— 无邻域模型无法体检注意力）')
        return None, None, info
    if qw.shape != (feat_dim, feat_dim) or kw.shape != (feat_dim, feat_dim):
        info['reason'] = '权重形状 %s/%s 与 FEAT_DIM=%d 不匹配' % (tuple(qw.shape), tuple(kw.shape), feat_dim)
        return None, None, info

    na = NeighborhoodAttention(feat_dim)
    with torch.no_grad():
        na.Q.weight.copy_(qw.float())
        na.K.weight.copy_(kw.float())
    info['loaded'] = True
    info['Q_norm'] = float(na.Q.weight.norm())
    info['K_norm'] = float(na.K.weight.norm())
    return na.Q, na.K, info


# ============================================================
# 3. 复算邻域注意力（与 NeighborhoodAttention.forward 完全同式，只多一个 scale）
# ============================================================
def probe(Q, K, s, n, m, layout, scale=1.0, use_ln=False):
    """返回行级（B*T）指标数组。

    行级 = 每个 (样本, 时间步) 一次邻域注意力；mask 对时间步共享。
    只统计权重分布特征，不做加权聚合（聚合在 models.py 里是 r = self + c）。
    """
    N, T, Fd = s.shape
    M = n.shape[2] if layout == 'TMF' else n.shape[1]
    if layout == 'MTF':
        n = np.transpose(n, (0, 2, 1, 3))          # -> [N,T,M,F]

    sf = torch.from_numpy(np.ascontiguousarray(s)).reshape(N * T, Fd)
    nf = torch.from_numpy(np.ascontiguousarray(n)).reshape(N * T, M, Fd)
    mf = (torch.from_numpy(np.ascontiguousarray(m))
          .unsqueeze(1).expand(-1, T, -1).reshape(N * T, M))
    ln = torch.nn.LayerNorm(Fd) if use_ln else None     # 未训练，仅用于展示“输入归一化”的尺度效应

    keys = ('Meff', 'H_raw', 'H_norm', 'maxw', 'absl', 'spread', 'lstd', 'qnorm', 'knorm', 'wpad')
    acc = {k: [] for k in keys}

    with torch.no_grad():
        for b0 in range(0, N * T, BLOCK):
            b1 = min(b0 + BLOCK, N * T)
            x, ki, mm = sf[b0:b1], nf[b0:b1], mf[b0:b1]
            if ln is not None:
                x = ln(x)
                ki = ln(ki)

            q = Q(x)                                     # [b, F]
            k = K(ki)                                    # [b, M, F]
            sc = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)).squeeze(1) * scale  # [b, M]
            sc_m = sc.masked_fill(~mm, float('-inf'))
            w = torch.nan_to_num(F.softmax(sc_m, dim=-1), nan=0.0)                    # [b, M]

            meff = mm.sum(-1)
            wv = w * mm
            H = -(wv * torch.log(wv + EPS)).sum(-1)
            Hn = H / torch.log(torch.clamp(meff.float(), min=2.0))
            Hn = torch.where(meff >= 2, Hn, torch.full_like(Hn, float('nan')))

            acc['Meff'].append(meff.cpu().numpy())
            acc['H_raw'].append(H.cpu().numpy())
            acc['H_norm'].append(Hn.cpu().numpy())
            acc['maxw'].append(w.max(-1).values.cpu().numpy())
            acc['absl'].append(
                (sc.masked_fill(~mm, 0.0).abs().sum(-1) / torch.clamp(meff.float(), min=1.0)).cpu().numpy())
            # ★ 决定 softmax 是否饱和的是 logit 之间的【极差/标准差】，而不是 |logit| 的绝对值
            m1 = meff >= 1
            mx = sc.masked_fill(~mm, float('-inf')).max(-1).values
            mn = sc.masked_fill(~mm, float('inf')).min(-1).values
            acc['spread'].append(torch.where(m1, mx - mn, torch.zeros_like(mx)).cpu().numpy())
            mean_l = sc.masked_fill(~mm, 0.0).sum(-1) / torch.clamp(meff.float(), min=1.0)
            var_l = (((sc - mean_l.unsqueeze(-1)) ** 2) * mm).sum(-1) / torch.clamp(meff.float(), min=1.0)
            acc['lstd'].append(torch.where(m1, torch.sqrt(var_l.clamp(min=0)), torch.zeros_like(var_l)).cpu().numpy())
            acc['qnorm'].append(q.norm(dim=-1).cpu().numpy())
            acc['knorm'].append((k.norm(dim=-1) * mm).sum(-1).div(torch.clamp(meff.float(), min=1.0)).cpu().numpy())
            acc['wpad'].append(((~mm).float() * w).sum(-1).cpu().numpy())

    out = {k: np.concatenate(v).astype(np.float64) for k, v in acc.items()}
    out['n_rows'] = np.array([len(out['Meff'])], dtype=np.float64)
    out['scale'] = scale
    out['use_ln'] = use_ln
    return out


def _med(a):
    """中位数；忽略 NaN（全 NaN 返回 nan，且不触发 numpy 警告）"""
    if a.size == 0:
        return float('nan')
    fin = a[np.isfinite(a)]
    return float(np.median(fin)) if fin.size else float('nan')


def summarize(p, tag):
    """整体 + 按有效邻居数分层的统计"""
    Meff = p['Meff']
    res = {'tag': tag, 'scale': p['scale'], 'use_ln': bool(p['use_ln']),
           'n_rows': int(Meff.size),
           'no_neighbor_ratio': float((Meff == 0).mean()) if Meff.size else float('nan')}

    m2 = Meff >= 2
    _hn = p['H_norm'][m2]
    _hn = _hn[np.isfinite(_hn)]
    res['overall_M_ge2'] = {
        'n': int(m2.sum()),
        'H_norm_median': _med(p['H_norm'][m2]),
        'H_norm_mean': float(_hn.mean()) if _hn.size else float('nan'),
        'maxw_median': _med(p['maxw'][m2]),
        'maxw_gt_0.9_ratio': float((p['maxw'][m2] > 0.9).mean()) if m2.sum() else float('nan'),
        'maxw_gt_0.99_ratio': float((p['maxw'][m2] > 0.99).mean()) if m2.sum() else float('nan'),
        'abs_logit_median': _med(p['absl'][m2]),
        'logit_spread_median': _med(p['spread'][m2]),
        'logit_std_median': _med(p['lstd'][m2]),
        'qnorm_median': _med(p['qnorm'][m2]),
    }
    res['wpad_max'] = float(p['wpad'].max()) if Meff.size else float('nan')   # 校验 masked_fill（应恒为 0）

    by_m = {}
    for mv in range(1, int(Meff.max()) + 1 if Meff.size else 1):
        msk = Meff == mv
        if msk.sum() == 0:
            continue
        by_m[str(mv)] = {
            'n': int(msk.sum()),
            'H_norm_median': _med(p['H_norm'][msk]),
            'maxw_median': _med(p['maxw'][msk]),
            'maxw_gt_0.9_ratio': float((p['maxw'][msk] > 0.9).mean()),
            'abs_logit_median': _med(p['absl'][msk]),
            'logit_spread_median': _med(p['spread'][msk]),
        }
    res['by_M'] = by_m
    return res


def print_summary(s):
    o = s['overall_M_ge2']
    print('  [%s] scale=%.4f%s | 行数 %d（无邻居 %.1f%%）| wpad_max=%.2e'
          % (s['tag'], s['scale'], ' +输入LN' if s['use_ln'] else '', s['n_rows'],
             100 * s['no_neighbor_ratio'], s['wpad_max']))
    if o['n'] == 0:
        print('      （没有 M>=2 的样本，无法评估饱和）')
        return
    print('      M>=2: n=%d | H_norm 中位 %.3f | maxw 中位 %.3f | maxw>0.9 %.1f%% | logit spread 中位 %.2f | |logit| 中位 %.2f'
          % (o['n'], o['H_norm_median'], o['maxw_median'],
             100 * o['maxw_gt_0.9_ratio'], o['logit_spread_median'], o['abs_logit_median']))
    for mv, d in sorted(s['by_M'].items(), key=lambda x: int(x[0])):
        hn = '  --  ' if np.isnan(d['H_norm_median']) else '%.3f' % d['H_norm_median']
        print('      M=%s: n=%-7d H_norm %s | maxw 中位 %.3f | maxw>0.9 %.1f%% | spread %.2f'
              % (mv, d['n'], hn, d['maxw_median'], 100 * d['maxw_gt_0.9_ratio'],
                 d.get('logit_spread_median', float('nan'))))


# ============================================================
# 4. 主流程
# ============================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    d = args.dir or PROCESSED_DIR
    print('=' * 84)
    print('  邻域注意力饱和体检 (NeighborhoodAttention softmax)')
    print('=' * 84)
    print('  分片目录:', d)

    shards = list_test_shards(d)[:max(1, args.shards)]
    if not shards:
        print('[错误] 没有找到 *_shard_*.npz')
        return 1

    s, n, m, l = load_shard_np(shards[0], args.max_samples, rng)
    layout = infer_layout(s, n)
    N, T, Fd = s.shape
    M = n.shape[2] if layout == 'TMF' else n.shape[1]
    print('  分片: %s | s=%s n=%s m=%s' % (os.path.basename(shards[0]), s.shape, n.shape, m.shape))
    print('  轴序=%s | F=%d  =>  1/sqrt(F) = %.4f' % (layout, Fd, Fd ** -0.5))
    print('  评估规模: %d 行 (= %d 样本 × %d 天)' % (N * T, N, T))

    msum = m.sum(1)
    vals, cnts = np.unique(msum, return_counts=True)
    print('  有效邻居数分布:', {int(a): int(b) for a, b in zip(vals, cnts)})
    print('  无邻居样本占比: %.1f%%   （这些样本的邻域分支会静默退化为恒等）'
          % (100 * float((msum == 0).mean())))

    # ---------- 输入尺度自检：分片是 n_ 原值，还是已做过 Z-score？ ----------
    # 说明: 只看"是否 unit-variance"是不够的。实测经验: per-(model,col) Z-score 之后，
    #       逐列 std 仍然可能远小于 1 —— 因为 SMART 列重尾，"样本 std" 被少数极端值撑大，
    #       于是绝大多数格子的 z 反而很小。所以这里同时报告 clip 痕迹(|x|>=4.999)。
    flat = np.nan_to_num(s).reshape(-1, Fd)
    col_mean = flat.mean(0)
    col_std = flat.std(0)
    nz = col_std > 1e-9
    ex2 = float((flat ** 2).mean())
    zero_ratio = float((flat == 0).mean())
    clip5_ratio = float((np.abs(flat) >= 4.999).mean())
    std_ratio_all = float(col_std.max() / max(col_std.min(), 1e-9))
    std_ratio = float(col_std[nz].max() / col_std[nz].min()) if nz.any() else float('nan')
    looks_z = bool((np.abs(col_mean).max() < 0.25)
                   and (0.8 < float(np.median(col_std[nz]) if nz.any() else 0.0) < 1.3))
    print('\n  [输入尺度自检] —— 判断分片的数据管线与尺度健康度')
    print('  逐列 mean 范围 [%.3f, %.3f] | 逐列 std 范围 [%.4f, %.4f]（含常量列）'
          % (col_mean.min(), col_mean.max(), col_std.min(), col_std.max()))
    print('  非零列 std 范围 [%.4f, %.4f] | std_max/std_min = %.1fx  （排除 %d 个常量列）'
          % (col_std[nz].min() if nz.any() else float('nan'),
             col_std[nz].max() if nz.any() else float('nan'),
             std_ratio, int((~nz).sum())))
    print('  E[x^2] = %.4f | 整体 min/max = %.3f / %.3f | 0 值占比 %.2f%%'
          % (ex2, float(flat.min()), float(flat.max()), 100 * zero_ratio))
    print('  |x|>=4.999 占比 %.5f%%   (非 0 => 数据来自 clip(±5) 的 Z-score 管线)'
          % (100 * clip5_ratio))
    print('  → 尺度判定: %s'
          % ('像【unit-variance / 已充分标准化】' if looks_z else
             '不是 unit-variance（可能是原值，也可能是 Z-score 后仍尺度偏小 —— 结合上一行 clip 痕迹判断）'))
    if (not looks_z) and clip5_ratio > 0:
        print('  [!] 有 clip 痕迹但逐列 std 仍远小于 1 => 已 Z-score 但被重尾 σ 拉小：')
        print('      后果: Q/K 点积小 → logit_spread 小 → softmax 趋于均匀 → 邻域退化为平均池化(稀释异常)')
        print('      对策: ① 输入端 LayerNorm(最省) ② 稳健标准化(median/MAD) 代替 mean/std')

    scale_plain = 1.0
    scale_scaled = Fd ** -0.5

    # ---------- 探针 ----------
    reports = {}
    print('\n' + '-' * 84)
    print('[P1] 随机初始化 Q/K（结构层面：模型一出生就饱和吗？）')
    na_rand = NeighborhoodAttention(Fd)
    with torch.no_grad():
        na_rand.Q.weight.copy_(torch.empty(Fd, Fd).uniform_(-Fd ** -0.5, Fd ** -0.5))
        na_rand.K.weight.copy_(torch.empty(Fd, Fd).uniform_(-Fd ** -0.5, Fd ** -0.5))
    p1a = summarize(probe(na_rand.Q, na_rand.K, s, n, m, layout, scale_plain), 'P1a_rand_scale1')
    p1b = summarize(probe(na_rand.Q, na_rand.K, s, n, m, layout, scale_scaled), 'P1b_rand_scaled')
    p1c = summarize(probe(na_rand.Q, na_rand.K, s, n, m, layout, scale_scaled, use_ln=True),
                    'P1c_rand_scaled_LN')
    for r in (p1a, p1b, p1c):
        print_summary(r)
    reports.update({r['tag']: r for r in (p1a, p1b, p1c)})

    ckpt_path = args.ckpt
    if ckpt_path is None:
        cand = os.path.join(SAVE_DIR, 'ntam_best.pt')
        ckpt_path = cand
    Q, K, info = try_load_qk(ckpt_path, Fd)
    print('\n' + '-' * 84)
    print('[P2/P3] 训练后权重（ckpt: %s）' % ckpt_path)
    if not info.get('loaded'):
        print('  ⚠️ 无法使用 ckpt 权重: %s' % info.get('reason', '未知原因'))
        print('     -> 仅结构层面结论可用（P1）。要体检训练后的注意力，请用 USE_NEIGHBORHOOD=True 训练一个 ckpt。')
        ref = p1a
        ref_name = 'P1a_rand_scale1'
        alt = p1b
        alt_name = 'P1b_rand_scaled'
    else:
        print('  ckpt 配置:', info.get('ckpt_config'))
        print('  ||Q||=%.3f  ||K||=%.3f' % (info.get('Q_norm', float('nan')), info.get('K_norm', float('nan'))))
        p2a = summarize(probe(Q, K, s, n, m, layout, scale_plain), 'P2a_ckpt_scale1')
        p2b = summarize(probe(Q, K, s, n, m, layout, scale_scaled), 'P2b_ckpt_scaled')
        p2c = summarize(probe(Q, K, s, n, m, layout, scale_scaled, use_ln=True), 'P2c_ckpt_scaled_LN')
        for r in (p2a, p2b, p2c):
            print_summary(r)
        reports.update({r['tag']: r for r in (p2a, p2b, p2c)})
        ref, ref_name, alt, alt_name = p2a, 'P2a_ckpt_scale1', p2b, 'P2b_ckpt_scaled'
    reports['ckpt_info'] = info

    # ---------- 结论 ----------
    o = ref['overall_M_ge2']
    hn = o['H_norm_median']
    ratio09 = o['maxw_gt_0.9_ratio']
    if o['n'] == 0:
        verdict, level = '样本不足，无法判定（没有 M>=2 的样本）', 'unknown'
    elif hn < 0.30 and ratio09 > 0.50:
        verdict, level = '饱和成立：权重塌缩到单个邻居，softmax 梯度≈0，邻域分支实际学不动', 'saturated'
    elif hn < 0.55:
        verdict, level = '轻度集中（部分饱和）：权重偏向少数邻居，表达力受限', 'mild'
    else:
        verdict, level = '未见明显饱和：权重分布较均匀，问题更可能在融合方式而非缩放', 'ok'

    gain = alt['overall_M_ge2']['H_norm_median'] - hn
    spread_scaled = o['logit_spread_median'] * scale_scaled  # ★ 缩放后 logit 极差（softmax 饱和的直接决定量）
    absl_scaled = o['abs_logit_median'] * scale_scaled
    ln_report = None
    for tag in ('P2c_ckpt_scaled_LN', 'P1c_rand_scaled_LN'):
        if tag in reports:
            ln_report = reports[tag]
            break

    print('\n' + '=' * 84)
    print('  结论')
    print('=' * 84)
    print('  基准（%s, scale=1）: H_norm 中位 %.3f | maxw 中位 %.3f | maxw>0.9 占比 %.1f%% | |logit| 中位 %.2f'
          % (ref_name, hn, o['maxw_median'], 100 * ratio09, o['abs_logit_median']))
    print('  判定: %s' % verdict)
    print('  加 1/sqrt(F)=%.4f 缩放后（%s）: H_norm 中位 %.3f  =>  提升 %+.3f'
          % (scale_scaled, alt_name, alt['overall_M_ge2']['H_norm_median'], gain))
    print('  logit 极差: 缩放前 %.2f -> 缩放后 %.2f   （softmax 是否饱和由【极差】决定；'
          '|logit| 绝对值大 ≠ 饱和）'
          % (o['logit_spread_median'], spread_scaled))
    if ln_report is not None:
        print('  再叠加输入 LayerNorm（%s）: H_norm 中位 %.3f  =>  相对基准 %+.3f'
              % (ln_report['tag'], ln_report['overall_M_ge2']['H_norm_median'],
                 ln_report['overall_M_ge2']['H_norm_median'] - hn))
    print()

    if level == 'saturated':
        if gain < 0.10:
            print('  ⚠️ 注意: 单靠缩放【不足以】摆脱饱和 —— 缩放后 logit 极差仍 ≈ %.1f（需降到 ~几 才会解除饱和）。'
                  % spread_scaled)
            print('     含义: Q/K 权重范数本身过大（ckpt 信息里的 ||Q||/||K|| 可佐证）。')
            print('     组合拳才能修好:')
            print('       ① 输入归一化(Z-score / LayerNorm)  ——   把特征尺度压到 ~N(0,1)')
            print('       ② 缩小 Q/K 初始化（如 std=0.02）或加 weight_decay')
            print('       ③ 再叠 scores *= F**-0.5')
            if ln_report is not None:
                print('     证据: 仅加输入 LayerNorm 后 H_norm 中位 = %.3f（基准 %.3f）'
                      % (ln_report['overall_M_ge2']['H_norm_median'], hn))
        else:
            print('  建议: ① scores *= F**-0.5（本次体检显示 H_norm 可从 %.3f 提到 %.3f）；' % (hn, alt['overall_M_ge2']['H_norm_median']))
            print('        ② 输入做 Z-score / LayerNorm（当前列间 sigma 差异可达 42x）；')
            print('        ③ 之后再看门控/后置融合，否则“融合效果”会被饱和掩盖。')
    elif level == 'mild':
        print('  建议: ① 加 1/sqrt(F) 缩放（低风险）；② 检查输入尺度是否不齐（Z-score）；')
        print('        ③ 重点验证门控/后置差分融合。')
    elif level == 'ok':
        print('  建议: 缩放不是主要矛盾 -> 优先改融合方式（门控 r=self+g*c，或后置 concat[s_self, s_nb, s_self-s_nb]）；')
        print('        同时对齐 lead time（训练 L=5 vs 测试 TEST_LEAD_TIME=7）并检查类不平衡设置。')

    # ---------- 落盘 ----------
    payload = {
        'shard': os.path.basename(shards[0]), 'shard_dir': d, 'layout': layout,
        'feat_dim': int(Fd), 'max_neighbors': int(M), 'n_samples': int(N), 'seq_len': int(T),
        'neighbor_count_dist': {int(a): int(b) for a, b in zip(vals, cnts)},
        'no_neighbor_ratio': float((msum == 0).mean()),
        'input_scale': {
            'col_mean': [float(v) for v in col_mean],
            'col_std': [float(v) for v in col_std],
            'std_max_over_min': std_ratio,
            'std_max_over_min_all': std_ratio_all,
            'E_x2': ex2,
            'min': float(flat.min()), 'max': float(flat.max()),
            'zero_ratio': zero_ratio,
            'clip5_ratio': clip5_ratio,
            'looks_unit_variance': looks_z,
        },
        'reports': reports,
        'verdict': verdict, 'level': level,
        'ref': ref_name, 'alt': alt_name,
        'H_norm_gain_from_scaling': float(gain),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('\n  -> 结果已保存: %s' % os.path.abspath(args.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())


