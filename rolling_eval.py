# rolling_eval.py
# 滚动重训 + 流式更新评估脚本（walk-forward + cumulative data refeed）
#
# 协议:
#   时间轴切成多个不重叠的评估窗口；每个窗口用"其之前一段历史(TPS) +
#   所有已评估测试样本(累积回灌)"增量训练模型，再评估该窗口。
#   已评估测试样本(带真实标签)回灌进训练池，实现流式/在线更新。
#
# 用法(在 NTAM 根目录下):
#   python .\NTAM-main\rolling_eval.py
#   python .\NTAM-main\rolling_eval.py --start 20180515 --end 20181231
#   python .\NTAM-main\rolling_eval.py --train-window 90 --test-window 14 --epochs 2
#   python .\NTAM-main\rolling_eval.py --refeed-neg-cap 20000
#   python .\NTAM-main\rolling_eval.py --no-warm-start --epochs 5   # 每周期从零训练

import argparse
import bisect
import glob
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from config import *
from models import NTAM
from data_utils import (FeatStore, PROCESSED_DIR, _scan_csv_dates,
                        _load_disk_info, _load_neighbor_map, _get_all_pids)
from eval_saved_model import compute_auc, compute_metrics


def parse_date(s):
    return s


def di_clamp(dates, d_str, lo):
    """把日期字符串映射到 dates 中的索引（lo=True 取>=d_str 的最小索引，否则取<=d_str 的最大索引）"""
    if lo:
        i = bisect.bisect_left(dates, d_str)
    else:
        i = bisect.bisect_right(dates, d_str) - 1
    return max(0, min(i, len(dates) - 1))


def build_window_entries(dates, date_to_di, disk_info, sampled_pids, neighbor_map,
                         pid_to_extract_idx, tr_s_di, tr_e_di, te_s_di, te_e_di,
                         train_neg_cap, test_neg_cap, rng):
    """构建一个周期的训练/测试样本条目 (w_idx, pi, label)。语义与 data_utils 一致。"""
    train_entries = []
    test_entries = []
    train_cands = [i for i in range(max(tr_s_di, SEQ_LEN - 1), tr_e_di + 1)]
    test_cands_days = [i for i in range(max(te_s_di, SEQ_LEN - 1), te_e_di + 1)]
    healthy_pis = []

    for pi, pid in enumerate(sampled_pids):
        if pid not in pid_to_extract_idx:
            continue
        info = disk_info[pid]
        if info['is_failure'] and info['failure_time'] is not None:
            ft_di = date_to_di.get(info['failure_time'].strftime('%Y%m%d'))
            if ft_di is None:
                continue
            # 训练 TPS: l=1..L
            for l in range(1, L + 1):
                end_di = ft_di - l
                if end_di < SEQ_LEN - 1:
                    break
                if tr_s_di <= end_di <= tr_e_di:
                    train_entries.append((list(range(end_di - SEQ_LEN + 1, end_di + 1)), pi, 1.0))
            # 测试: l=1..TEST_LEAD_TIME 中随机选 1 条
            te_cands = []
            for l in range(1, TEST_LEAD_TIME + 1):
                end_di = ft_di - l
                if end_di < SEQ_LEN - 1:
                    break
                if te_s_di <= end_di <= te_e_di:
                    te_cands.append(list(range(end_di - SEQ_LEN + 1, end_di + 1)))
            if te_cands:
                test_entries.append((te_cands[rng.randint(0, len(te_cands))], pi, 1.0))
        else:
            healthy_pis.append(pi)

    # 健康盘负样本（每盘 1 条，按 cap 降采样控制内存）
    if train_cands and healthy_pis:
        n_sel = min(train_neg_cap, len(healthy_pis))
        for k in rng.choice(len(healthy_pis), size=n_sel, replace=False):
            end_di = train_cands[rng.randint(0, len(train_cands))]
            train_entries.append((list(range(end_di - SEQ_LEN + 1, end_di + 1)), healthy_pis[k], 0.0))
    if test_cands_days and healthy_pis:
        n_sel = min(test_neg_cap, len(healthy_pis))
        for k in rng.choice(len(healthy_pis), size=n_sel, replace=False):
            end_di = test_cands_days[rng.randint(0, len(test_cands_days))]
            test_entries.append((list(range(end_di - SEQ_LEN + 1, end_di + 1)), healthy_pis[k], 0.0))

    return train_entries, test_entries

def materialize(entries, feat_store, sampled_pids, neighbor_map):
    """把条目 (w_idx, pi, label) 物化为内存 numpy 数组 [s, n, m, l]。
    复用 FeatStore 按需读取，语义与 data_utils._copy_sample 一致。"""
    n = len(entries)
    S = np.zeros((n, SEQ_LEN, FEAT_DIM), np.float32)
    N = np.zeros((n, MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM), np.float32)
    M_ = np.zeros((n, MAX_NEIGHBORS), np.bool_)
    L_ = np.zeros((n, 1), np.float32)
    k = 0
    for w_idx, pi, label in entries:
        pid = sampled_pids[pi]
        seq = feat_store.get(pid, w_idx)
        if seq is None or np.isnan(seq).all() or np.all(seq == 0):
            continue
        S[k] = seq
        for j, npid in enumerate(neighbor_map.get(pid, [])[:MAX_NEIGHBORS]):
            nseq = feat_store.get(npid, w_idx)
            if nseq is None or np.isnan(nseq).all() or np.all(nseq == 0):
                continue
            N[k, j] = nseq
            M_[k, j] = True
        L_[k] = label
        k += 1
    return S[:k], N[:k], M_[:k], L_[:k]


def refeed_samples(s, n, m, l, y_true, neg_cap, rng):
    """从已评估测试样本中提取回灌子集：正样本全量 + 负样本按 cap 降采样。"""
    y_true = np.asarray(y_true).ravel()
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if neg_cap > 0 and len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, size=neg_cap, replace=False)
    idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    return s[idx], n[idx], m[idx], l[idx]


def train_model(model, s, n, m, l, epochs, lr, device, batch_size):
    """在训练池上训练模型（warm-start 下从上周期继续）。"""
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(s), torch.from_numpy(n), torch.from_numpy(m).bool(), torch.from_numpy(l))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(epochs):
        for sf, nf, nm, lb in loader:
            sf, nf, nm, lb = sf.to(device), nf.to(device), nm.to(device), lb.to(device)
            opt.zero_grad()
            _, logits = model(sf, nf, nm)
            loss = crit(logits, lb)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


def predict(model, s, n, m, device, batch_size):
    """批量推理，返回预测概率数组 y_score。"""
    model.eval()
    probs = []
    for i in range(0, len(s), batch_size):
        sf = torch.from_numpy(s[i:i + batch_size]).to(device)
        nf = torch.from_numpy(n[i:i + batch_size]).to(device)
        nm = torch.from_numpy(m[i:i + batch_size]).to(device)
        with torch.no_grad():
            prob, _ = model(sf, nf, nm)
        probs.append(prob.cpu().numpy().ravel())
    return np.concatenate(probs)


def best_threshold_metrics(y_true, y_score, grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """在阈值网格上选最大 F1，返回对应指标 dict。"""
    best = None
    for thr in grid:
        mm = compute_metrics(y_true, y_score, thr)
        if best is None or mm['f1'] > best['f1']:
            best = mm
    return best

def main():
    ap = argparse.ArgumentParser(description="滚动重训 + 流式更新评估（walk-forward + cumulative refeed）")
    ap.add_argument("--start", default="20180515", help="滚动开始日期 YYYYMMDD")
    ap.add_argument("--end", default="20181231", help="滚动结束日期 YYYYMMDD")
    ap.add_argument("--train-window", type=int, default=90, help="训练窗口天数")
    ap.add_argument("--test-window", type=int, default=14, help="评估窗口天数/步长")
    ap.add_argument("--epochs", type=int, default=2, help="每周期 epoch 数")
    ap.add_argument("--lr", type=float, default=3e-5, help="每周期学习率")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-neg-cap", type=int, default=200000, help="每周期训练负样本上限")
    ap.add_argument("--test-neg-cap", type=int, default=50000, help="每周期测试负样本上限")
    ap.add_argument("--refeed-neg-cap", type=int, default=20000, help="每窗口回灌负样本上限")
    ap.add_argument("--refeed-mode", default="sliding", choices=["cumulative", "sliding"],
                    help="refeed policy: cumulative=keep all / sliding=keep recent N windows")
    ap.add_argument("--refeed-window", type=int, default=3, help="sliding mode: keep recent N refeed windows")

    ap.add_argument("--warm-start", dest="warm_start", action="store_true", default=True)
    ap.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="rolling_results.json", help="结果输出 JSON")
    ap.add_argument("--save-preds", default=None, help="保存所有周期预测为 npz")
    ap.add_argument("--baseline-auc", type=float, default=0.8592, help="一次性训练基线 AUC")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else (DEVICE if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"))
    print(f"设备: {device} | 滚动 {args.start}~{args.end} | 训练窗口 {args.train_window} 天 / 评估窗口 {args.test_window} 天")

    print("[1] 加载磁盘清单 / 邻居 / 特征索引...")
    dates, _ = _scan_csv_dates()
    feat_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "feat_day_*.npy")))
    if len(feat_files) < len(dates):
        print(f"  feat_day {len(feat_files)} 天（日期列表 {len(dates)} 天），按 feat_day 对齐")
        dates = dates[:len(feat_files)]
    date_to_di = {d: i for i, d in enumerate(dates)}
    disk_info = _load_disk_info()
    neighbor_map = _load_neighbor_map(disk_info)
    sampled_pids = _get_all_pids(disk_info)
    all_needed = set(sampled_pids)
    for pid in sampled_pids:
        all_needed.update(neighbor_map.get(pid, [])[:MAX_NEIGHBORS])
    extract_pids = sorted(all_needed)
    pid_to_extract_idx = {pid: i for i, pid in enumerate(extract_pids)}
    feat_store = FeatStore(feat_files, pid_to_extract_idx, max_cache=30)
    print(f"  目标盘 {len(sampled_pids):,} | 提取盘 {len(extract_pids):,} | 日期 {len(dates)} 天")

    start_di = di_clamp(dates, args.start, True)
    end_di_all = di_clamp(dates, args.end, False)
    cycles = []
    cur = start_di
    while cur <= end_di_all:
        te_e_di = min(cur + args.test_window - 1, end_di_all)
        tr_e_di = cur - 1
        tr_s_di = cur - args.train_window
        if tr_s_di < SEQ_LEN - 1:
            print(f"  ⚠️ 跳过 {dates[cur]}：训练窗口不足")
            cur = te_e_di + 1
            continue
        cycles.append((max(tr_s_di, 0), tr_e_di, cur, te_e_di))
        cur = te_e_di + 1
    print(f"[2] 共 {len(cycles)} 个评估窗口")

    model = None
    acc_S, acc_N, acc_M, acc_L = [], [], [], []
    results = []
    all_preds = []
    for ci, (tr_s_di, tr_e_di, te_s_di, te_e_di) in enumerate(cycles):
        t0 = time.time()
        tr_str = f"{dates[tr_s_di]}~{dates[tr_e_di]}"
        te_str = f"{dates[te_s_di]}~{dates[te_e_di]}"
        print(f"\n[Cycle {ci+1:02d}/{len(cycles)}] 训练 {tr_str} | 评估 {te_str}")

        train_entries, test_entries = build_window_entries(
            dates, date_to_di, disk_info, sampled_pids, neighbor_map, pid_to_extract_idx,
            tr_s_di, tr_e_di, te_s_di, te_e_di, args.train_neg_cap, args.test_neg_cap, rng)
        print(f"  窗口条目: 训练 {len(train_entries):,} | 测试 {len(test_entries):,}")
        win_S, win_N, win_M, win_L = materialize(train_entries, feat_store, sampled_pids, neighbor_map)
        print(f"  物化训练: {len(win_S):,} (正 {int(win_L.sum()):,})")

        if acc_S:
            train_S = np.concatenate([win_S] + acc_S)
            train_N = np.concatenate([win_N] + acc_N)
            train_M = np.concatenate([win_M] + acc_M)
            train_L = np.concatenate([win_L] + acc_L)
        else:
            train_S, train_N, train_M, train_L = win_S, win_N, win_M, win_L
        n_train = len(train_S)
        refeed_cum = sum(len(a) for a in acc_S)
        del win_S, win_N, win_M, win_L
        print(f"  训练池: {n_train:,} (正 {int(train_L.sum()):,} | 累积回灌 {refeed_cum:,})")

        if model is None or not args.warm_start:
            model = NTAM(feat_dim=FEAT_DIM, seq_len=SEQ_LEN, max_neighbors=MAX_NEIGHBORS,
                         num_layers=TRANSFORMER_LAYERS, nhead=NUM_HEADS, dropout=DROPOUT,
                         use_neighborhood=USE_NEIGHBORHOOD).to(device)
        train_model(model, train_S, train_N, train_M, train_L, args.epochs, args.lr, device, args.batch_size)
        del train_S, train_N, train_M, train_L

        te_S, te_N, te_M, te_L = materialize(test_entries, feat_store, sampled_pids, neighbor_map)
        y_score = predict(model, te_S, te_N, te_M, device, args.batch_size)
        y_true = te_L.ravel()
        auc = compute_auc(y_true, y_score)
        m50 = compute_metrics(y_true, y_score, 0.5)
        mbest = best_threshold_metrics(y_true, y_score)
        print(f"  AUC={auc:.4f} | F1@0.5={m50['f1']:.4f} (P {m50['precision']:.3f} R {m50['recall']:.3f}) | "
              f"F1@最佳({mbest['threshold']:.1f})={mbest['f1']:.4f} | 正样本 {int(y_true.sum()):,}")
        results.append(dict(cycle=ci + 1, train_range=tr_str, test_range=te_str,
                            n_train=n_train, n_test=len(te_S), n_pos=int(y_true.sum()),
                            auc=float(auc), f1_05=float(m50['f1']),
                            precision_05=float(m50['precision']), recall_05=float(m50['recall']),
                            best_thr=float(mbest['threshold']), best_f1=float(mbest['f1']),
                            refeed_cum=refeed_cum))
        if args.save_preds:
            all_preds.append((y_true, y_score, te_str))

        rf_S, rf_N, rf_M, rf_L = refeed_samples(te_S, te_N, te_M, te_L, y_true, args.refeed_neg_cap, rng)
        acc_S.append(rf_S); acc_N.append(rf_N); acc_M.append(rf_M); acc_L.append(rf_L)
        if args.refeed_mode == "sliding" and len(acc_S) > args.refeed_window:
            acc_S.pop(0); acc_N.pop(0); acc_M.pop(0); acc_L.pop(0)
        del te_S, te_N, te_M, te_L
        print(f"  回灌 {len(rf_S):,} (正 {int(rf_L.sum()):,}) | 周期耗时 {time.time()-t0:.0f}s")

    print("\n" + "=" * 66)
    print("滚动重训 + 流式更新汇总:")
    aucs = [r['auc'] for r in results]
    f1s = [r['f1_05'] for r in results]
    bf1s = [r['best_f1'] for r in results]
    print(f"  周期数: {len(results)}")
    print(f"  平均 AUC: {np.mean(aucs):.4f}   (基线一次性训练: {args.baseline_auc:.4f})")
    print(f"  平均 F1@0.5: {np.mean(f1s):.4f} | 平均 F1@最佳阈值: {np.mean(bf1s):.4f}")
    print("  各周期 AUC:", " ".join(f"{a:.3f}" for a in aucs))
    print("=" * 66)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), cycles=results,
                       summary=dict(mean_auc=float(np.mean(aucs)), mean_f1_05=float(np.mean(f1s)),
                                    mean_best_f1=float(np.mean(bf1s)), baseline_auc=args.baseline_auc)),
                  f, indent=2, ensure_ascii=False)
    print(f"✅ 结果已保存: {os.path.abspath(args.out)}")

    if args.save_preds:
        np.savez(args.save_preds, labels=np.concatenate([p[0] for p in all_preds]),
                 probs=np.concatenate([p[1] for p in all_preds]),
                 ranges=np.array([p[2] for p in all_preds]))
        print(f"✅ 预测已保存: {os.path.abspath(args.save_preds)}")


if __name__ == "__main__":
    main()