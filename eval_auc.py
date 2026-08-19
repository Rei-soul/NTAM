# eval_auc.py - 对已保存模型做 AUC 评估（整体 + 逐分片 + 阈值扫描）
# 用法（在 NTAM 根目录）: python code/eval_auc.py
import os
import torch
import numpy as np
from config import *
from models import NTAM
from data_utils import load_test_shard, get_test_shard_ids
from sklearn.metrics import roc_auc_score, average_precision_score


def load_model(path):
    """加载已保存模型（兼容 train.py 保存格式: ckpt['config']）"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt['config']
    model = NTAM(
        feat_dim=cfg['feat_dim'],
        seq_len=cfg['seq_len'],
        max_neighbors=cfg['max_neighbors'],
        num_layers=cfg['transformer_layers'],
        nhead=cfg['num_heads'],
        dropout=cfg['dropout'],
        use_neighborhood=cfg['use_neighborhood'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def collect_probs(model):
    """遍历全部测试分片，返回 probs, labels, sids"""
    probs, labels, sids = [], [], []
    with torch.no_grad():
        for s in get_test_shard_ids():
            loader = load_test_shard(s)
            for sf, nf, nm, lb in loader:
                p, _ = model(sf.to(DEVICE), nf.to(DEVICE), nm.to(DEVICE))
                probs.append(p.detach().cpu().numpy().ravel())
                labels.append(lb.cpu().numpy().ravel())
                sids.append(np.full(len(lb), s))
    if not probs:
        raise RuntimeError("测试分片为空，请先运行 train.py 生成测试分片")
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(sids)


def metrics_at_threshold(probs, labels, t):
    """给定阈值计算 P/R/F1/F0.5 与混淆矩阵"""
    preds = (probs > t).astype(float)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    f0_5 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)
    return prec, rec, f1, f0_5, int(tp), int(fp), int(fn)


def main():
    print("=" * 78)
    print("[AUC 评估]")
    print("=" * 78)
    path = os.path.join(SAVE_DIR, "ntam_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到模型: {path}")
    model, ckpt = load_model(path)
    print(f"✅ 加载模型: {path} (best_epoch={ckpt.get('best_epoch')}, "
          f"neighborhood={ckpt['config']['use_neighborhood']})")

    probs, labels, sids = collect_probs(model)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    roc = roc_auc_score(labels, probs)
    pr = average_precision_score(labels, probs)

    print(f"\n=== 整体指标 ===")
    print(f"测试样本: {len(labels):,} | 正:{n_pos:,} 负:{n_neg:,} | 正负比 1:{n_neg // max(n_pos, 1)}")
    print(f"ROC-AUC = {roc:.4f} | PR-AUC = {pr:.4f}")

    print(f"\n=== 逐分片（shard 递增 = 时间演进）===")
    for s in get_test_shard_ids():
        m = sids == s
        lbl = labels[m]
        pv = probs[m]
        sp = int((lbl == 1).sum())
        sn = int((lbl == 0).sum())
        r = roc_auc_score(lbl, pv) if len(np.unique(lbl)) >= 2 else float('nan')
        pa = average_precision_score(lbl, pv) if sp > 0 else float('nan')
        print(f"shard{s:02d}: ROC-AUC={r:.4f} PR-AUC={pa:.4f} | 正:{sp} 负:{sn} | 比例 1:{sn // max(sp, 1)}")

    print(f"\n=== 阈值扫描 ===")
    print(f"{'阈值':>5} | {'Prec':>7} {'Rec':>7} {'F1':>7} {'F0.5':>7} | {'TP':>5} {'FP':>7} {'FN':>5}")
    best_f1, best_f05 = None, None
    for t in np.arange(0.30, 0.91, 0.05):
        prec, rec, f1, f05, tp, fp, fn = metrics_at_threshold(probs, labels, round(float(t), 2))
        print(f"{t:5.2f} | {prec:7.4f} {rec:7.4f} {f1:7.4f} {f05:7.4f} | {tp:5d} {fp:7d} {fn:5d}")
        if best_f1 is None or f1 > best_f1[2]:
            best_f1 = (t, prec, f1)
        if best_f05 is None or f05 > best_f05[2]:
            best_f05 = (t, prec, f05)
    print("-" * 78)
    print(f"最优F1  : 阈值 t={best_f1[0]:.2f} → F1={best_f1[2]:.4f} (P={best_f1[1]:.4f})")
    print(f"最优F0.5: 阈值 t={best_f05[0]:.2f} → F0.5={best_f05[2]:.4f} (P={best_f05[1]:.4f})")
    print("=" * 78)


if __name__ == "__main__":
    main()
