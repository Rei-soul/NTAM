# fine_tune.py
# 用训练集最后 N 个分片微调已保存的最佳模型（分片级时间权重：越近越偏重）
# 无需重新生成任何分片 —— 微调数据直接复用现有训练分片尾部（分片按时间排序，尾部=最近）
# 用法（在 NTAM 根目录）: python code/fine_tune.py
import os
import torch
import torch.optim as optim
import numpy as np
from config import *
from models import NTAM
from data_utils import load_train_shard, load_test_shard, get_train_shard_ids, get_test_shard_ids
from sklearn.metrics import roc_auc_score


def load_model(path):
    """加载已保存的 best 模型（兼容 train.py 保存格式: ckpt['config']）"""
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
    """遍历全部测试分片，返回 (probs, labels, sids)；sids 用于按分片分组（分片按时间排序）"""
    probs, labels, sids = [], [], []
    model.eval()
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


def evaluate(model, tag):
    """整体 + 逐分片评估，返回指标 dict"""
    probs, labels, sids = collect_probs(model)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    auc = roc_auc_score(labels, probs)
    prec, rec, f1, f0_5, tp, fp, fn = metrics_at_threshold(probs, labels, PRED_THRESHOLD)
    print(f"[{tag}] 整体 (阈值 {PRED_THRESHOLD}): AUC={auc:.4f} | P={prec:.4f} R={rec:.4f} "
          f"F1={f1:.4f} F0.5={f0_5:.4f} | 正:{n_pos} 负:{n_neg} | TP:{tp} FP:{fp} FN:{fn}")
    print(f"[{tag}] 逐分片（shard 号递增 = 时间演进）:")
    for s in get_test_shard_ids():
        m = sids == s
        if m.sum() == 0:
            continue
        lbl = labels[m]
        sp = int((lbl == 1).sum())
        if len(np.unique(lbl)) < 2:
            a = float('nan')
        else:
            a = roc_auc_score(lbl, probs[m])
        p, r, f1s, f05s, *_ = metrics_at_threshold(probs[m], lbl, PRED_THRESHOLD)
        print(f"    shard{s:02d}: AUC={a:7.4f} P={p:.4f} R={r:.4f} F1={f1s:.4f} F0.5={f05s:.4f} | 正:{sp}")
    return {'auc': auc, 'prec': prec, 'rec': rec, 'f1': f1, 'f0_5': f0_5}, probs, labels


def scan_thresholds(probs, labels, start=0.10, end=0.90, step=0.05):
    """扫描阈值，返回 [(t, P, R, F1, F0.5, TP, FP, FN), ...]"""
    rows = []
    t = start
    while t <= end + 1e-9:
        p, r, f1, f05, tp, fp, fn = metrics_at_threshold(probs, labels, round(t, 2))
        rows.append((round(t, 2), p, r, f1, f05, tp, fp, fn))
        t += step
    return rows


def print_threshold_scan(probs_before, labels_before, probs_after, labels_after):
    """打印微调前后在各阈值下的指标 + 各自最优阈值"""
    print("\n" + "=" * 78)
    print("=== 阈值扫描（微调前后在各阈值下的 P/R/F1/F0.5）===")
    print(f"{'阈值':>5} | {'前P':>7} {'前R':>7} {'前F1':>7} {'前F0.5':>7} | "
          f"{'后P':>7} {'后R':>7} {'后F1':>7} {'后F0.5':>7}")
    print("-" * 78)
    rows_b = scan_thresholds(probs_before, labels_before)
    rows_a = scan_thresholds(probs_after, labels_after)
    best_f1_b = max(rows_b, key=lambda r: r[3])
    best_f1_a = max(rows_a, key=lambda r: r[3])
    best_f05_b = max(rows_b, key=lambda r: r[4])
    best_f05_a = max(rows_a, key=lambda r: r[4])
    for rb, ra in zip(rows_b, rows_a):
        print(f"{rb[0]:5.2f} | {rb[1]:7.4f} {rb[2]:7.4f} {rb[3]:7.4f} {rb[4]:7.4f} | "
              f"{ra[1]:7.4f} {ra[2]:7.4f} {ra[3]:7.4f} {ra[4]:7.4f}")
    print("-" * 78)
    print(f"最优F1  : 微调前 t={best_f1_b[0]:.2f} F1={best_f1_b[3]:.4f} | "
          f"微调后 t={best_f1_a[0]:.2f} F1={best_f1_a[3]:.4f}")
    print(f"最优F0.5: 微调前 t={best_f05_b[0]:.2f} F0.5={best_f05_b[4]:.4f} | "
          f"微调后 t={best_f05_a[0]:.2f} F0.5={best_f05_a[4]:.4f}")
    print("=" * 78)
def main():
    print("=" * 78)
    print("[Fine-tune] 用训练集最后几个分片微调（分片级时间权重）")
    print("=" * 78)

    # 1. 加载已保存的最佳模型
    best_path = os.path.join(SAVE_DIR, "ntam_best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"未找到模型: {best_path}\n请先运行 train.py 完成训练")
    model, ckpt = load_model(best_path)
    print(f"✅ 已加载 best 模型: {best_path} (best_epoch={ckpt.get('best_epoch')})")

    # 2. 微调前评估（基线）
    print("\n[评估] 微调前基线：")
    before, probs_before, labels_before = evaluate(model, "微调前")

    # 3. 确定微调分片与权重
    train_ids = get_train_shard_ids()
    ft_shard_ids = train_ids[-FINE_TUNE_LAST_SHARDS:]
    if not ft_shard_ids:
        raise RuntimeError(f"训练分片不足（当前 {len(train_ids)} 片），FINE_TUNE_LAST_SHARDS 过大")
    if FINE_TUNE_SHARD_WEIGHTS and len(FINE_TUNE_SHARD_WEIGHTS) == len(ft_shard_ids):
        shard_weights = dict(zip(ft_shard_ids, FINE_TUNE_SHARD_WEIGHTS))
    else:
        shard_weights = {sid: 1.0 for sid in ft_shard_ids}
    print(f"\n[微调] 分片: {ft_shard_ids} | 分片权重: {shard_weights} | "
          f"epochs={FINE_TUNE_EPOCHS} lr={FINE_TUNE_LR}")

    # 4. 微调训练（小学习率 + 分片级损失权重：越近越偏重）
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    model.train()
    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        total_loss = 0.0
        total_samples = 0
        for shard_id in ft_shard_ids:
            w = shard_weights[shard_id]
            loader = load_train_shard(shard_id)
            for sf, nf, nm, lb in loader:
                sf, nf, nm, lb = sf.to(DEVICE), nf.to(DEVICE), nm.to(DEVICE), lb.to(DEVICE)
                optimizer.zero_grad()
                _, logits = model(sf, nf, nm)
                loss = criterion(logits, lb) * w  # 分片级时间权重
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(sf)
                total_samples += len(sf)
        print(f"  Epoch {epoch}/{FINE_TUNE_EPOCHS}: loss={total_loss / max(total_samples, 1):.4f}")

    # 5. 微调后评估
    print("\n[评估] 微调后：")
    after, probs_after, labels_after = evaluate(model, "微调后")

    # 6. 对比表
    print("\n" + "=" * 78)
    print("=== 微调前后对比 ===")
    print(f"{'指标':>6} | {'微调前':>10} | {'微调后':>10} | {'Δ':>10}")
    print("-" * 46)
    for k in ['auc', 'prec', 'rec', 'f1', 'f0_5']:
        print(f"{k:>6} | {before[k]:10.4f} | {after[k]:10.4f} | {after[k] - before[k]:+10.4f}")
    print("=" * 78)

    # 7. 阈值扫描对比（AUC 提升但固定阈值下指标下降时，看各自最优阈值）
    print_threshold_scan(probs_before, labels_before, probs_after, labels_after)

    # 8. 保存微调模型（不覆盖原 best）
    os.makedirs(os.path.dirname(FINE_TUNE_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': ckpt['config'],
        'base_model': os.path.basename(best_path),
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'fine_tune_lr': FINE_TUNE_LR,
        'fine_tune_last_shards': FINE_TUNE_LAST_SHARDS,
        'fine_tune_shard_weights': shard_weights,
        'final_metrics': after,
        'pred_threshold': PRED_THRESHOLD,
    }, FINE_TUNE_SAVE_PATH)
    print(f"✅ 微调模型已保存: {os.path.abspath(FINE_TUNE_SAVE_PATH)}")


if __name__ == "__main__":
    main()

