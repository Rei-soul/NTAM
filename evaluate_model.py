# evaluate_model.py
# 加载已保存的最佳模型 (datasets/processed/best_model.pt) 并在测试集上完整评估
# 用法（必须在 NTAM 根目录运行）:
#   python code/evaluate_model.py
# 依赖: 先跑过 train.py 并生成 best_model.pt
import os
import torch
import numpy as np
import config
from models import NTAM
from data_utils import load_test_shard, get_num_test_shards
from sklearn.metrics import roc_auc_score, average_precision_score


def load_best_model(path=config.MODEL_SAVE_PATH):
    """从 .pt 文件加载最佳模型，返回 (model, metadata)"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到模型文件: {path}\n请先运行 train.py 完成训练（会自动保存最佳模型）。")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    model = NTAM(
        feat_dim=meta["feat_dim"],
        seq_len=meta["seq_len"],
        max_neighbors=meta["max_neighbors"],
        num_layers=meta["num_layers"],
        nhead=meta["nhead"],
        dropout=meta["dropout"],
        use_neighborhood=meta["use_neighborhood"],
    ).to(config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta


def collect_probs(model):
    """遍历所有测试分片，返回 (probs, labels)"""
    probs, labels = [], []
    with torch.no_grad():
        for s in range(get_num_test_shards()):
            loader = load_test_shard(s)
            for sf, nf, nm, lb in loader:
                p, _ = model(sf.to(config.DEVICE), nf.to(config.DEVICE), nm.to(config.DEVICE))
                probs.append(p.detach().cpu().numpy().ravel())
                labels.append(lb.detach().cpu().numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels)


def threshold_metrics(probs, labels, t):
    preds = probs > t
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    f0_5 = 1.25 * prec * rec / max(0.25 * prec + rec, 1e-9)  # F0.5（β=0.5，更看重精确率）
    return prec, rec, f1, f0_5, tp, fp, fn


def main():
    print("=" * 60)
    model, meta = load_best_model()
    print(f"加载最佳模型成功 | 来源: {config.MODEL_SAVE_PATH}")
    print(f"  结构: feat_dim={meta['feat_dim']} seq_len={meta['seq_len']} "
          f"neighbors={meta['max_neighbors']} layers={meta['num_layers']} heads={meta['nhead']} "
          f"use_neighborhood={meta['use_neighborhood']}")
    print(f"  训练信息: 最佳epoch={meta['best_epoch']} | 当时 F1={meta['test_f1']:.4f} "
          f"(P={meta['test_prec']:.4f}, R={meta['test_rec']:.4f}) | "
          f"NEG_RATIO={meta['neg_ratio']} batch={meta['batch_size']} lr={meta['learning_rate']}")
    print("=" * 60)

    probs, labels = collect_probs(model)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"\n测试样本: {len(labels):,} | 正:{n_pos:,} 负:{n_neg:,} | 比例 1:{n_neg // max(n_pos, 1)}")
    print(f"ROC-AUC = {roc_auc_score(labels, probs):.4f} | PR-AUC = {average_precision_score(labels, probs):.4f}")

    print(f"\n{'阈值':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F0.5':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    for t in config.THRESHOLDS:
        prec, rec, f1, f0_5, tp, fp, fn = threshold_metrics(probs, labels, t)
        print(f"{t:6.2f} {prec:8.4f} {rec:8.4f} {f1:8.4f} {f0_5:8.4f} {int(tp):6d} {int(fp):6d} {int(fn):6d}")

    prec, rec, f1, f0_5, tp, fp, fn = threshold_metrics(probs, labels, config.PRED_THRESHOLD)
    print(f"\n最终评估结果 (阈值 {config.PRED_THRESHOLD}):")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | F0.5: {f0_5:.4f}")
    print(f"  混淆矩阵: TP:{int(tp)} FP:{int(fp)} FN:{int(fn)} TN:{int(n_neg - fp)}")


if __name__ == "__main__":
    main()
