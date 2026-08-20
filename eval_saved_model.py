# eval_saved_model.py
# 独立评估脚本：加载已保存的 NTAM checkpoint，在任意指定的分片上评估。
#
# 用法示例:
#   python eval_saved_model.py                                     # 评估全部测试分片
#   python eval_saved_model.py --shards 0,1,2                      # 指定测试分片
#   python eval_saved_model.py --shards 0-2                        # 范围写法
#   python eval_saved_model.py --shards 0-2,4                      # 混合写法
#   python eval_saved_model.py --set train --shards 3,4            # 评估训练分片
#   python eval_saved_model.py --model saved_models/ntam_best.pt   # 指定模型文件
#   python eval_saved_model.py --thresholds 0.3,0.5,0.7            # 多阈值扫描
#   python eval_saved_model.py --save-preds preds.npz              # 保存预测概率
#   python eval_saved_model.py --device cpu                        # 覆盖设备

import argparse
import os
import sys
import numpy as np
import torch

from config import *
from models import NTAM
from data_utils import (load_train_shard, load_test_shard,
                        get_num_train_shards, get_num_test_shards,
                        TRAIN_SHARD_PATTERN, TEST_SHARD_PATTERN)


def parse_shards(shards_arg, n_available):
    """解析分片参数。'all' = 全部存在分片；支持逗号分隔与范围，如 '0,1,2' / '0-2' / '0-2,4'。"""
    if shards_arg.strip().lower() == 'all':
        return list(range(n_available))
    ids = []
    for tok in shards_arg.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = (int(x) for x in tok.split('-', 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(tok))
    return sorted(set(ids))


def load_model(model_path, device):
    """加载 checkpoint 并重建 NTAM 模型。

    兼容两种存档格式:
      {model_state_dict, config, best_epoch, final_metrics}   # train.py save_trained_model 输出
      裸 state_dict                                            # torch.save(model.state_dict())
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    ck = torch.load(model_path, map_location='cpu', weights_only=False)

    if isinstance(ck, dict) and 'model_state_dict' in ck:
        state_dict, meta = ck['model_state_dict'], ck
    elif isinstance(ck, dict) and 'state_dict' in ck:
        state_dict, meta = ck['state_dict'], ck
    elif isinstance(ck, dict):
        state_dict, meta = ck, {}
    else:
        raise ValueError(f"无法识别的 checkpoint 类型: {type(ck)}")

    cfg = meta.get('config') or {}
    kw = dict(
        feat_dim=cfg.get('feat_dim', FEAT_DIM),
        seq_len=cfg.get('seq_len', SEQ_LEN),
        max_neighbors=cfg.get('max_neighbors', MAX_NEIGHBORS),
        num_layers=cfg.get('transformer_layers', TRANSFORMER_LAYERS),
        nhead=cfg.get('num_heads', NUM_HEADS),
        dropout=cfg.get('dropout', DROPOUT),
        use_neighborhood=cfg.get('use_neighborhood', USE_NEIGHBORHOOD),
    )
    model = NTAM(**kw)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, kw, meta


def compute_auc(y_true, y_score):
    """AUC (ROC)。优先使用 sklearn，缺失时手写 Mann-Whitney U（处理平局）。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        pass
    pos = y_true == 1
    neg = y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and y_score[order[j + 1]] == y_score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based 平均排名
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(y_true, y_score, threshold=0.5):
    """计算给定阈值下的分类指标。"""
    y_true = np.asarray(y_true).astype(np.int64)
    preds = (np.asarray(y_score) >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    n = len(y_true)
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    f05 = (1 + 0.5 ** 2) * prec * rec / max(0.5 ** 2 * prec + rec, 1e-8)
    return dict(threshold=threshold, acc=acc, precision=prec, recall=rec, f1=f1, f0_5=f05,
                tp=tp, fp=fp, fn=fn, tn=tn, n_pos=tp + fn, n_neg=fp + tn)


def run_inference(model, shard_ids, device, set_name='test'):
    """在指定分片上推理，返回 (labels, probs) 与逐分片统计。"""
    labels_all, probs_all = [], []
    per_shard = []
    for sid in shard_ids:
        loader = load_test_shard(sid) if set_name == 'test' else load_train_shard(sid)
        labels_s, probs_s = [], []
        with torch.no_grad():
            for sf, nf, nm, lb in loader:
                sf, nf, nm = sf.to(device), nf.to(device), nm.to(device)
                prob, _ = model(sf, nf, nm)
                labels_s.append(lb.view(-1).numpy())
                probs_s.append(prob.view(-1).cpu().numpy())
        labels_s = np.concatenate(labels_s)
        probs_s = np.concatenate(probs_s)
        n_pos = int(labels_s.sum())
        auc_s = compute_auc(labels_s, probs_s)
        print(f"  [{set_name.capitalize()} Shard {sid:02d}] {len(labels_s):,} 样本 (正:{n_pos:,}) | AUC={auc_s:.4f}")
        labels_all.append(labels_s)
        probs_all.append(probs_s)
        per_shard.append((sid, len(labels_s), n_pos, auc_s))
        del loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.concatenate(labels_all), np.concatenate(probs_all), per_shard


def main():
    ap = argparse.ArgumentParser(description="评估已保存的 NTAM 模型（可任意选择分片）")
    ap.add_argument("--model", default=os.path.join(SAVE_DIR, "ntam_best.pt"),
                    help="checkpoint 路径（默认: saved_models/ntam_best.pt）")
    ap.add_argument("--shards", default="all",
                    help="评估的分片 ID，逗号分隔/范围，如 '0,1,2' 或 '0-2'；默认 all")
    ap.add_argument("--set", default="test", choices=["test", "train"],
                    help="评估测试集还是训练集分片（默认 test）")
    ap.add_argument("--threshold", type=float, default=0.5, help="判定阈值（默认 0.5）")
    ap.add_argument("--thresholds", default=None,
                    help="多阈值扫描（逗号分隔，如 0.3,0.5,0.7），会额外打印扫描表")
    ap.add_argument("--save-preds", default=None, help="将标签与预测概率保存为 .npz（含分片 ID）")
    ap.add_argument("--device", default=None, help="覆盖设备，如 cpu / cuda:0")
    args = ap.parse_args()

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {os.path.abspath(args.model)}")
    model, kw, meta = load_model(args.model, device)
    print(f"模型架构: {kw}")
    if meta.get('best_epoch') is not None:
        print(f"  存档信息: best_epoch={meta.get('best_epoch')}, final_metrics={meta.get('final_metrics')}")

    # 确定要评估的分片
    if args.set == 'test':
        n_available = get_num_test_shards()
    else:
        n_available = get_num_train_shards()
    if n_available == 0:
        print(f"⚠️ 未找到任何 {args.set} 分片文件，请先运行 train.py 生成分片。")
        sys.exit(1)
    shard_ids = parse_shards(args.shards, n_available)
    pattern = TEST_SHARD_PATTERN if args.set == 'test' else TRAIN_SHARD_PATTERN
    missing = [sid for sid in shard_ids if not os.path.exists(pattern.format(sid))]
    if missing:
        print(f"⚠️ 以下分片不存在: {missing}")
        shard_ids = [sid for sid in shard_ids if sid not in missing]
        if not shard_ids:
            print("没有有效分片，退出。")
            sys.exit(1)
    print(f"评估分片 ({args.set}): {shard_ids}")

    # 推理
    y_true, y_score, per_shard = run_inference(model, shard_ids, device, args.set)

    # 汇总指标
    auc = compute_auc(y_true, y_score)
    m = compute_metrics(y_true, y_score, args.threshold)
    print("=" * 64)
    print("评估结果汇总:")
    print(f"  样本总数: {len(y_true):,} (正:{m['n_pos']:,} 负:{m['n_neg']:,})")
    print(f"  AUC = {auc:.4f}")
    print(f"  Threshold = {m['threshold']:.2f}")
    print(f"  Accuracy = {m['acc']:.4f} | Precision = {m['precision']:.4f} | "
          f"Recall = {m['recall']:.4f} | F1 = {m['f1']:.4f} | F0.5 = {m['f0_5']:.4f}")
    print(f"  TP={m['tp']:,} FP={m['fp']:,} FN={m['fn']:,} TN={m['tn']:,}")

    # 多阈值扫描
    if args.thresholds:
        print("\n多阈值扫描:")
        print(f"  {'Thr':>5} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | {'F0.5':>7} | {'Acc':>7}")
        print("  " + "-" * 50)
        for thr in sorted(float(x) for x in args.thresholds.split(',')):
            mm = compute_metrics(y_true, y_score, thr)
            print(f"  {thr:5.2f} | {mm['precision']:7.4f} | {mm['recall']:7.4f} | {mm['f1']:7.4f} | "
                  f"{mm['f0_5']:7.4f} | {mm['acc']:7.4f}")
    print("=" * 64)

    # 保存预测
    if args.save_preds:
        np.savez(args.save_preds, label=y_true, prob=y_score,
                 shard_ids=np.asarray(shard_ids), set_name=args.set)
        print(f"✅ 预测已保存: {os.path.abspath(args.save_preds)}")


if __name__ == "__main__":
    main()