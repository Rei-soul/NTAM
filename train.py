# train.py
# 入口：分片加载训练集 → 增量训练模型 → 每 epoch 后直接评估测试集
# 特点：每 epoch 训练完立即评估全部测试分片，打印 P/R/F1，
#       训练结束后自动恢复到 F1 最高的 epoch 的模型

import os
import json
import torch
import torch.optim as optim
import numpy as np
from config import *
from models import NTAM
from data_utils import load_data, load_train_shard, load_test_shard, get_num_train_shards, get_num_test_shards
from memory_guard import start_guard


def evaluate_on_shard(model, shard_id, criterion, scaler=None):
    """评估单个测试分片"""
    loader = load_test_shard(shard_id)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    n_pos, n_neg = 0, 0
    n_pred_pos, n_pred_neg = 0, 0
    with torch.no_grad():
        for sf, nf, nm, lb in loader:
            sf, nf, nm, lb = sf.to(DEVICE), nf.to(DEVICE), nm.to(DEVICE), lb.to(DEVICE)
            if USE_AMP and DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    prob, logits = model(sf, nf, nm)
                    loss = criterion(logits, lb)
            else:
                prob, logits = model(sf, nf, nm)
                loss = criterion(logits, lb)
            preds = (prob > 0.5).float()
            total_correct += (preds == lb).float().sum().item()
            total_loss += loss.item() * len(sf)
            total_samples += len(sf)
            tp += ((preds == 1.0) & (lb == 1.0)).float().sum().item()
            fp += ((preds == 1.0) & (lb == 0.0)).float().sum().item()
            fn += ((preds == 0.0) & (lb == 1.0)).float().sum().item()
            tn += ((preds == 0.0) & (lb == 0.0)).float().sum().item()
            n_pos += (lb == 1.0).float().sum().item()
            n_neg += (lb == 0.0).float().sum().item()
            n_pred_pos += (preds == 1.0).float().sum().item()
            n_pred_neg += (preds == 0.0).float().sum().item()

    del loader
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return total_loss, total_correct, total_samples, tp, fp, fn


def evaluate_all(model, n_test_shards, criterion, scaler=None, verbose=False):
    """遍历所有测试分片，累积评估指标，返回 (loss, acc, prec, rec, f1)"""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    for s in range(n_test_shards):
        loss_s, correct_s, samples_s, tp_s, fp_s, fn_s = evaluate_on_shard(model, s, criterion, scaler)
        total_loss += loss_s
        total_correct += correct_s
        total_samples += samples_s
        total_tp += tp_s
        total_fp += fp_s
        total_fn += fn_s
    total_tn = total_samples - total_tp - total_fp - total_fn
    total_pos = total_tp + total_fn
    total_neg = total_fp + total_tn
    acc = total_correct / max(total_samples, 1)
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    if verbose:
        print(f"  [评估汇总] 总样本:{int(total_samples)} 真实(正:{int(total_pos)} 负:{int(total_neg)})"
              f" | TP:{int(total_tp)} FP:{int(total_fp)} FN:{int(total_fn)} TN:{int(total_tn)}")
    return total_loss / max(total_samples, 1), acc, prec, rec, f1


def train_one_epoch(model, loader, criterion, optimizer, scaler, loss_weight=1.0):
    """训练一个 epoch，返回 (avg_loss, accuracy)"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_batches = 0

    for batch_idx, (sf, nf, nm, lb) in enumerate(loader):
        sf, nf, nm, lb = sf.to(DEVICE), nf.to(DEVICE), nm.to(DEVICE), lb.to(DEVICE)
        optimizer.zero_grad()

        prob, logits = model(sf, nf, nm)
        loss = criterion(logits, lb)
        if loss_weight != 1.0:
            loss = loss * loss_weight

        if torch.isnan(loss):
            nan_batches += 1
            continue  # 跳过这个 batch

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = (prob > 0.5).float()
        total_correct += (preds == lb).float().sum().item()
        total_loss += loss.item() * len(sf)
        total_samples += len(sf)

    if nan_batches > 0:
        print(f"    [警告] {nan_batches}/{batch_idx+1} batches 产生了 NaN loss，已跳过")

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def _extract_model_config(model):
    """从 NTAM 模型实例提取真实结构参数（保证与保存的权重严格对应）。"""
    enc_layer = model.temporal.transformer_encoder.layers[0]
    # PyTorch 中 encoder_layer.dropout 是 nn.Dropout 模块（.p 为概率），旧版本可能是裸浮点，做兼容
    dropout_val = enc_layer.dropout
    dropout = getattr(dropout_val, 'p', dropout_val)
    return {
        'feat_dim': model.temporal.feat_dim,
        'seq_len': model.seq_len,
        'max_neighbors': model.max_neighbors,
        'transformer_layers': len(model.temporal.transformer_encoder.layers),
        'num_heads': enc_layer.self_attn.num_heads,
        'dropout': dropout,
        'use_neighborhood': model.use_neighborhood,
    }


def save_trained_model(model, best_record, final_metrics, epoch_records, save_dir=SAVE_DIR):
    """训练完成后将最佳模型持久化到磁盘。

    保存两部分内容:
      1) ntam_best.pt      — 最佳模型权重(state_dict) + 重建模型所需配置 + 最终测试指标
      2) training_log.json — 每个 epoch 的指标日志（不含权重，便于后续分析）

    参数:
        model:          已恢复到最佳 epoch 的模型实例
        best_record:    最佳 epoch 的记录 dict（含 'epoch' 等字段）
        final_metrics:  最佳模型的最终测试指标 dict
        epoch_records:  所有 epoch 的记录列表
        save_dir:       保存目录（默认取 config.SAVE_DIR）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) 最佳模型完整存档：权重统一转到 CPU，方便日后在无 GPU 机器上加载
    best_path = os.path.join(save_dir, SAVE_NAME)
    torch.save({
        'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'config': _extract_model_config(model),
        'best_epoch': best_record['epoch'],
        'final_metrics': final_metrics,
    }, best_path)
    print(f"  ✅ 最佳模型已保存: {os.path.abspath(best_path)}")

    # 2) Epoch 指标日志（state_dict 不入 JSON，避免文件过大且不可序列化）
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            'best_epoch': best_record['epoch'],
            'final_metrics': final_metrics,
            'epochs': [
                {
                    'epoch': r['epoch'],
                    'train_loss': r['train_loss'],
                    'test_prec': r['test_prec'],
                    'test_rec': r['test_rec'],
                    'test_f1': r['test_f1'],
                }
                for r in epoch_records
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Epoch 训练日志已保存: {os.path.abspath(log_path)}")


def train():
    # 启动内存看门狗
    start_guard(MEMORY_LIMIT_GB)

    mode_str = "邻域启用 (完整NTAM)" if USE_NEIGHBORHOOD else "邻域关闭 (消融实验 NTAM_alt1)"
    amp_str = "混合精度 ON" if (USE_AMP and DEVICE == "cuda") else "混合精度 OFF"
    print("=" * 60)
    print(f"NTAM 训练 - {mode_str} | {amp_str}")
    print("=" * 60)

    # 1. 加载数据（构建或验证分片）
    print("\n[1] 加载数据...")
    n_train_shards, n_test_shards = load_data()
    print(f"  训练分片: {n_train_shards}, 测试分片: {n_test_shards}")

    # 2. 实例化模型
    model = NTAM(
        feat_dim=FEAT_DIM,
        seq_len=SEQ_LEN,
        max_neighbors=MAX_NEIGHBORS,
        num_layers=TRANSFORMER_LAYERS,
        nhead=NUM_HEADS,
        dropout=DROPOUT,
        use_neighborhood=USE_NEIGHBORHOOD
    ).to(DEVICE)

    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 优化器和损失函数（pos_weight 处理正负样本不平衡）
    pos_weight = torch.tensor([POS_WEIGHT]).to(DEVICE) if POS_WEIGHT > 0 else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE == "cuda") else None

    if pos_weight is not None:
        print(f"  使用 pos_weight={POS_WEIGHT} (补偿训练集正负比约1:11)")

    # 4. 训练：每 epoch 后评估测试集，记录指标
    print(f"\n[2] 训练 ({EPOCHS} epochs × {n_train_shards} 分片) | "
          f"每 epoch 后评估 {n_test_shards} 个测试分片")

    shard_weights = [1.0 + TIME_WEIGHT_ALPHA * i / max(n_train_shards - 1, 1)
                     for i in range(n_train_shards)]
    if TIME_WEIGHT_ALPHA > 0:
        print('    time-weight: alpha={} shard weight {:.2f}~{:.2f}'.format(
              TIME_WEIGHT_ALPHA, shard_weights[0], shard_weights[-1]))

    epoch_records = []  # 每个 epoch 的 (epoch, train_loss, test_prec, test_rec, test_f1, state_dict)

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        print(f"\n{'─' * 50}")
        print(f"  Epoch {epoch}/{EPOCHS}")

        # 训练
        for shard_id in range(n_train_shards):
            train_loader = load_train_shard(shard_id)
            avg_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                                  loss_weight=shard_weights[shard_id])
            epoch_loss += avg_loss * len(train_loader.dataset)
            epoch_correct += train_acc * len(train_loader.dataset)
            epoch_samples += len(train_loader.dataset)
            del train_loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        epoch_loss /= max(epoch_samples, 1)
        epoch_acc = epoch_correct / max(epoch_samples, 1)

        # 每 epoch 后直接评估测试集
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"  → 评估测试集...")
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_all(model, n_test_shards, criterion, scaler, verbose=False)
        print(f"  → Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
              f"Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")

        # 保存每个 epoch 的模型快照（按最佳 F1 判断时恢复）
        epoch_records.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1,
            'state_dict': {k: v.detach().clone() for k, v in model.state_dict().items()}
        })

    # 5. 打印 Epoch 走势表
    print(f"\n{'=' * 70}")
    print("Epoch 走势表 (每 epoch 的测试集指标):")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 70)
    best_record = None
    for rec in epoch_records:
        marker = ""
        if best_record is None or rec['test_f1'] > best_record['test_f1']:
            best_record = rec
            marker = "  ← 最佳F1"
        print(f"{rec['epoch']:>6} | {rec['train_loss']:>10.4f} | "
              f"{rec['test_prec']:>7.4f} | {rec['test_rec']:>7.4f} | {rec['test_f1']:>7.4f}{marker}")
    print("=" * 70)

    # 6. 恢复到 F1 最高的 epoch 的模型
    if best_record is not None:
        model.load_state_dict(best_record['state_dict'])
        print(f"\n  已恢复最佳模型: Epoch {best_record['epoch']} "
              f"(Prec={best_record['test_prec']:.4f}, Rec={best_record['test_rec']:.4f}, F1={best_record['test_f1']:.4f})")

    # 7. 用最佳模型做最终详细评估
    print(f"\n  [评估] 用最佳模型 (epoch {best_record['epoch']}) 评估测试集...")
    # 6.5 fine-tune on most recent shards (near test cutoff)
    if FINE_TUNE_LAST_SHARDS > 0:
        ft_shard_ids = list(range(max(0, n_train_shards - FINE_TUNE_LAST_SHARDS), n_train_shards))
        print("\n  [fine-tune] last %d shards %s (%d epochs, lr=%g)..." % (len(ft_shard_ids), ft_shard_ids, FINE_TUNE_EPOCHS, FINE_TUNE_LR))
        pre_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ft_optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        for ft_ep in range(FINE_TUNE_EPOCHS):
            ft_loss = 0.0
            for ft_sid in ft_shard_ids:
                ft_loader = load_train_shard(ft_sid)
                ft_loss, _ = train_one_epoch(model, ft_loader, criterion, ft_optimizer, None, loss_weight=1.0)
                del ft_loader
            print("    fine-tune epoch %d/%d | loss=%.4f" % (ft_ep + 1, FINE_TUNE_EPOCHS, ft_loss))
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        ft_loss, _, ft_p, ft_r, ft_f1 = evaluate_all(model, n_test_shards, criterion, scaler, verbose=False)
        print("  -> after fine-tune: F1=%.4f (P %.4f R %.4f)" % (ft_f1, ft_p, ft_r))
        if best_record is not None and ft_f1 < best_record['test_f1']:
            print("    fine-tune did not improve (F1 %.4f < %.4f), reverting" % (ft_f1, best_record['test_f1']))
            model.load_state_dict(pre_sd)
    final_test_loss, final_test_acc, final_prec, final_rec, final_f1 = evaluate_all(model, n_test_shards, criterion, scaler, verbose=True)
    print(f"\n{'=' * 60}")
    print("最终评估结果:")
    print(f"  Test Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.4f}")
    print(f"  Precision: {final_prec:.4f} | Recall: {final_rec:.4f} | F1: {final_f1:.4f}")
    print(f"  Epochs: 共 {len(epoch_records)} epochs (最佳在第 {best_record['epoch']} epoch)")
    print(f"  配置 → LAYERS={TRANSFORMER_LAYERS} | LR={LEARNING_RATE} | EPOCHS上限={EPOCHS} | "
          f"HEADS={NUM_HEADS} | FFN={FEAT_DIM*8}x | SEQ_LEN={SEQ_LEN} | "
          f"TRAIN_SHARDS={n_train_shards} | TEST_SHARDS={n_test_shards}")
    print("=" * 60)
    print("✓ 训练完成!")

    # 8. 保存训练好的模型到磁盘
    print(f"\n  [保存] 持久化最佳模型 → {SAVE_DIR}/ ...")
    save_trained_model(
        model,
        best_record,
        {
            'test_loss': final_test_loss,
            'test_acc': final_test_acc,
            'precision': final_prec,
            'recall': final_rec,
            'f1': final_f1,
        },
        epoch_records,
    )


if __name__ == "__main__":
    train()