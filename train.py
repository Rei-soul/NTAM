# train.py
# 入口：分片加载训练集 → 增量训练模型 → 逐分片评估
# 支持混合精度训练（autocast + GradScaler）减少显存占用

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

    print(f"  [Shard {shard_id}] {total_samples}样本 (真实正:{int(n_pos)} 负:{int(n_neg)})"
          f" → 预测正:{int(n_pred_pos)} 负:{int(n_pred_neg)}"
          f" | TP:{int(tp)} FP:{int(fp)} FN:{int(fn)} TN:{int(tn)}")
    del loader
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return total_loss, total_correct, total_samples, tp, fp, fn


def evaluate_all(model, n_test_shards, criterion, scaler=None):
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
    print(f"  [评估汇总] 总样本:{int(total_samples)} 真实(正:{int(total_pos)} 负:{int(total_neg)})"
          f" | TP:{int(total_tp)} FP:{int(total_fp)} FN:{int(total_fn)} TN:{int(total_tn)}")
    return total_loss / max(total_samples, 1), acc, prec, rec, f1


def train_one_epoch(model, loader, criterion, optimizer, scaler):
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

    # 3. 优化器和损失函数
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler() if (USE_AMP and DEVICE == "cuda") else None

    # 4. 标准 Epoch 训练（外层 epoch，内层遍历所有分片）
    print(f"\n[2] 训练 ({EPOCHS} epochs × {n_train_shards} 分片)")

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        print(f"\n{'─' * 50}")
        print(f"  Epoch {epoch}/{EPOCHS}")

        for shard_id in range(n_train_shards):
            train_loader = load_train_shard(shard_id)
            avg_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            epoch_loss += avg_loss * len(train_loader.dataset)
            epoch_correct += train_acc * len(train_loader.dataset)
            epoch_samples += len(train_loader.dataset)
            del train_loader
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # 每个 epoch 结束后评估一次
        epoch_loss /= max(epoch_samples, 1)
        epoch_acc = epoch_correct / max(epoch_samples, 1)
        test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_all(model, n_test_shards, criterion, scaler)

        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"  Test  Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
              f"Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")

    # 5. 最终评估
    print(f"\n{'=' * 60}")
    print("最终评估...")
    final_test_loss, final_test_acc, final_prec, final_rec, final_f1 = evaluate_all(model, n_test_shards, criterion, scaler)
    print(f"  最终 Test Loss: {final_test_loss:.4f} | Acc: {final_test_acc:.4f}")
    print(f"  Precision: {final_prec:.4f} | Recall: {final_rec:.4f} | F1: {final_f1:.4f}")
    print(f"  总训练 Epochs: {EPOCHS} × {n_train_shards} 分片")
    print("=" * 60)
    print("✓ 训练完成!")


if __name__ == "__main__":
    train()