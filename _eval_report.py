# _eval_report.py — 用已保存的最佳模型生成完整评估报告
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import config
from models import NTAM
from data_utils import load_test_shard, get_num_test_shards
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------- 加载最佳模型 ----------
ckpt = torch.load(config.MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
meta = ckpt["metadata"]
print("=" * 72)
print("加载最佳模型")
print("  结构: feat_dim={feat_dim} seq_len={seq_len} max_neighbors={max_neighbors} "
      "layers={num_layers} heads={nhead} use_neighborhood={use_neighborhood}".format(**meta))
print("  训练信息: best_epoch={best_epoch} F1={test_f1:.4f} (P={test_prec:.4f} R={test_rec:.4f}) "
      "neg_ratio={neg_ratio} batch={batch_size} lr={learning_rate}".format(**meta))

model = NTAM(feat_dim=meta["feat_dim"], seq_len=meta["seq_len"], max_neighbors=meta["max_neighbors"],
             num_layers=meta["num_layers"], nhead=meta["nhead"], dropout=meta["dropout"],
             use_neighborhood=meta["use_neighborhood"]).to(config.DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------- 收集所有测试样本概率 ----------
probs_all, labels_all, shard_ids = [], [], []
with torch.no_grad():
    for s in range(get_num_test_shards()):
        loader = load_test_shard(s)
        for sf, nf, nm, lb in loader:
            sf = sf.to(config.DEVICE); nf = nf.to(config.DEVICE)
            nm = nm.to(config.DEVICE); lb = lb.to(config.DEVICE)
            prob, _ = model(sf, nf, nm)
            b = len(lb)
            probs_all.append(prob.detach().cpu().numpy().ravel())
            labels_all.append(lb.detach().cpu().numpy().ravel())
            shard_ids.append(np.full(b, s))
probs = np.concatenate(probs_all); labels = np.concatenate(labels_all)
shard_ids = np.concatenate(shard_ids)
print(f"  测试样本总计: {len(probs):,} | 正:{int((labels==1).sum()):,} 负:{int((labels==0).sum()):,}")

# ---------- 报告工具 ----------
def metrics(p, y, t):
    pr = p > t
    tp = float(((pr == 1) & (y == 1)).sum()); fp = float(((pr == 1) & (y == 0)).sum())
    fn = float(((pr == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, tp, fp, fn

def report(name, p, y):
    npos = int((y == 1).sum())
    print("\n" + "=" * 72)
    print(f"【{name}】 样本 {len(y):,} | 正:{npos:,} 负:{int((y==0).sum()):,} | 比例 1:{int((y==0).sum()/max(npos,1))}")
    print(f"  ROC-AUC = {roc_auc_score(y, p):.4f} | PR-AUC = {average_precision_score(y, p):.4f}")
    print(f"  {'阈值':>6} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    for t in config.THRESHOLDS:
        prec, rec, f1, tp, fp, fn = metrics(p, y, t)
        print(f"  {t:6.2f} {prec:8.4f} {rec:8.4f} {f1:8.4f} {int(tp):6d} {int(fp):6d} {int(fn):6d}")

def dist(name, p, y):
    pp, pn = p[y == 1], p[y == 0]
    print(f"\n【{name}】概率分布分位数")
    print(f"  正样本: {np.round(np.percentile(pp, [10,25,50,75,90,95,99]),3)}")
    print(f"  负样本: {np.round(np.percentile(pn, [50,90,95,99,99.5,99.9,99.95]),4)}")

# ---------- 全测试集 ----------
report("全测试集 (30分片, 1:832)", probs, labels)
dist("全测试集", probs, labels)

# ---------- 按时间段（测试分片排序=时间排序） ----------
mask_first = shard_ids < 10
mask_mid = (shard_ids >= 10) & (shard_ids < 20)
mask_last = shard_ids >= 20
report("前10分片 (旧口径, 11月中-12月初)", probs[mask_first], labels[mask_first])
report("中10分片 (12月上旬-中旬)", probs[mask_mid], labels[mask_mid])
report("后10分片 (12月中-月底, 故障最稀疏)", probs[mask_last], labels[mask_last])

# ---------- 按正负样本判对能力 ----------
print("\n" + "=" * 72)
print("【判别力快照】最佳阈值=0.8 时的判对情况")
for name, m, p, y in [("前10分片", mask_first, probs[mask_first], labels[mask_first]),
                       ("中10分片", mask_mid, probs[mask_mid], labels[mask_mid]),
                       ("后10分片", mask_last, probs[mask_last], labels[mask_last])]:
    prec, rec, f1, tp, fp, fn = metrics(p, y, 0.8)
    print(f"  {name}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} TP={int(tp)} FP={int(fp)} FN={int(fn)}")
print("=" * 72)
