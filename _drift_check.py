# _drift_check.py - 时间漂移三重验证
# 实验1: 深度模型 vs LR 按时间分组 ROC-AUC（区分"数据漂移"vs"模型泛化失败"；ROC 不受类别不平衡影响）
# 实验2: 健康盘特征分布随时间的漂移（验证特征层面是否真的变了）
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import glob
import config
from models import NTAM
from data_utils import load_test_shard, load_train_shard, get_num_test_shards
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_GROUPS, GROUP_SIZE = 6, 5

print("=" * 76)
print("[load best DL model]")
ckpt = torch.load(config.MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
meta = ckpt["metadata"]
model = NTAM(feat_dim=meta["feat_dim"], seq_len=meta["seq_len"], max_neighbors=meta["max_neighbors"],
             num_layers=meta["num_layers"], nhead=meta["nhead"], dropout=meta["dropout"],
             use_neighborhood=meta["use_neighborhood"]).to(config.DEVICE)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()

print("[train LR baseline: 4 train shards, 1:10 downsample, flattened 240d]")
fs = sorted(glob.glob("datasets/processed/train_shard_*.npz"))[:4]
X_tr, y_tr = [], []
for f in fs:
    d = np.load(f); X_tr.append(d["s"].reshape(len(d["l"]), -1)); y_tr.append(d["l"].ravel())
X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
pos = y_tr == 1; neg_idx = np.where(~pos)[0]; npos = int(pos.sum())
sel = np.random.RandomState(0).choice(neg_idx, npos * 10, replace=False)
idx = np.concatenate([np.where(pos)[0], sel])
lr = LogisticRegression(max_iter=1000).fit(X_tr[idx], y_tr[idx])

print("[collect train-neg last-day features]")
tr_neg = []; cnt = 0
for s in range(4):
    loader = load_train_shard(s)
    for sf, nf, nm, lb in loader:
        sflat = sf.cpu().numpy().reshape(len(lb), -1)
        nmask = lb.cpu().numpy().ravel() == 0
        tr_neg.append(sflat[nmask, -30:]); cnt += int(nmask.sum())
        if cnt > 3000: break
    if cnt > 3000: break
tr_neg = np.concatenate(tr_neg); tr_mean = tr_neg.mean(axis=0)

print("[collect DL & LR probs on full test]")
probs_dl, probs_lr, labels, sids = [], [], [], []
te_neg = [[] for _ in range(N_GROUPS)]
with torch.no_grad():
    for s in range(get_num_test_shards()):
        loader = load_test_shard(s)
        for sf, nf, nm, lb in loader:
            b = len(lb)
            sflat = sf.cpu().numpy().reshape(b, -1)
            lbn = lb.cpu().numpy().ravel()
            p, _ = model(sf.to(config.DEVICE), nf.to(config.DEVICE), nm.to(config.DEVICE))
            probs_dl.append(p.detach().cpu().numpy().ravel())
            probs_lr.append(lr.predict_proba(sflat)[:, 1])
            labels.append(lbn); sids.append(np.full(b, s))
            g = min(s // GROUP_SIZE, N_GROUPS - 1)
            nmask = lbn == 0
            if nmask.sum() > 0:
                te_neg[g].append(sflat[nmask, -30:])
probs_dl = np.concatenate(probs_dl); probs_lr = np.concatenate(probs_lr)
labels = np.concatenate(labels); sids = np.concatenate(sids)
te_neg = [np.concatenate(v) if v else np.zeros((0, 30)) for v in te_neg]

print()
print("[Exp1] ROC-AUC by time group (ROC is imbalance-invariant)")
print(f"{'Grp':>3} {'shards':>8} {'pos':>5} {'neg':>7} {'ratio':>7} {'DL_AUC':>8} {'LR_AUC':>8}")
for g in range(N_GROUPS):
    m = (sids >= g * GROUP_SIZE) & (sids < (g + 1) * GROUP_SIZE)
    npg = int(labels[m].sum()); nng = int((labels[m] == 0).sum())
    dl_auc = roc_auc_score(labels[m], probs_dl[m])
    lr_auc = roc_auc_score(labels[m], probs_lr[m])
    print(f"{g+1:>3} {g*GROUP_SIZE:>4}-{g*GROUP_SIZE+4:>3} {npg:>5} {nng:>7} "
          f"1:{nng//max(npg,1):>6} {dl_auc:>8.3f} {lr_auc:>8.3f}")

print()
print("[Exp2] healthy last-day feature drift (train-neg vs each test group)")
te_means = np.array([te_neg[g].mean(axis=0) for g in range(N_GROUPS)])
diffs = np.abs(te_means - tr_mean).max(axis=0)
print("top-8 drifted dims (train vs max group diff):")
for di in np.argsort(diffs)[::-1][:8]:
    print(f"  dim{di:>3}: train={tr_mean[di]:+.3f} group_means={np.round(te_means[:, di], 3)}")
print("mean abs feature diff per group (avg over 30 dims):")
for g in range(N_GROUPS):
    print(f"  group{g+1}: {np.abs(te_means[g] - tr_mean).mean():.4f}")
print("=" * 76)
