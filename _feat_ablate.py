# _feat_ablate.py - 剔除弱特征快速验证
# 对比: 全30维 vs 剔除候选弱特征(9个) 的 LR 在测试分片0上的表现
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from data_utils import load_train_shard, load_test_shard
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

WEAK_IDX = [22, 23, 24, 25, 26, 27, 28, 29]  # 187,197,184,177,206,182,181,232,233
print("剔除的弱特征索引:", WEAK_IDX, "(对应 Gini<0.015 的 9 列)")

print("[load train (4 shards, 1:10, last-day)]")
X_tr, y_tr = [], []
for s in range(4):
    loader = load_train_shard(s)
    for sf, nf, nm, lb in loader:
        sflat = sf.cpu().numpy().reshape(len(lb), -1)
        X_tr.append(sflat[:, -30:]); y_tr.append(lb.cpu().numpy().ravel())
X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
pos = y_tr == 1; neg_idx = np.where(~pos)[0]; npos = int(pos.sum())
sel = np.random.RandomState(0).choice(neg_idx, npos * 10, replace=False)
idx = np.concatenate([np.where(pos)[0], sel]); X_tr, y_tr = X_tr[idx], y_tr[idx]

print("[load test shard 0]")
X_te, y_te = [], []
loader = load_test_shard(0)
for sf, nf, nm, lb in loader:
    sflat = sf.cpu().numpy().reshape(len(lb), -1)
    X_te.append(sflat[:, -30:]); y_te.append(lb.cpu().numpy().ravel())
X_te = np.concatenate(X_te); y_te = np.concatenate(y_te)

def eval_lr(Xtr, ytr, Xte, yte, tag):
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    p = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    best = (0, 0, 0, 0)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pr = p > t
        tp = float(((pr == 1) & (yte == 1)).sum()); fp = float(((pr == 1) & (yte == 0)).sum())
        fn = float(((pr == 0) & (yte == 1)).sum())
        prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best[0]: best = (f1, prec, rec, t)
    print(f"  {tag}: ROC-AUC={auc:.4f} | 最佳F1={best[0]:.4f} (P={best[1]:.3f} R={best[2]:.3f} @t={best[3]})")

print()
keep = [i for i in range(30) if i not in WEAK_IDX]
print(f"全量特征数: 30  vs  剔除后: {len(keep)}")
eval_lr(X_tr, y_tr, X_te, y_te, "全 30 维 LR")
eval_lr(X_tr[:, keep], y_tr, X_te[:, keep], y_te, f"剔除 {len(WEAK_IDX)} 弱特征 LR")
print("[done]")
