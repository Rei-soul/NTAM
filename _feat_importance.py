# _feat_importance.py - 30维SMART特征重要性分析（借鉴 WEFR 论文思想）
# 方法1: RandomForest Gini importance（训练数据, 1:10降采样, 最后一天特征）
# 方法2: Permutation importance（测试分片0, 用RF, 打乱后ROC-AUC下降量）
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import config
from data_utils import load_train_shard, load_test_shard
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

SMART_IDS = [5, 9, 12, 170, 171, 172, 173, 174, 175, 177,
             180, 181, 182, 183, 184, 187, 188, 190, 192, 194,
             195, 196, 197, 198, 199, 206, 232, 233, 241, 242]
SMART_NAMES = {
    5: 'Reallocated Sectors', 9: 'Power-On Hours', 12: 'Power Cycle',
    170: 'Avail Reserved Space', 171: 'Prog Fail Count', 172: 'Erase Fail Count',
    173: 'Wear Leveling', 174: 'Unexpected Power Loss', 175: 'PowerLossProt Fail',
    177: 'Wear Range Delta', 180: 'Unused Reserved Blk', 181: 'ProgFail Total',
    182: 'EraseFail Total', 183: 'Downshift Error', 184: 'End-to-End Error',
    187: 'Uncorrectable Errors', 188: 'Command Timeout', 190: 'Airflow Temp',
    192: 'PowerOff Retract', 194: 'Temperature', 195: 'HW ECC Recovered',
    196: 'Realloc Event', 197: 'Pending Sectors', 198: 'Offline Uncorr',
    199: 'UDMA CRC Error', 206: 'Flying Height', 232: 'Endurance Remain',
    233: 'Media Wearout', 241: 'LBAs Written', 242: 'LBAs Read',
}

print("=" * 78)
print("[1] build train data (last-day features, 4 train shards, 1:10 downsample)")
X_tr, y_tr = [], []
for s in range(4):
    loader = load_train_shard(s)
    for sf, nf, nm, lb in loader:
        sflat = sf.cpu().numpy().reshape(len(lb), -1)
        X_tr.append(sflat[:, -30:]); y_tr.append(lb.cpu().numpy().ravel())
X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
pos = y_tr == 1; neg_idx = np.where(~pos)[0]; npos = int(pos.sum())
sel = np.random.RandomState(0).choice(neg_idx, npos * 10, replace=False)
idx = np.concatenate([np.where(pos)[0], sel]); X = X_tr[idx]; y = y_tr[idx]
print(f"    train: {len(idx)} samples (pos {npos} + neg {len(sel)})")

print("[2] build test data (test shard 0, last-day features)")
X_te, y_te = [], []
loader = load_test_shard(0)
for sf, nf, nm, lb in loader:
    sflat = sf.cpu().numpy().reshape(len(lb), -1)
    X_te.append(sflat[:, -30:]); y_te.append(lb.cpu().numpy().ravel())
X_te = np.concatenate(X_te); y_te = np.concatenate(y_te)
print(f"    test: {len(y_te)} samples (pos {int((y_te==1).sum())})")

print("[3] train RandomForest (100 trees, depth 13)")
rf = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=0, n_jobs=-1).fit(X, y)
gini = rf.feature_importances_

print("[4] permutation importance on test (n_repeats=3, scoring=roc_auc)")
perm = permutation_importance(rf, X_te, y_te, scoring='roc_auc', n_repeats=3,
                              random_state=0, n_jobs=-1)
perm_mean = perm.importances_mean

print()
print("rank | SMART | feature name                  | GiniImp | PermImp(test) | verdict")
print("-" * 98)
order = np.argsort(gini)[::-1]
for r, di in enumerate(order):
    sid = SMART_IDS[di]
    g = gini[di]; p = perm_mean[di]
    # 判定: 两个指标都低 => 弱特征
    if g < np.percentile(gini, 30) and p < np.percentile(perm_mean, 30):
        verdict = "WEAK (candidate drop)"
    elif g < np.percentile(gini, 30):
        verdict = "weak-gini"
    elif p < np.percentile(perm_mean, 30):
        verdict = "weak-perm"
    else:
        verdict = "strong"
    print(f"{r+1:>4} | {sid:>4} | {SMART_NAMES.get(sid,'?'):<25} | {g:7.4f} | {p:+8.4f} | {verdict}")

print("-" * 98)
print("注: GiniImp=训练集RF基尼重要性(越大越重要); PermImp=测试集打乱该特征后ROC-AUC下降量(负得越多越重要)")
print("[done]")
