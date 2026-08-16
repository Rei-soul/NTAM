# _compare_AB.py - 受控对比：A=降采样1:10 vs B=不降采样
# 两套评估口径：全量30分片(1:832) + 前10分片(旧口径1:420)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import config
from models import NTAM
from data_utils import load_test_shard, get_num_test_shards
from sklearn.metrics import roc_auc_score, average_precision_score

MODEL_A = "datasets/processed/best_model.pt"           # 降采样1:10
MODEL_B = "datasets/processed/best_model_nodown.pt"    # 不降采样

def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt["metadata"]
    m = NTAM(feat_dim=meta["feat_dim"], seq_len=meta["seq_len"], max_neighbors=meta["max_neighbors"],
             num_layers=meta["num_layers"], nhead=meta["nhead"], dropout=meta["dropout"],
             use_neighborhood=meta["use_neighborhood"]).to(config.DEVICE)
    m.load_state_dict(ckpt["model_state_dict"]); m.eval()
    return m, meta

def collect(model, shard_range):
    ps, ls = [], []
    with torch.no_grad():
        for s in shard_range:
            loader = load_test_shard(s)
            for sf, nf, nm, lb in loader:
                p, _ = model(sf.to(config.DEVICE), nf.to(config.DEVICE), nm.to(config.DEVICE))
                ps.append(p.detach().cpu().numpy().ravel())
                ls.append(lb.detach().cpu().numpy().ravel())
    return np.concatenate(ps), np.concatenate(ls)

def tm(p, y, t):
    pr = p > t
    tp = float(((pr == 1) & (y == 1)).sum()); fp = float(((pr == 1) & (y == 0)).sum())
    fn = float(((pr == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1e-9); rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, tp, fp, fn

def block(title, pA, lA, pB, lB):
    npos = int(lA.sum())
    print("\n" + "=" * 74)
    print(f"{title}  (正:{npos} 负:{int((lA == 0).sum())}, 1:{int((lA==0).sum())//max(npos,1)})")
    print(f"{'阈值':>5} | {'A_P':>7} {'A_R':>7} {'A_F1':>7} | {'B_P':>7} {'B_R':>7} {'B_F1':>7}  "
          f"{'A_TP/FP':>9} {'B_TP/FP':>9}")
    for t in config.THRESHOLDS:
        a = tm(pA, lA, t); b = tm(pB, lB, t)
        print(f"{t:5.2f} | {a[0]:7.3f} {a[1]:7.3f} {a[2]:7.3f} | {b[0]:7.3f} {b[1]:7.3f} {b[2]:7.3f}  "
              f"{int(a[3]):>4}/{int(a[4]):>5}  {int(b[3]):>4}/{int(b[4]):>5}")
    print(f"ROC-AUC: A={roc_auc_score(lA, pA):.4f}  B={roc_auc_score(lB, pB):.4f}")
    print(f"PR-AUC : A={average_precision_score(lA, pA):.4f}  B={average_precision_score(lB, pB):.4f}")

mA, metaA = load_model(MODEL_A)
mB, metaB = load_model(MODEL_B)
print(f"A={MODEL_A} (neg_ratio={metaA['neg_ratio']}, best_epoch={metaA['best_epoch']})")
print(f"B={MODEL_B} (neg_ratio={metaB['neg_ratio']}, best_epoch={metaB['best_epoch']})")

n_test = get_num_test_shards()
pA, lA = collect(mA, range(n_test))
pB, lB = collect(mB, range(n_test))
block("全量测试集 (30分片, 含漂移)", pA, lA, pB, lB)
block("前10分片 (旧口径, 无漂移)", pA[:268300], lA[:268300], pB[:268300], lB[:268300])
print("\n[done]")
