import json, os, random
import numpy as np
import torch
import torch.optim as optim
from config import *
from models import NTAM
from data_utils import (load_data, load_train_shard, load_val_shard, load_test_shard,
                        get_num_train_shards, get_num_val_shards, get_num_test_shards)
from memory_guard import start_guard

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _auc(y, p):
    y=np.asarray(y).astype(np.int64); p=np.asarray(p); pos=p[y==1]; neg=p[y==0]
    if not len(pos) or not len(neg): return float('nan')
    order=np.argsort(p, kind='mergesort'); ranks=np.empty(len(p),float); ranks[order]=np.arange(1,len(p)+1)
    return float((ranks[y==1].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def _ap(y,p):
    y=np.asarray(y).astype(np.int64); p=np.asarray(p); n=int(y.sum())
    if n==0:return float('nan')
    order=np.argsort(-p,kind='mergesort'); ys=y[order]; ranks=np.arange(1,len(y)+1); return float((np.cumsum(ys)[ys==1] / ranks[ys==1]).sum()/n)

def _metrics(y,p,threshold=.5):
    y=np.asarray(y).astype(np.int64); pred=(np.asarray(p)>=threshold).astype(np.int64); tp=int(((pred==1)&(y==1)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-12)
    return {'precision':prec,'recall':rec,'f1':f1,'accuracy':(tp+tn)/max(len(y),1),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'roc_auc':_auc(y,p),'pr_auc':_ap(y,p)}

def _collect(model, split, count, criterion=None):
    loader_fn={'val':load_val_shard,'test':load_test_shard,'train':load_train_shard}[split]; probs=[]; labels=[]; total_loss=0.; total=0; model.eval()
    with torch.no_grad():
        for sid in range(count):
            loader=loader_fn(sid)
            for sf,nf,nm,lb in loader:
                sf,nf,nm,lb=sf.to(DEVICE),nf.to(DEVICE),nm.to(DEVICE),lb.to(DEVICE); prob,logits=model(sf,nf,nm); probs.append(prob.detach().cpu().numpy().reshape(-1)); labels.append(lb.detach().cpu().numpy().reshape(-1));
                if criterion is not None: total_loss += criterion(logits,lb).item()*len(sf); total += len(sf)
    return np.concatenate(labels) if labels else np.empty(0),np.concatenate(probs) if probs else np.empty(0),total_loss/max(total,1)

def evaluate(model, split, count, criterion=None, threshold=.5):
    y,p,loss=_collect(model,split,count,criterion); m=_metrics(y,p,threshold); m['loss']=loss; return m,y,p

def train_one_epoch(model, shard_count, criterion, optimizer):
    model.train(); total_loss=0.; total=0
    for sid in range(shard_count):
        loader=load_train_shard(sid)
        for sf,nf,nm,lb in loader:
            sf,nf,nm,lb=sf.to(DEVICE),nf.to(DEVICE),nm.to(DEVICE),lb.to(DEVICE); optimizer.zero_grad(set_to_none=True); _,logits=model(sf,nf,nm); loss=criterion(logits,lb)
            if not torch.isfinite(loss): raise FloatingPointError(f'non-finite loss in train shard {sid}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total_loss+=loss.item()*len(sf); total+=len(sf)
    return total_loss/max(total,1)

def hard_negative_epoch(model, shard_count, criterion, optimizer):
    """One pass over the highest-scoring negative training examples per shard."""
    if not HARD_NEGATIVE_MINING: return 0.
    model.eval(); total_loss=0.; total=0
    for sid in range(shard_count):
        d=np.load(__import__('data_utils').TRAIN_SHARD_PATTERN.format(sid)); y=np.asarray(d['l']).reshape(-1); neg=np.flatnonzero(y==0)
        if not len(neg): continue
        import data_utils
        loader=data_utils._loader(data_utils.TRAIN_SHARD_PATTERN.format(sid),False,BATCH_SIZE); scores=[]
        with torch.no_grad():
            for sf,nf,nm,lb in loader:
                sf,nf,nm=sf.to(DEVICE),nf.to(DEVICE),nm.to(DEVICE); scores.extend(model(sf,nf,nm)[0].cpu().numpy().reshape(-1))
        hard=neg[np.argsort(np.asarray(scores)[neg])[-min(len(neg),max(1,len(neg)//10)):]]
        loader=load_train_shard(sid,hard); model.train()
        for sf,nf,nm,lb in loader:
            sf,nf,nm,lb=sf.to(DEVICE),nf.to(DEVICE),nm.to(DEVICE),lb.to(DEVICE); optimizer.zero_grad(set_to_none=True); _,logits=model(sf,nf,nm); loss=criterion(logits,lb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total_loss+=loss.item()*len(sf); total+=len(sf)
    return total_loss/max(total,1)

def _best_threshold(y,p):
    thresholds=np.unique(np.concatenate([np.linspace(.01,.99,99),p]))
    best=(0.,.5)
    for t in thresholds:
        f=_metrics(y,p,float(t))['f1']
        if f>best[0]: best=(f,float(t))
    return best[1]

def train():
    seed_all(); start_guard(MEMORY_LIMIT_GB); ntrain,nval,ntest=load_data(); criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT],device=DEVICE)); model=NTAM(INPUT_FEAT_DIM,SEQ_LEN,MAX_NEIGHBORS,TRANSFORMER_LAYERS,NUM_HEADS,DROPOUT,USE_NEIGHBORHOOD,MODEL_DIM).to(DEVICE); opt=optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4); best_state=None; best_val=-1.; best_epoch=0; best_thr=.5; stale=0; records=[]
    total_steps=max(1,EPOCHS*ntrain); step=0
    for epoch in range(1,EPOCHS+1):
        loss=train_one_epoch(model,ntrain,criterion,opt); step += ntrain; frac=min(1.,step/max(WARMUP_EPOCHS*ntrain,1)); lr=LEARNING_RATE*(frac if epoch<=WARMUP_EPOCHS else .5*(1+np.cos(np.pi*(epoch-WARMUP_EPOCHS)/max(EPOCHS-WARMUP_EPOCHS,1))));
        for g in opt.param_groups:g['lr']=lr
        if epoch==EPOCHS//2 and HARD_NEGATIVE_MINING: hard_negative_epoch(model,ntrain,criterion,opt)
        vm,vy,vp=evaluate(model,'val',nval,criterion,.5); thr=_best_threshold(vy,vp); vm=_metrics(vy,vp,thr); vm['loss']=evaluate(model,'val',nval,criterion,thr)[0]['loss']; records.append({'epoch':epoch,'train_loss':loss,'val':vm,'threshold':thr}); print(f'Epoch {epoch}: train_loss={loss:.5f} val_f1={vm["f1"]:.4f} val_p={vm["precision"]:.4f} val_r={vm["recall"]:.4f} thr={thr:.3f}')
        if vm['f1']>best_val: best_val=vm['f1']; best_epoch=epoch; best_thr=thr; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=PATIENCE: break
    if best_state is None: raise RuntimeError('no valid checkpoint')
    model.load_state_dict(best_state); tm,_,_=evaluate(model,'test',ntest,criterion,best_thr); print(f'Final test: F1={tm["f1"]:.4f} P={tm["precision"]:.4f} R={tm["recall"]:.4f} PR-AUC={tm["pr_auc"]:.4f} ROC-AUC={tm["roc_auc"]:.4f} threshold={best_thr:.3f}')
    os.makedirs(SAVE_DIR,exist_ok=True); torch.save({'model_state_dict':model.state_dict(),'config':{'input_dim':INPUT_FEAT_DIM,'model_dim':MODEL_DIM,'seq_len':SEQ_LEN,'max_neighbors':MAX_NEIGHBORS,'layers':TRANSFORMER_LAYERS,'heads':NUM_HEADS,'dropout':DROPOUT,'use_neighborhood':USE_NEIGHBORHOOD},'best_epoch':best_epoch,'threshold':best_thr,'test_metrics':tm},os.path.join(SAVE_DIR,'ntam_best.pt'))
    with open(os.path.join(SAVE_DIR,'training_log.json'),'w',encoding='utf-8') as f: json.dump({'best_epoch':best_epoch,'threshold':best_thr,'test_metrics':tm,'epochs':records},f,indent=2,ensure_ascii=False)

if __name__=='__main__': train()
