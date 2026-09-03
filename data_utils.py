"""Causal N-only feature extraction and temporal sample generation."""
import gc, json, os
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from config import *

DATA_DIR = "/mnt/newdisk/shujie/dataset/alibaba_ssd"
PROCESSED_DIR = "/mnt/newdisk/qhmiao/datasets/processed"
TARGET_FILE = os.path.join(PROCESSED_DIR, "target_disks.csv")
NEIGHBOR_MAP_FILE = os.path.join(PROCESSED_DIR, "neighbor_map.csv")
TRAIN_SHARD_PATTERN = os.path.join(PROCESSED_DIR, f"{DATASET_VERSION}_train_shard_{{:02d}}.npz")
VAL_SHARD_PATTERN = os.path.join(PROCESSED_DIR, f"{DATASET_VERSION}_val_shard_{{:02d}}.npz")
TEST_SHARD_PATTERN = os.path.join(PROCESSED_DIR, f"{DATASET_VERSION}_test_shard_{{:02d}}.npz")
FEATURE_PATTERN = os.path.join(PROCESSED_DIR, f"{DATASET_VERSION}_feat_day_{{:04d}}.npy")
MANIFEST_PATH = os.path.join(PROCESSED_DIR, f"{DATASET_VERSION}_manifest.json")
N_COLS = [f"n_{sid}" for sid in SMART_IDS]
RNG = np.random.RandomState(SEED)

def _scan_csv_dates():
    fs = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and f.startswith('2018') and DATA_MONTH_START <= int(f[4:6]) <= DATA_MONTH_END)
    dates = sorted(f[:8] for f in fs)
    return dates, {d: os.path.join(DATA_DIR, f'{d}.csv') for d in dates}

def _load_disk_info():
    df = pd.read_csv(TARGET_FILE); df['failure_time'] = pd.to_datetime(df['failure_time']); out = {}
    for _, r in df.iterrows():
        out[str(r['pair_id'])] = {'model': str(r['model']), 'is_failure': bool(r['is_failure']), 'failure_time': r['failure_time'] if pd.notna(r['failure_time']) else None, 'node_id': int(r['node_id']) if pd.notna(r['node_id']) and r['node_id'] >= 0 else None}
    return out

def _load_neighbor_map(disk_info):
    df = pd.read_csv(NEIGHBOR_MAP_FILE); df['pair_id'] = df['pair_id'].astype(str); df['neighbor_pair_id'] = df['neighbor_pair_id'].astype(str)
    out = df.groupby('pair_id')['neighbor_pair_id'].apply(list).to_dict()
    for p in disk_info: out.setdefault(p, [])
    return out

def _get_all_pids(disk_info):
    pids = sorted(disk_info)
    if MAX_DISKS <= 0 or len(pids) <= MAX_DISKS: return pids
    fails = [p for p in pids if disk_info[p]['is_failure']]; healthy = [p for p in pids if not disk_info[p]['is_failure']]; RNG.shuffle(healthy)
    return sorted(fails + healthy[:max(0, MAX_DISKS-len(fails))])

def _reservoir_stats(dates, date_to_file):
    cap=50000; reservoirs=[]
    for d in dates:
        if d > TRAIN_CUTOFF: break
        try: df=pd.read_csv(date_to_file[d],usecols=N_COLS)
        except Exception: continue
        if not reservoirs: reservoirs=[[] for _ in N_COLS]
        for j,c in enumerate(N_COLS):
            vals=pd.to_numeric(df[c],errors='coerce').dropna().to_numpy(np.float32)
            if len(vals)>5000: vals=vals[RNG.choice(len(vals),5000,replace=False)]
            reservoirs[j].append(vals)
        del df
    med=np.zeros(len(N_COLS),np.float32); lo=np.zeros_like(med); hi=np.zeros_like(med)
    for j in range(len(N_COLS)):
        x=np.concatenate(reservoirs[j]) if reservoirs else np.empty(0,np.float32)
        if len(x)>cap: x=x[RNG.choice(len(x),cap,replace=False)]
        if len(x): med[j]=np.median(x); lo[j],hi[j]=np.percentile(x,[.5,99.5])
    return med,lo,hi

def _extract_and_build_feat(disk_info, sampled_pids, neighbor_map):
    dates,date_to_file=_scan_csv_dates(); os.makedirs(PROCESSED_DIR,exist_ok=True); needed=set(sampled_pids)
    for p in sampled_pids: needed.update(neighbor_map.get(p,[])[:MAX_NEIGHBORS])
    extract_pids=sorted(needed); pid_to_idx={p:i for i,p in enumerate(extract_pids)}; n=len(extract_pids)
    med,clip_lo,clip_hi=_reservoir_stats(dates,date_to_file)
    lookup=pd.DataFrame([{'disk_id':int(p.split('_',1)[0]),'model':p.split('_',1)[1],'_idx':i} for i,p in enumerate(extract_pids)])
    lookup['model']=lookup['model'].astype(str); files=[]
    for di in range(len(dates)):
        fp=FEATURE_PATTERN.format(di); init=np.zeros((n,INPUT_FEAT_DIM),np.float32); init[:,:RAW_FEAT_DIM]=med; init[:,RAW_FEAT_DIM:2*RAW_FEAT_DIM]=1.; np.save(fp,init); files.append(fp)
    for di,d in enumerate(dates):
        try:
            df=pd.read_csv(date_to_file[d],usecols=['disk_id','model']+N_COLS); df['disk_id']=df['disk_id'].astype(int); df['model']=df['model'].astype(str); merged=df.merge(lookup,on=['disk_id','model'],how='inner')
            if len(merged):
                vals=merged[N_COLS].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); miss=~np.isfinite(vals); vals[miss]=np.take(med,np.where(miss)[1]); vals=np.clip(vals,clip_lo,clip_hi); arr=np.load(files[di],mmap_mode='r+'); idx=merged['_idx'].to_numpy(np.int64); arr[idx,:RAW_FEAT_DIM]=vals; arr[idx,RAW_FEAT_DIM:2*RAW_FEAT_DIM]=miss.astype(np.float32); arr.flush(); del arr
            del df,merged
        except Exception as exc: print(f'[feature] skip {d}: {exc}')
        if (di+1)%30==0: print(f'[feature] {di+1}/{len(dates)}')
        gc.collect()
    for di in range(len(dates)):
        cur=np.load(files[di],mmap_mode='r+'); vals=np.asarray(cur[:,:RAW_FEAT_DIM]); prev=np.asarray(np.load(files[di-1],mmap_mode='r')[:,:RAW_FEAT_DIM]) if di else vals; delta=vals-prev; start=max(0,di-SLOPE_DAYS+1); hist=np.stack([np.asarray(np.load(files[k],mmap_mode='r')[:,:RAW_FEAT_DIM]) for k in range(start,di+1)]); 
        if len(hist)>1:
            x=np.arange(len(hist),dtype=np.float32); xm=x.mean(); slope=((hist-hist.mean(axis=0))*(x-xm)[:,None,None]).sum(axis=0)/(float(((x-xm)**2).sum()) or 1.)
        else: slope=np.zeros_like(vals)
        bstart=max(0,di-BASELINE_DAYS); base=np.median(np.stack([np.asarray(np.load(files[k],mmap_mode='r')[:,:RAW_FEAT_DIM]) for k in range(bstart,di)]),axis=0) if di>bstart else vals
        cur[:,2*RAW_FEAT_DIM:3*RAW_FEAT_DIM]=np.clip(delta,-DERIVED_CLIP,DERIVED_CLIP); cur[:,3*RAW_FEAT_DIM:4*RAW_FEAT_DIM]=np.clip(slope,-DERIVED_CLIP,DERIVED_CLIP); cur[:,4*RAW_FEAT_DIM:5*RAW_FEAT_DIM]=np.clip(vals-base,-DERIVED_CLIP,DERIVED_CLIP); cur.flush(); del cur
    with open(MANIFEST_PATH,'w',encoding='utf-8') as f: json.dump({'version':DATASET_VERSION,'feature_source':'n_raw','columns':N_COLS,'input_dim':INPUT_FEAT_DIM,'train_cutoff':TRAIN_CUTOFF,'val_cutoff':VAL_CUTOFF,'dates':dates},f,indent=2)
    return dates,extract_pids,pid_to_idx,files

class FeatStore:
    def __init__(self,files,pid_to_idx,max_cache=8): self.files,self.pid_to_idx,self.cache,self.max_cache=files,pid_to_idx,OrderedDict(),max_cache
    def get(self,pid,indices):
        idx=self.pid_to_idx.get(pid)
        return None if idx is None else np.stack([self._load(i)[idx] for i in indices])
    def _load(self,i):
        if i not in self.cache:
            self.cache[i]=np.load(self.files[i],mmap_mode='r')
            if len(self.cache)>self.max_cache: self.cache.popitem(last=False)
        else: self.cache.move_to_end(i)
        return self.cache[i]
    def clear(self): self.cache.clear()

def _entries(dates,disk_info,pids):
    d2i={d:i for i,d in enumerate(dates)}; out={k:([],[]) for k in ('train','val','test')}; ends={k:[i for i,d in enumerate(dates) if i>=SEQ_LEN-1 and ((k=='train' and d<=TRAIN_CUTOFF) or (k=='val' and VAL_START<=d<=VAL_CUTOFF) or (k=='test' and d>=TEST_START))] for k in out}
    def allowed(k,d): return (k=='train' and d<=TRAIN_CUTOFF) or (k=='val' and VAL_START<=d<=VAL_CUTOFF) or (k=='test' and d>=TEST_START)
    for pi,pid in enumerate(pids):
        info=disk_info[pid]
        if info['is_failure'] and info['failure_time'] is not None:
            fi=d2i.get(info['failure_time'].strftime('%Y%m%d'))
            if fi is None: continue
            for k,leadmax in (('train',L),('val',TEST_LEAD_TIME),('test',TEST_LEAD_TIME)):
                for lead in range(1,leadmax+1):
                    ei=fi-lead
                    if ei>=SEQ_LEN-1 and allowed(k,dates[ei]):
                        out[k][0].append((list(range(ei-SEQ_LEN+1,ei+1)),pi,1.0))
                        if k!='train': break
        else:
            for k in out:
                if ends[k]:
                    ei=ends[k][RNG.randint(len(ends[k]))]; out[k][1].append((list(range(ei-SEQ_LEN+1,ei+1)),pi,0.0))
    pos,neg=out['train'];
    if len(neg)>len(pos)*NEGATIVE_RATIO: neg=[neg[i] for i in RNG.choice(len(neg),len(pos)*NEGATIVE_RATIO,replace=False)]
    out['train']=(pos,neg); return out

def _save_split(entries,pattern,shards,store,pids,neighbor_map):
    all_entries=entries[0]+entries[1]
    if not all_entries: return 0
    size=(len(all_entries)+shards-1)//shards; actual=(len(all_entries)+size-1)//size
    for s in range(actual):
        group=all_entries[s*size:(s+1)*size]; n=len(group); sa=np.zeros((n,SEQ_LEN,INPUT_FEAT_DIM),np.float32); na=np.zeros((n,MAX_NEIGHBORS,SEQ_LEN,INPUT_FEAT_DIM),np.float32); ma=np.zeros((n,MAX_NEIGHBORS),bool); la=np.zeros((n,1),np.float32)
        for j,(w,pi,y) in enumerate(group):
            pid=pids[pi]; seq=store.get(pid,w)
            if seq is None or not (seq[:,RAW_FEAT_DIM:2*RAW_FEAT_DIM] < .5).any(): continue
            sa[j]=seq; la[j,0]=y
            for k,npid in enumerate(neighbor_map.get(pid,[])[:MAX_NEIGHBORS]):
                ns=store.get(npid,w)
                if ns is not None and np.isfinite(ns).all() and (ns[:,RAW_FEAT_DIM:2*RAW_FEAT_DIM] < .5).any(): na[j,k]=ns; ma[j,k]=True
        np.savez_compressed(pattern.format(s),s=sa,n=na,m=ma,l=la)
    return actual

def build_and_save_samples():
    info=_load_disk_info(); neighbors=_load_neighbor_map(info); pids=_get_all_pids(info); dates,_,pmap,files=_extract_and_build_feat(info,pids,neighbors); store=FeatStore(files,pmap); ent=_entries(dates,info,pids); return (_save_split(ent['train'],TRAIN_SHARD_PATTERN,TRAIN_SHARDS,store,pids,neighbors),_save_split(ent['val'],VAL_SHARD_PATTERN,VAL_SHARDS,store,pids,neighbors),_save_split(ent['test'],TEST_SHARD_PATTERN,TEST_SHARDS,store,pids,neighbors))

def _loader(path,shuffle,batch_size,indices=None):
    d=np.load(path); idx=np.arange(len(d['l'])) if indices is None else np.asarray(indices); ds=torch.utils.data.TensorDataset(torch.from_numpy(np.asarray(d['s'])[idx]).float(),torch.from_numpy(np.asarray(d['n'])[idx]).float(),torch.from_numpy(np.asarray(d['m'])[idx]).bool(),torch.from_numpy(np.asarray(d['l'])[idx]).float()); return torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=shuffle,drop_last=False)
def load_data():
    if not (os.path.exists(TRAIN_SHARD_PATTERN.format(0)) and os.path.exists(VAL_SHARD_PATTERN.format(0)) and os.path.exists(TEST_SHARD_PATTERN.format(0))): return build_and_save_samples()
    return get_num_train_shards(),get_num_val_shards(),get_num_test_shards()
def load_train_shard(i,indices=None): return _loader(TRAIN_SHARD_PATTERN.format(i),True,BATCH_SIZE,indices)
def load_val_shard(i): return _loader(VAL_SHARD_PATTERN.format(i),False,BATCH_SIZE*4)
def load_test_shard(i): return _loader(TEST_SHARD_PATTERN.format(i),False,BATCH_SIZE*4)
def _count(pattern,n): return sum(os.path.exists(pattern.format(i)) for i in range(n))
def get_num_train_shards(): return _count(TRAIN_SHARD_PATTERN,TRAIN_SHARDS)
def get_num_val_shards(): return _count(VAL_SHARD_PATTERN,VAL_SHARDS)
def get_num_test_shards(): return _count(TEST_SHARD_PATTERN,TEST_SHARDS)
