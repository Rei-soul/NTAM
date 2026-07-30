# NTAM复现

```python
disk_failure_prediction/
│—— xxxx               # 外部库等
├── models.py          # 核心：存放所有深度学习组件（Neighborhood, Temporal, NTAM）
├── data_utils.py      # 辅助：生成样本，分片策略训练
├── train.py           # 入口：读取数据、实例化模型、训练循环
├── config.py          # 可选：存放所有超参数（feature_dim, L, lr等）
├── memory_guard.py    # 一个检测内存并防止溢出导致电脑死机的策略
└── prefilter.py       # 数据预处理，生成target_disks.csv，将node_id和model映射起来，作为唯一硬盘标识。同时记录failture time等信息，作为总的清单
```

## 结果：

### 指标：

|训练集时间跨度|测试集时间跨度|采样硬盘数|Acc|Precision|Recall|F1|
|---|---|---|---|---|---|---|
|2018年1月初-2.15|20180215-二月底|2w|0.9446|0.0092|0.2381|0.0177|

模型尽量全猜健康，因为测试集正负样本相差悬殊，所以Acc无意义
真正反映模型能力的指标是Recall 23.8%（模型能检测到约 1/4 的故障盘），以及 F1=0.0177，综合表现很差。

|训练集时间跨度|测试集时间跨度|采样硬盘数|Acc|Precision|Recall|F1|
|---|---|---|---|---|---|---|
|2018年1月初-2.15|20180215-二月底|6w|0.9799|0.0100|0.5000|0.0195|

|训练集时间跨度|测试集时间跨度|采样硬盘数|Acc|Precision|Recall|F1|
|---|---|---|---|---|---|---|
|2018年1月初-4.15|20180415-4月底|4w|0.9976|0.0000|0.0000|0.0000|

---

## 审查文件的推荐顺序（按数据流自底向上）

```PowerShell
① config.py      超参数中心（SEQ_LEN, NUM_HEADS, USE_NEIGHBORHOOD）
        ↓
② prefilter.py   离线预处理（node_id 提取 → model 采样 → parquet 输出）
        ↓
③ data_utils.py  运行时数据加载（邻居混合策略 + 按时间切分 + 下采样）
        ↓
④ models.py      NTAM 模型定义（邻域注意力 / 时序编码 / 决策组件）
        ↓
⑤ train.py       训练入口（数据加载 → 模型初始化 → 训练循环 → 评估）
```

### 各文件审查重点

| 顺序 | 文件 | 重点检查 |
|---|---|---|
| **①** | `config.py` | 参数值是否合理；`USE_NEIGHBORHOOD` 开关机制 |
| **②** | `prefilter.py` | `node_id` 从 `ssd_failure_tag2.csv` 正确提取；磁盘 `model` 唯一性过滤；parquet 输出完整性 |
| **③** | `data_utils.py` | **第 132-230 行 `build_samples()` 是核心** — 按时间切分训练/测试（`train_cutoff_date="20180120"`）；邻居混合策略（node_id 优先 + model 补齐）；下采样比例 1:10 |
| **④** | `models.py` | `NeighborhoodAttention` 残差连接；`TemporalEncoder` 位置编码+Transformer+时间注意力；`NTAM.forward()` 中 `use_neighborhood` 分支逻辑 |
| **⑤** | `train.py` | BCEWithLogitsLoss 用法；Adam 优化器配置；评估指标计算 |

---

## `prefilter.py` 运行结果

```
[1/5] 加载全量位置信息...
  位置信息覆盖 965,495 个 (disk_id, model) 组合 ← 来自 location_info_of_ssd.csv

[2/5] 扫描1月CSV (31天)...
  总 (disk_id, model) 组合数: 818,667

[3/5] 加载故障标签 (ssd_failure_tag2.csv)...
  故障标签数: 18,387
  在1月数据中出现的: 14,742

[4/5] 筛选...
  目标磁盘对: 818,667 (故障: 14,742, 健康: 803,925)
  位置覆盖率: 100% (无丢弃)
  邻居潜力: 127,078 个 node 有≥2个盘, 平均每 node 4.1 盘

[5/5] 提取 SMART 数据...
  输出: smart_201801.parquet (24,300,680 行 × 55 列)
  输出: target_disks.csv (818,667 条)
```

### 关键结论

| 指标 | 数值 |
|---|---|
| 总磁盘对 | **818,667** |
| 故障:健康 | **14,742 : 803,925** (约 1:55) |
| 位置覆盖 | **100%** — 每块盘都有 node_id |
| 节点邻居 | **平均 4.1 盘/节点** — 邻域组件有足够的候选邻居 |

这是一个高质量的全量中间数据集，问题出在后续 `data_utils.py` 的划分策略上。

## `data_utils.py` 的职能 — 简明版

```
输入（从 prefilter.py 来）:
  target_disks.csv         818K 磁盘对清单
  smart_201801.parquet     24M 行 SMART 数据
  feat_tensor.npy           3D 张量 (首次运行时生成)

处理流程:
  ① 随机采样 MAX_PAIRS 个磁盘对 (60K)
  ② 构建邻居映射 (node_id物理邻居 + model补齐)
  ③ 15天滑动窗口 × 60K = 1,015,808 个窗口样本
  ④ 标签: 窗口结束日距故障日 1~15天 → positive
  ⑤ 训练/测试按窗口结束日划分 (1月20日为界)
  ⑥ 训练集下采样 1:10，测试集全量
  ⑦ 保存 train_samples.npz + test_samples.npz

输出（给 train.py）:
  train_loader, test_loader (PyTorch DataLoader)
```

### 总结

把 818K 个磁盘对的 31 天 SMART 数据，转换成可供 NTAM 模型训练的 `[B, 15, 51]` 时序样本，并处理类别不平衡、邻居映射、训练/测试划分。


# 邻居怎么选？每次打乱随机选择邻居还是固定邻居

## 三个问题的详细回答

经过逐行审查代码，以下是我的分析：

---

### 问题 1：故障盘 A 于 1 月 20 日故障，1 月 20 日之前的 15 天是否都被标记为故障？

**是的，代码正确实现了时间渐进采样。** 核心逻辑在 `data_utils.py` 的 `_generate_and_save_samples` 中：

```python
window_end_dt = pd.to_datetime(date_str, format="%Y%m%d")
days = (info['failure_time'] - window_end_dt).days  # 故障日 - 窗口结束日
if 1 <= days <= 15:
    label = 1.0  # 标记为"即将故障"
```

具体到 1 月 20 日故障的盘 A：

| 窗口结束日期 | 包含的 SMART 数据 | days (20 - D) | Label |
|:---:|------|:---:|:---:|
| 1 月 15 日 | 1 月 1-15 日 | 5 ✅ | **1** |
| 1 月 16 日 | 1 月 2-16 日 | 4 ✅ | **1** |
| 1 月 17 日 | 1 月 3-17 日 | 3 ✅ | **1** |
| 1 月 18 日 | 1 月 4-18 日 | 2 ✅ | **1** |
| 1 月 19 日 | 1 月 5-19 日 | 1 ✅ | **1** |
| 1 月 20 日 | 1 月 6-20 日 | 0 ❌ | **0**（当天已故障，不预测） |

每个窗口是一个 **15 天滑动窗口**（`SEQ_LEN=15`），这意味着：
- 1 月 15 日的样本：看到 1-15 日的数据，预测"5 天后会故障" ✅
- 1 月 19 日的样本：看到 5-19 日的数据，预测"1 天后会故障" ✅

这正是论文中描述的时间渐进采样——**同一块盘的故障前数据会被生成多个正样本**，每个正样本代表从不同时间点预测同一个故障事件。

---

### 问题 2：模型输入和 Transformer 时序编码是否正确？

**是的，代码完全正确。** 数据流如下：

```
输入:
  sf: [B, 15, 51]        ← 当前磁盘的 15 天 SMART 特征
  nf: [B, 5, 15, 51]     ← 5 个邻居的 15 天 SMART 特征
  nm: [B, 5]              ← 邻居有效 mask

模型前向过程 (NTAM.forward):
  ┌─────────────────────────────────────────────┐
  │ 1. 展平时间维度                              │
  │    self_feat:  [B×15, 51]                    │
  │    neigh_feat: [B×15, 5, 51]                 │
  │                                             │
  │ 2. 每个时间步独立做邻域注意力                 │
  │    NeighborhoodAttention(self_feat, neigh)   │
  │    → r_flat: [B×15, 51]                      │
  │    → r_seq:  [B, 15, 51]  (恢复时序形状)     │
  │                                             │
  │ 3. TemporalEncoder(时序编码)                  │
  │    r_seq + pos_embedding                     │
  │    → Transformer Encoder (多头自注意力)       │
  │    → Time-Aware Attention Pooling            │
  │    → s: [B, 51]                              │
  │                                             │
  │ 4. Classifier → logits → prob                │
  └─────────────────────────────────────────────┘
```

`TemporalEncoder` 的实现细节：
- **位置编码**：`nn.Parameter(torch.randn(1, seq_len, feat_dim))`，可学习的
- **Transformer Encoder**：3 层，3 头注意力，`batch_first=True`，直接处理 `[B, T, F]` 格式
- **时间感知注意力**：`nn.Linear(feat_dim, 1)` → `softmax` → 加权求和得到单一向量

**结论**：邻域组件处理后、带有时间信息的数据正确传入了 Transformer，且 Transformer 的 `batch_first=True` 确保了 `[B, T, F]` 格式被正确解析（T 维度是时间步）。已验证之前在测试中 `Loss` 持续下降（从 0.64 → 0.10），证明梯度正确流动。

---

### 问题 3：能否生成中间文件防止崩溃后重新生成样本？

**当前代码已经实现了这个机制。** 流程如下：

```
python code/train.py
    ↓
load_data()
    ├─ 检查 train_shard_00.npz 和 test_shard_00.npz 是否存在
    │
    ├─ 不存在 → build_and_save_samples()
    │            ├─ 采样磁盘 + 找邻居
    │            ├─ 从 CSV 提取 feat_tensor.npy (memmap, 磁盘直写)
    │            ├─ 生成 train_shard_00~09.npz  (10 个文件)
    │            └─ 生成 test_shard_00~09.npz   (10 个文件)
    │
    └─ 已存在 → 跳过构建，直接加载 npz → 训练
```

---

## 中间产物清理规则

### 哪些是中间产物

| 文件 | 作用 | 何时生成 |
|------|------|:---:|
| `feat_tensor.npy` | 所有磁盘的 SMART 特征 3D 张量 | `build_and_save_samples()` |
| `train_shard_XX.npz` | 训练集分片样本 | `build_and_save_samples()` |
| `test_shard_XX.npz` | 测试集分片样本 | `build_and_save_samples()` |
| `*.tmp` | memmap 临时文件 | 生成过程中，正常结束会自动删除 |

### 什么情况需要清理

| 场景 | 需要清理？ | 原因 |
|------|:---:|------|
| **只改了代码逻辑**（如改梯度裁剪、学习率） | ❌ 不需要 | 数据没变，只是训练方式变了 |
| **改了数据范围**（DATA_MONTH_END、MAX_PAIRS） | ✅ 需要 | 数据变了，旧 npz 是旧数据 |
| **改了采样逻辑**（如分层采样） | ✅ 需要 | 样本变了 |
| **改了 TRAIN_CUTOFF** | ✅ 需要 | 训练/测试划分变了 |
| **重复运行同一配置** | ❌ 不需要 | `load_data()` 会自动跳过构建 |

### 内置跳过机制

`data_utils.py` 的 `load_data()` 已经做了判断：

```python
if not os.path.exists(TRAIN_SHARD_PATTERN.format(0)):
    return build_and_save_samples()   # 构建新数据
else:
    # 已存在 → 直接加载，跳过构建 ✅
```

---