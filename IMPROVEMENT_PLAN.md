# NTAM 模型改进方案

## 问题诊断总结

### 1. 根本问题：输入尺度偏小
- **现象**：E[x²] = 0.175（应为1.0）、‖q‖ = 0.87、logit_spread = 0.036
- **后果**：注意力权重近乎均匀（maxw ≈ 1/M），邻域模块退化为"等权平均池化"
- **原因**：SMART数据是"重尾+大量0"的稀疏分布，Z-score标准化后σ被高估，导致大部分值|z|很小

### 2. 特征ablation实验结果
```
组合                      AUC      AP      维度
J delta+lastk+absmax    0.8518   0.7254   90   ⭐ 用60%维度追平全特征
A 全30×5                0.8515   0.7168   150
I raw+vol+nzdays        0.8449   0.7291   90
F raw+vol               0.8366   0.7186   60
D 仅vol                 0.8121   0.6874   30
E 仅raw                 0.7805   0.6643   30
```

**关键发现**：
- **delta + lastk + absmax** 三件套包含核心信息
- **vol（波动性）** 比 **raw（均值）** 更有判别力
- 当前NTAM (0.823) < 逻辑回归+手工特征 (0.8518)

### 3. 邻域增量≈0
- ΔAUC(有邻域 - 无邻域) ≈ +0.001
- 9列"机架共模"特征：自身判别力强，但邻居同样强 → 污染注意力

## 改进方案（按优先级）

### 🔥 Priority A：输入LayerNorm（最小改动，实测效果最强）
**目标**：让每行特征达到unit-variance，消除列间尺度差异

**改动**：
```python
# models.py - NeighborhoodAttention.__init__
self.input_ln_self = nn.LayerNorm(feat_dim)   # 对自身特征归一化
self.input_ln_neigh = nn.LayerNorm(feat_dim)  # 对邻居特征归一化

# forward
self_feat = self.input_ln_self(self_feat)
neigh_feat = self.input_ln_neigh(neigh_feat)
```

**预期**：
- logit_spread: 0.036 → 0.32（×9倍）
- ‖q‖: 0.87 → 2.90
- 注意力从"均匀分布"恢复到"可选择性分布"

**成本**：3行代码 + 2个config开关

---

### 🔥 Priority B：添加统计特征通道
**目标**：让模型直接"看到" vol/absmax/delta，而不是从原始序列自己算

**改动**：
1. 在data_utils中增加特征通道：
   - `raw`: 窗口均值（现有）
   - `vol`: 窗口标准差（波动性）
   - `absmax`: 窗口内|x|最大值（历史峰值）
   - `lastk`: 最后3天均值（近期水平）
   - `delta`: 后7天 - 前23天（变化趋势）

2. 输入维度：30列 × 5通道 = 150维

**预期**：
- 逻辑回归探针：0.8518（已验证）
- NTAM有望超过0.85（获得时序建模的额外增益）

**成本**：修改build_feat_r.py + 重建分片（2-4小时）

---

### 🔥 Priority C：改进邻域融合方式

#### 方案C1：门控融合（推荐）
```python
# r = self + g ⊙ c
# g = σ(MLP([self, c, self-c]))
self.gate_fc = nn.Sequential(
    nn.Linear(feat_dim * 3, feat_dim),
    nn.Sigmoid()
)

def forward(self, self_feat, neigh_feat, mask):
    # ... 计算 c（邻域聚合）...
    diff = self_feat - c
    gate = self.gate_fc(torch.cat([self_feat, c, diff], dim=-1))
    r = self_feat + gate * c  # 可学习的选择性融合
    return r
```

#### 方案C2：后置差分（简单）
```python
# 在时序编码后拼接差异特征
s_self = self.temporal(self_feat_seq)       # [B, F]
s_nb = self.temporal(r_seq)                 # [B, F]
s_diff = s_self - s_nb
s = torch.cat([s_self, s_nb, s_diff], dim=-1)  # [B, 3F]
# classifier输入维度改为 3*feat_dim
```

**预期**：
- 异常度保持率：0.72× → 0.90+×
- 邻域增量变为可观测

**成本**：30行 + config开关

---

### Priority D：1/√F缩放（在A之后）
```python
# models.py - NeighborhoodAttention.forward
scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1))
scores = scores / (self.feat_dim ** 0.5)  # 标准Transformer缩放
```

**注意**：必须在LayerNorm之后加，否则会让本已很小的logits更小

---

### Priority E：删除退化列（可选）
基于feature_utility.json：
- 删除：232, 233（全0退化）
- 删除：177, 182（max-strength < 0.04）
- 保留26列，改NUM_HEADS=2（26不能被3整除）

**预期**：性能无损（B组AUC=0.8526 ≈ A组0.8515）

---

## 实验矩阵（消融实验）

| 变体 | 输入LN | 统计通道 | 1/√F | 融合方式 | 预期AUC |
|------|--------|----------|------|----------|---------|
| V0 (baseline) | ✗ | ✗ (仅raw) | ✗ | 前置加法 | 0.823 |
| V1 | ✓ | ✗ | ✗ | 前置加法 | 0.83~0.84 |
| V2 | ✓ | ✓ (5通道) | ✗ | 前置加法 | 0.84~0.85 |
| V3 | ✓ | ✓ | ✓ | 前置加法 | 0.84~0.85 |
| **V4** | ✓ | ✓ | ✓ | **门控** | **0.85~0.86** |
| V5 | ✓ | ✓ | ✓ | 后置差分 | 0.85~0.86 |

## 实施步骤

### 阶段1：快速验证（1天）
1. ✅ 跑feature_ablation实验（已完成）
2. 🔧 实现A（输入LayerNorm）+ D（缩放）+ 自检诊断工具
3. 训练V1，验证logit_spread是否×9

### 阶段2：完整方案（2-3天）
4. 🔧 实现B（统计通道），重建分片
5. 训练V2/V3
6. 🔧 实现C1/C2（融合改造）
7. 训练V4/V5，选最优

### 阶段3：精细化（可选）
8. 稳健标准化（median+MAD替代mean+std）
9. 缺失掩码（区分0=缺失 vs 0=正常）
10. 模型压缩（删退化列、降维到24）

## 预期收益

| 改进项 | 成本 | AUC提升 | 优先级 |
|--------|------|---------|--------|
| 输入LayerNorm | 3行代码 | +0.01~0.02 | ⭐⭐⭐ |
| 统计通道 | 重建分片2h | +0.02~0.03 | ⭐⭐⭐ |
| 门控融合 | 30行代码 | +0.01~0.02 | ⭐⭐ |
| 1/√F缩放 | 1行代码 | +0.005 | ⭐ |

**累计预期**：0.823 → 0.85~0.86（+0.027~0.037）

## 风险与注意事项

1. **LayerNorm位置**：必须在邻域注意力之前，而非Transformer之前
2. **缩放时机**：必须在LayerNorm之后，否则适得其反
3. **统计通道**：需要重建分片，确保备份原分片
4. **消融实验**：每个变体都要跑，才能定位每个改动的真实贡献

## 论文撰写建议

### 消融表（Table X）
```
| Model | Input LN | Stat Channels | Scaling | Fusion | AUC | AP | F1 |
|-------|----------|---------------|---------|--------|-----|----|----|
| NTAM-baseline | ✗ | ✗ | ✗ | Add | 0.823 | ... | ... |
| NTAM-v1 | ✓ | ✗ | ✗ | Add | 0.83x | ... | ... |
| NTAM-v2 | ✓ | ✓ | ✗ | Add | 0.84x | ... | ... |
| NTAM-v4 (ours) | ✓ | ✓ | ✓ | Gate | 0.85x | ... | ... |
```

### 分析章节要点
1. **诊断发现**：用attention_saturation.json + feature_utility.json的数据
2. **设计动机**：基于logit_spread和feature ablation的发现
3. **效果归因**：逐项拆解每个改动的贡献（消融实验）

---

**作者**：基于DeepSeek分析 + 特征ablation实验  
**日期**：2025-01  
**状态**：待实施
