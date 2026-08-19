# models.py
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

# ============================================================
# 组件1：邻域感知组件 (对应论文 Figure 2)
# ============================================================
class NeighborhoodAttention(nn.Module):
    # [a,b]中a是行数，b是列数
    # batch_size是同时处理的硬盘的数量
    # feature_dim是特征数量
    # max_neighbors是最大邻居数
    
    """
    输入形状:
        self_feat:   [batch_size, feature_dim]     当前磁盘在单个时间戳的特征
        neigh_feat:  [batch_size, max_neighbors, feature_dim]  所有邻居在同一个时间戳的特征
        mask:        [batch_size, max_neighbors]    1=真实邻居, 0=padding
    输出:
        r:           [batch_size, feature_dim]      融合邻域信息后的编码向量
    """

    def __init__(self, feat_dim):
        super().__init__()
        # Q 和 K 是单层全连接 (论文里没写多层, 默认单层)
        self.Q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.K = nn.Linear(feat_dim, feat_dim, bias=False)
        # 注意: 没有 V 层, Value 直接取原始邻居特征

    def forward(self, self_feat, neigh_feat, mask):
        # Step 1: 计算 Query 和 Key
        q = self.Q(self_feat)                    # [B, F]
        k = self.K(neigh_feat)                   # [B, M, F]

        # Step 2: 点积注意力 (论文式 (2): w_j = exp(q·k_j) / Σ exp(q·k_z)，无缩放因子)
        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1))  # [B, 1, M]

        # Step 3: 关键! 对 padding 位置进行掩码 (设为 -inf)
        if mask is not None:
            # mask 形状 [B, M] -> unsqueeze(1) -> [B, 1, M]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        # Step 4: Softmax 得到权重
        attn_weights = F.softmax(scores, dim=-1)   # [B, 1, M]
        # 修复：全 padding 时 softmax([-inf, -inf, ...]) → NaN，置零后残差连接退化为只保留自身特征
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Step 5: 加权聚合邻居的 Value (直接使用原始邻居特征)
        c = torch.matmul(attn_weights, neigh_feat)  # [B, 1, F]
        c = c.squeeze(1)                            # [B, F]

        # Step 6: 残差连接 (r = a + c)
        r = self_feat + c
        return r


# ============================================================
# 组件2：时序编码组件 (对应论文 Figure 3)
# ============================================================
class TemporalEncoder(nn.Module):
    """
    输入:
        r_sequence: [batch_size, seq_len, feature_dim]  邻域编码后的向量序列
    输出:
        s:          [batch_size, feature_dim]           融合时序信息后的向量
    """
    def __init__(self, feat_dim, seq_len, num_layers, nhead, dropout):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # 2.1 位置编码 (Positional Embedding) - 使用可学习的版本
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, feat_dim))

        # 2.2 Transformer 编码器 (只使用 Encoder 部分)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=nhead,
            dim_feedforward=feat_dim * 8,   # 加宽FFN (4x→8x) 提升模型容量
            dropout=dropout,
            activation='relu',
            batch_first=True               # 让输入变成 [B, T, F]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 2.3 时间感知注意力层 (Time-Aware Attention)
        # 论文式 (3): v_t = softmax(FC(r''(t))), 为每个时间步计算权重
        self.time_attn_fc = nn.Linear(feat_dim, 1)   # 输出一个标量分数

    def forward(self, r_sequence):
        # r_sequence: [B, T, F]

        # 加入位置编码
        r_sequence = r_sequence + self.pos_embedding   # [B, T, F]

        # 通过 Transformer 编码器 (让每个时间步融合上下文信息)
        r_encoded = self.transformer_encoder(r_sequence)  # [B, T, F]

        # 时间感知注意力: 计算每个时间步的权重
        scores = self.time_attn_fc(r_encoded)           # [B, T, 1]
        attn_weights = F.softmax(scores, dim=1)         # [B, T, 1]

        # 加权求和得到最终向量 s
        s = torch.sum(attn_weights * r_encoded, dim=1)  # [B, F]

        return s


# ============================================================
# 组件3：完整 NTAM 模型 (对应论文 Figure 1)
# ============================================================
class NTAM(nn.Module):
    """
    输入:
        self_feat_seq:   [batch_size, seq_len, feature_dim]   当前磁盘的时序数据
        neigh_feat_seq:  [batch_size, max_neighbors, seq_len, feature_dim]  所有邻居的时序数据
        neighbor_mask:   [batch_size, max_neighbors]           有效邻居掩码 (所有时间步共享)
    输出:
        prob:            [batch_size, 1]                      故障概率
    
    use_neighborhood: True=完整NTAM, False=消融实验(跳过邻域组件, 对应论文NTAM_alt1)
    """
    def __init__(self, feat_dim, seq_len, max_neighbors, num_layers, nhead, dropout,
                 use_neighborhood=True):
        super().__init__()
        self.seq_len = seq_len
        self.max_neighbors = max_neighbors
        self.use_neighborhood = use_neighborhood

        # 邻域感知组件
        if use_neighborhood:
            self.neighborhood = NeighborhoodAttention(feat_dim)

        # 时序编码组件
        self.temporal = TemporalEncoder(feat_dim, seq_len, num_layers, nhead, dropout)

        # 决策组件 (分类头)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1)   # 注意: 没有 Sigmoid, 配合 BCEWithLogitsLoss
        )

    def forward(self, self_feat_seq, neigh_feat_seq, neighbor_mask):
        B, T, F = self_feat_seq.shape
        M = self.max_neighbors

        if self.use_neighborhood:
            # ====== 完整路径：邻域注意力 → 时序编码 → 决策 ======
            # 1) 展平批次和时间维度
            self_feat_flat = self_feat_seq.view(B * T, F)          # [B*T, F]
            # data_utils 的真实布局是 [B, M, T, F]（neigh_seq_arr 为 (MAX_NEIGHBORS, SEQ_LEN, FEAT_DIM)）。
            # 必须先 permute 成 [B, T, M, F] 再展平，否则 view 会把 (邻居, 时间) 两维混在一起。
            if neigh_feat_seq.shape[1] != M or neigh_feat_seq.shape[2] != T:
                raise ValueError(
                    f"neigh_feat_seq 形状异常: {tuple(neigh_feat_seq.shape)}，"
                    f"期望 [B, M={M}, T={T}, F]")
            neigh_feat_flat = neigh_feat_seq.permute(0, 2, 1, 3).reshape(B * T, M, F)  # [B*T, M, F]

            # 2) 扩展掩码以匹配 B*T（expand 零拷贝，不额外分配内存）
            mask_flat = neighbor_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, M)  # [B*T, M]

            # 3) 通过邻域感知组件 (每个时间步独立聚合邻居)
            r_flat = self.neighborhood(self_feat_flat, neigh_feat_flat, mask_flat)  # [B*T, F]

            # 4) 恢复时序形状 [B, T, F]
            r_seq = r_flat.view(B, T, F)
        else:
            # ====== 消融实验路径：跳过邻域组件，自身特征直接传入时序编码 ======
            r_seq = self_feat_seq

        # 5) 通过时序编码组件
        s = self.temporal(r_seq)   # [B, F]

        # 6) 通过决策组件
        logits = self.classifier(s)  # [B, 1]
        prob = torch.sigmoid(logits) # [B, 1]
        return prob, logits
