import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborhoodAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(dim + 1, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, self_feat, neigh_feat, mask):
        valid = mask.bool()
        q = self.q(self_feat).unsqueeze(1)
        k = self.k(neigh_feat)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self_feat.size(-1))
        scores = scores.masked_fill(~valid.unsqueeze(1), -torch.finfo(scores.dtype).max)
        weights = F.softmax(scores, dim=-1)
        weights = weights * valid.unsqueeze(1).to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.matmul(weights / denom, neigh_feat).squeeze(1)
        presence = valid.float().mean(dim=1, keepdim=True)
        gate = self.gate(torch.cat([self_feat, presence], dim=-1))
        return self.norm(self_feat + gate * context)

class TemporalEncoder(nn.Module):
    def __init__(self, dim, seq_len, layers, heads, dropout):
        super().__init__()
        self.feat_dim, self.seq_len = dim, seq_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                           dropout=dropout, activation='gelu', batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.time_attn = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(self.encoder(x + self.pos_embedding))
        weights = F.softmax(self.time_attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class NTAM(nn.Module):
    def __init__(self, feat_dim, seq_len, max_neighbors, num_layers, nhead, dropout,
                 use_neighborhood=True, model_dim=None):
        super().__init__()
        model_dim = model_dim or feat_dim
        self.seq_len, self.max_neighbors = seq_len, max_neighbors
        self.use_neighborhood = use_neighborhood
        self.input_projection = nn.Sequential(nn.Linear(feat_dim, model_dim), nn.LayerNorm(model_dim), nn.GELU())
        if use_neighborhood: self.neighborhood = NeighborhoodAttention(model_dim)
        self.temporal = TemporalEncoder(model_dim, seq_len, num_layers, nhead, dropout)
        self.classifier = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 1))

    def forward(self, self_feat_seq, neigh_feat_seq, neighbor_mask):
        b, t, _ = self_feat_seq.shape
        self_x = self.input_projection(self_feat_seq)
        if self.use_neighborhood:
            if neigh_feat_seq.shape[1] != self.max_neighbors or neigh_feat_seq.shape[2] != t:
                raise ValueError(f'expected neighbors [B,{self.max_neighbors},{t},F], got {tuple(neigh_feat_seq.shape)}')
            neigh_x = self.input_projection(neigh_feat_seq)
            flat_self = self_x.reshape(b*t, -1)
            flat_neigh = neigh_x.permute(0, 2, 1, 3).reshape(b*t, self.max_neighbors, -1)
            flat_mask = neighbor_mask.unsqueeze(1).expand(-1, t, -1).reshape(b*t, self.max_neighbors)
            x = self.neighborhood(flat_self, flat_neigh, flat_mask).reshape(b, t, -1)
        else:
            x = self_x
        logits = self.classifier(self.temporal(x))
        return torch.sigmoid(logits), logits
