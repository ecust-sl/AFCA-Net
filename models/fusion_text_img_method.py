import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, MultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

class AddFusion(nn.Module):
    def __init__(self, C1=512, C2=768):
        super().__init__()
        # video→text 通道映射
        self.proj   = nn.Conv3d(C1, C2, kernel_size=(3,4,4), stride=(1,4,4))
        self.flatten= nn.Flatten(start_dim=2, end_dim=4)
        self.reduce = nn.Conv1d(C2, C2, kernel_size=3, padding=1)
        # 两个分类头
        self.binary_head = nn.Linear(C2, 1)
        self.three_head  = nn.Linear(C2, 3)

    def forward(self, x1, x2):
        # x1: [B,512,3,28,28], x2: [B, L, 768]
        B, L, C2 = x2.shape
        # ==== 视频特征投影 & 展平 ====
        v = self.proj(x1)               # → [B,768,t',h',w']
        v = self.flatten(v)             # → [B,768,N]
        v = self.reduce(v)              # → [B,768,N]
        v = v.transpose(1,2)            # → [B,N,768]
        # ==== 序列长插值到 L ====
        # 先变成 [B,768,N,1] 才能用双线性插值到 (L,1)
        v = v.permute(0,2,1).unsqueeze(-1)              # [B,768,N,1]
        v = F.interpolate(v, size=(L,1), mode='bilinear', align_corners=False)
        v = v.squeeze(-1).permute(0,2,1)                # [B,L,768]
        # ==== 融合 & 分类 ====
        fused = v + x2                                 # [B,L,768]
        cls_feat = fused.mean(dim=1)                   # [B,768]
        return fused, self.binary_head(cls_feat), self.three_head(cls_feat)


class CatFusion(nn.Module):
    def __init__(self, C1=512, C2=768, hidden=1024):
        super().__init__()
        # 融合 MLP
        self.mlp = nn.Sequential(
            nn.Linear(C1 + C2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, C2)
        )
        self.binary_head = nn.Linear(C2, 1)
        self.three_head  = nn.Linear(C2, 3)

    def forward(self, x1, x2):
        # x1: [B,512,3,28,28], x2: [B, L, 768]
        B, L, C2 = x2.shape
        # ==== 自适应池到 L tokens ====
        v = F.adaptive_avg_pool3d(x1, output_size=(L,1,1))  # [B,512,L,1,1]
        v = v.view(B, x1.size(1), L)                        # [B,512,L]
        v = v.permute(0,2,1)                                # [B,L,512]
        # ==== 拼接 & MLP 融合 ====
        cat   = torch.cat([v, x2], dim=-1)                  # [B,L,1280]
        fused = self.mlp(cat)                               # [B,L,768]
        cls_feat = fused.mean(dim=1)                        # [B,768]
        return fused, self.binary_head(cls_feat), self.three_head(cls_feat)

class CrossAttnFusion(nn.Module):
    def __init__(self, Cq=768, Ckv=512, n_heads=8):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2, end_dim=4)
        self.k_proj = nn.Linear(Ckv, Cq)
        self.v_proj = nn.Linear(Ckv, Cq)
        self.cross_attn = MultiheadAttention(embed_dim=Cq,
                                             num_heads=n_heads,
                                             batch_first=True)
        self.binary_head = nn.Linear(Cq, 1)
        self.three_head = nn.Linear(Cq, 3)

    def forward(self, x1, x2):
        # x1: [B,512,3,28,28]
        v = self.flatten(x1)             # [B,512,2352]
        v = v.transpose(1,2)             # [B,2352,512]
        k = self.k_proj(v)               # [B,2352,768]
        v_ = self.v_proj(v)              # [B,2352,768]
        # x2: [B,65,768]
        attn_out, _ = self.cross_attn(query=x2,
                                      key=k,
                                      value=v_)
        # attn_out: [B,65,768]
        cls_feat = attn_out.mean(dim=1)  # [B,768]
        bin_logits = self.binary_head(cls_feat)
        three_logits = self.three_head(cls_feat)
        return attn_out, bin_logits, three_logits
