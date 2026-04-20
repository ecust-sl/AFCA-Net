import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self,
                 text_dim: int,        # 文本 embedding 维度（例如 768）
                 img_channels: int = 512,
                 depth: int = 3,
                 height: int = 28,
                 width: int = 28,
                 embed_dim: int = 256,  # attention 内部维度
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.img_channels = img_channels
        self.depth = depth
        self.height = height
        self.width = width

        # 三个线性层分别将 img_tokens 映射到 Q，将 text_tokens 映射到 K 和 V
        self.q_proj = nn.Linear(img_channels, embed_dim)
        self.k_proj = nn.Linear(text_dim, embed_dim)
        self.v_proj = nn.Linear(text_dim, embed_dim)

        # 多头注意力
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout,
                                          batch_first=True)

        # 将注意力输出投回到 img_channels
        self.out_proj = nn.Linear(embed_dim, img_channels)

    def forward(self,
                text_feat: torch.Tensor,
                img_feat: torch.Tensor):
        """
        text_feat: [B, S, text_dim]   (例如 [16, 4, 768])
        img_feat:  [B, C, D, H, W]     (例如 [16, 512, 3, 28, 28])
        returns:   [B, C, D, H, W]
        """
        B, C, D, H, W = img_feat.shape
        S = text_feat.size(1)  # 文本序列长度

        # 1) 展平图像 tokens -> [B, N, C]
        N = D * H * W
        img_tokens = img_feat.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        # 2) 线性映射到 Q, K, V
        Q = self.q_proj(img_tokens)        # [B, N, E]
        K = self.k_proj(text_feat)         # [B, S, E]
        V = self.v_proj(text_feat)         # [B, S, E]

        # 3) 交叉注意力
        attn_out, attn_weights = self.attn(Q, K, V)
        # attn_out: [B, N, E]

        # 4) 投影回通道数
        img_fused = self.out_proj(attn_out)  # [B, N, C]

        # 5) reshape 回原始维度
        img_fused = img_fused.permute(0, 2, 1).contiguous()  # [B, C, N]
        img_fused = img_fused.view(B, C, D, H, W)             # [B, C, D, H, W]

        return img_fused


# ==== 使用示例 ====
if __name__ == "__main__":
    batch_size = 16
    seq_len    = 4
    text_dim   = 768
    C, D, H, W = 512, 3, 28, 28

    model = CrossAttentionFusion(
        text_dim=text_dim,
        img_channels=C,
        depth=D,
        height=H,
        width=W,
        embed_dim=256,
        num_heads=8,
        dropout=0.1
    )

    # 随机伪数据
    text_feat = torch.randn(batch_size, seq_len, text_dim)  # [16, 4, 768]
    img_feat  = torch.randn(batch_size, C, D, H, W)         # [16, 512, 3, 28, 28]

    out = model(text_feat, img_feat)  # [16, 512, 3, 28, 28]
    print("Fused feature shape:", out.shape)
