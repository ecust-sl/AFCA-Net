import torch
import torch.nn as nn
import torch.nn.functional as F

class DualModalAttention(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        # 注意力权重生成层
        self.attention_fc = nn.Linear(feature_dim * 2, 2)  # 输入1024 → 输出2
        # 最终融合层
        self.fusion_fc = nn.Linear(feature_dim * 2, feature_dim)

    def forward(self, mod1, mod2):
        """
        输入:
            mod1: [32, 512] (batch_size=32, features=512)
            mod2: [32, 512]
        输出:
            output: [32, 512]
        """
        # 拼接特征
        combined = torch.cat([mod1, mod2], dim=-1)  # [32, 1024]

        # 生成注意力权重
        att_scores = self.attention_fc(combined)  # [32, 2]
        att_weights = F.softmax(att_scores, dim=-1)  # 在特征维度做softmax

        # 分解权重
        weight1 = att_weights[:, 0].unsqueeze(-1)  # [32, 1]
        weight2 = att_weights[:, 1].unsqueeze(-1)  # [32, 1]

        # 加权特征（使用广播机制）
        weighted_mod1 = mod1 * weight1  # [32,512] * [32,1] → [32,512]
        weighted_mod2 = mod2 * weight2

        # 拼接加权后的特征
        fused = torch.cat([weighted_mod1, weighted_mod2], dim=-1)  # [32, 1024]

        # 最终融合
        output = self.fusion_fc(fused)  # [32, 512]
        return output


# 使用示例
mod1 = torch.randn(32, 512)  # 假设batch_size=32
mod2 = torch.randn(32, 512)

model = DualModalAttention()
output = model(mod1, mod2)
print(output.shape)  # torch.Size([32, 512])
class EnhancedAttentionFusion3D(nn.Module):
    def __init__(self, in_channels=512, reduction=4):
        super().__init__()

        # 确保中间通道数不小于1
        mid_channels = 1

        # 通道注意力分支（修正通道数计算）
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels * 2, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, in_channels * 2, 1),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(in_channels * 2, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        batch_size, channel, D, W, H = x1.shape
        combined = torch.cat([x1, x2], dim=1)

        # 通道注意力
        channel_weights = self.channel_attention(combined)
        channel_refined = combined * channel_weights

        # 空间注意力
        spatial_weights = self.spatial_attention(channel_refined)
        w1, w2 = torch.chunk(spatial_weights, 2, dim=1)

        return x1 * w1 + x2 * w2
#
#
# # 验证示例
# if __name__ == "__main__":
#     x1 = torch.randn(2, 1, 16, 32, 32)
#     x2 = torch.randn(2, 1, 16, 32, 32)
#
#     fusion = EnhancedAttentionFusion3D(reduction=2)  # 测试不同reduction值
#     output = fusion(x1, x2)
#     print(f"输出形状: {output.shape}")