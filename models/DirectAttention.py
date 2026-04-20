import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 注意力权重生成器（直接处理原始特征）
        self.conv_atten_1 = nn.Conv3d(in_channels, 1, kernel_size=1)  # 通道保持原始输入
        self.conv_atten_2 = nn.Conv3d(in_channels, 1, kernel_size=1)

        # 特征融合卷积（保持通道一致性）
        # self.conv_fuse = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feature1, feature2):
        # 直接生成注意力图（无需拼接其他特征）
        atten_map1 = self.conv_atten_1(feature1)  # [B, 1, D, W, H]
        atten_map2 = self.conv_atten_2(feature2)

        # 归一化注意力权重
        atten_weights = torch.cat([atten_map1, atten_map2], dim=1)
        atten_weights = F.softmax(atten_weights, dim=1)
        atten_weight1, atten_weight2 = torch.chunk(atten_weights, 2, dim=1)

        # 纯注意力驱动融合
        fused_feature = (feature1 * atten_weight1) + (feature2 * atten_weight2)

        # 最终特征增强
        return fused_feature


# 验证示例
if __name__ == "__main__":
    # 输入特征：batch=2, channels=16, 3D尺寸=32×64×64
    x1 = torch.randn(2, 16, 32, 64, 64)
    x2 = torch.randn(2, 16, 32, 64, 64)

    model = DirectAttentionFusion(in_channels=16, out_channels=32)
    output = model(x1, x2)

    print("输入形状:", x1.shape, x2.shape)  # 输出: torch.Size([2, 16, 32, 64, 64])
    print("输出形状:", output.shape)  # 输出: torch.Size([2, 32, 32, 64, 64])