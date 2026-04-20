import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 注意力卷积层（输入含原始特征+平均融合特征）
        self.conv_atten_1 = nn.Conv3d(in_channels * 2, 1, kernel_size=1)
        self.conv_atten_2 = nn.Conv3d(in_channels * 2, 1, kernel_size=1)

        # 调整最终卷积层输入通道数（只需处理加权融合结果）
        # self.conv_fuse = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, feature1, feature2):
        # 平均融合特征计算
        avg_fused = (feature1 + feature2) / 2

        # 拼接原始特征与平均融合特征
        concat_feature1 = torch.cat([feature1, avg_fused], dim=1)
        concat_feature2 = torch.cat([feature2, avg_fused], dim=1)

        # 生成注意力图
        atten_map1 = self.conv_atten_1(concat_feature1)
        atten_map2 = self.conv_atten_2(concat_feature2)

        # 注意力权重归一化
        atten_weights = torch.cat([atten_map1, atten_map2], dim=1)
        atten_weights = F.softmax(atten_weights, dim=1)
        atten_weight1, atten_weight2 = torch.chunk(atten_weights, 2, dim=1)

        # print(atten_weight2.shape)
        # 加权特征融合（直接作为最终特征）
        weighted_fused = (feature1 * atten_weight1) + (feature2 * atten_weight2)

        # print(weighted_fused.shape)

        # 直接对加权结果进行卷积（不再拼接avg_fused）
        return weighted_fused


