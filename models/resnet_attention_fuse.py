import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from models.fused import EnhancedAttentionFusion3D
from models.fused import DualModalAttention
from models.cross_attention import CrossAttentionFusion
from models.DualPathFusion import DualPathFusion
from models.fusion_text_img_method import AddFusion, CatFusion, CrossAttnFusion
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def fuse_image_text_features(image_features, text_features):
    """
    将图像特征和文本特征融合为一个新的特征向量。

    参数：
    - image_features: 形状为 [batch_size, channels, height, width] 的图像特征。
    - text_features: 形状为 [batch_size, num_texts, text_feature_dim] 的文本特征。

    返回：
    - 融合后的特征，形状为 [batch_size, image_feature_size + num_texts * text_feature_dim]。
    """
    # 展平图像特征为 [batch_size, channels * height * width]
    image_flattened = image_features.view(image_features.size(0), -1)

    # 将文本特征从 [batch_size, num_texts, text_feature_dim] 转换为 [batch_size, num_texts * text_feature_dim]
    text_features_flattened = text_features.view(text_features.size(0),
                                                 -1)  # [batch_size, num_texts * text_feature_dim]

    # 拼接图像特征和文本特征
    combined_features = torch.cat((image_flattened, text_features_flattened), dim=1)  # 在维度1上拼接

    return combined_features

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


import torch
import torch.nn as nn


class FeatureFusionClassifier(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()

        # 文本特征长度维度降维到 dmodel：用 mean pooling 方式示范
        # 如果你有别的降维方式可以替换
        self.text_pool = nn.AdaptiveAvgPool1d(1)  # 对 length 做平均池化

        # 融合后的总维度，3个 dmodel 拼接
        fusion_dim = 512  + 768
        hidden_dim = 512
        # 二分类和三分类的线性层
        self.cls2 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.cls3 = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, text_feat, img_feat1):
        # text_feat: (B, slength, dmodel)
        # img_feat1, img_feat2: (B, dmodel)

        # 先对文本特征通过平均池化去 length 维度
        # 先变换维度以适配 AdaptiveAvgPool1d
        # text_feat.permute(0,2,1): (B, dmodel, slength)
        pooled_text = self.text_pool(text_feat.permute(0, 2, 1))  # (B, dmodel, 1)
        pooled_text = pooled_text.squeeze(-1)  # (B, dmodel)

        # concat 三个特征 (B, 3*dmodel)
        fusion_feat = torch.cat([pooled_text, img_feat1], dim=1)

        # 分别经过二分类和三分类全连接层
        out2 = self.cls2(fusion_feat)  # (B, 1)
        out3 = self.cls3(fusion_feat)  # (B, 3)

        return out2, out3


# 举个例子：
# if __name__ == '__main__':
#     batch_size, slength, dmodel = 32, 10, 768
#     text_feat = torch.randn(batch_size, slength, dmodel)
#     img_feat1 = torch.randn(batch_size, dmodel)
#     img_feat2 = torch.randn(batch_size, dmodel)
#
#     model = FeatureFusionClassifier(dmodel)
#     out2, out3 = model(text_feat, img_feat1, img_feat2)
#
#     print(out2.shape)  # torch.Size([32, 2])
#     print(out3.shape)  # torch.Size([32, 3])


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 fusion_method=None,
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,  # 这里是输入图像特征的channel，如果是DWI和FLAIR叠到一起，这里就是2
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.fusion = EnhancedAttentionFusion3D()  # 测试不同reduction值
        # self.fusion_flat  = CrossAttentionWithGating(512, 8)
        self.fusion_attention = DualPathFusion(in_channels=512)
        self.fusion_img_text_add = AddFusion()
        self.fusion_img_text_cat = CatFusion()
        self.fusion_img_text_ca = CrossAttnFusion()
        self.fusion_method = fusion_method
        text_dim = 768
        C, D, H, W = 512, 3, 28, 28

        self.model_text_fusion = CrossAttentionFusion(
            text_dim=text_dim,
            img_channels=C,
            depth=D,
            height=H,
            width=W,
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        # 这个是分割头的反卷积代码
        self.conv_seg = nn.Sequential(
                                        nn.ConvTranspose3d(
                                        512 * block.expansion,
                                        32,
                                        2,
                                        stride=2
                                        ),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        32,
                                        kernel_size=3,
                                        stride=(1, 1, 1),
                                        padding=(1, 1, 1),
                                        bias=False),
                                        nn.BatchNorm3d(32),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(
                                        32,
                                        2,
                                        kernel_size=1,
                                        stride=(1, 1, 1),
                                        bias=False)
                                        )
        # 这个是二分类的分类头代码
        # self.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        #                                nn.Linear(in_features=512, out_features=1, bias=True))
        # 这个是多分类的分类头代码
        self.conv_seg1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                      nn.Linear(in_features=512, out_features=1, bias=True))

        self.conv_seg2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                       nn.Linear(in_features=512, out_features=3, bias=True))
        self.conv_1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                       )

        self.conv_2 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
                                       )

        self.conv_flat = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten())
        self.linear1 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.linear2 = nn.Linear(in_features=512, out_features=3, bias=True)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward_first(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # print(f'x1-shape:{x.shape}')
        x = self.layer2(x)
        # print(f'x2-shape:{x.shape}')
        x = self.layer3(x)
        # print(f'x3-shape:{x.shape}')
        x = self.layer4(x)
        # x = self.conv_flat(x)
        return x
    def forward(self, x1):
        print(x1.shape)
        # x1 = x1 * label_mask
        # x2 = x2 * label_mask
        # print(x1.shape)

        # dwi_text_featurs = dwi_text_featurs.to(x1.device)
        # flair_text_features = flair_text_features.to(x1.device)

        x1 = self.forward_first(x1)
        # print(x1.shape)
        # x2 = self.forward_first(x1)
        # print(f'dwi-shape:{x1.shape},flair-shape:{x2.shape} dwi-text-shape:{dwi_text_featurs.shape} flair-text-shape:{flair_text_features.shape}')
        # x1 = self.model_text_fusion(dwi_text_featurs, x1)
        # x2 = self.model_text_fusion(flair_text_features, x2)
        # print('dwi-shape:',dwi_text_featurs.shape)
        # print(f'x1-shape:{x1.shape}')
        # img_fusion_x = self.fusion_attention(x1, x2)
        # text_fusion_x = torch.cat([dwi_text_featurs, flair_text_features], dim=1)



        # print(f'x-shape:{x1.shape}')
        # x = self.conv_1(x)
        # fuse_model = FeatureFusionClassifier().to(x1.device)
        # print(f'text-shape:{text_featurs.shape}')
        # cls_2, cls_3 = fuse_model(text_featurs, x)
        # cls_2 = self.conv_seg1(x)
        # cls_3 = self.conv_seg2(x)
        cls_2 = self.conv_seg1(x1)
        cls_3 = self.conv_seg2(x1)
        # print(cls_2.shape)
        # if self.fusion_method == "add":
        #     _, cls_2, cls_3 = self.fusion_img_text_add(img_fusion_x, text_fusion_x)
        # elif self.fusion_method == 'cat':
        #     _, cls_2, cls_3 = self.fusion_img_text_cat(img_fusion_x, text_fusion_x)
        # else:
        #     _, cls_2, cls_3 = self.fusion_img_text_ca(img_fusion_x, text_fusion_x)
        # x3 = self.conv_seg(x)
        # x = torch.sigmoid_(x)
        # x = self.fc(x)  这里是进行分类的代码，需要根据需求进行激活使用

        return cls_2, cls_3


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
