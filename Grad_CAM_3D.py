import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from grad_cam import GradCAM,show_cam_on_image
from torch import nn
import torch.nn.functional as F
def visualize_comparison(input_data, feature_map, slice_idx):
    """三维可视化对比函数"""
    # 获取原始切片
    original_slices = {
        'x': input_data[0, 0, slice_idx, :, :].numpy(),
        'y': input_data[0, 0, :, slice_idx, :].numpy(),
        'z': input_data[0, 0, :, :, slice_idx].numpy()
    }

    # 特征图预处理
    if isinstance(feature_map, np.ndarray):
        feature_map = torch.tensor(feature_map, dtype=torch.float32)
        feature_map=feature_map.unsqueeze(0)
    # 三维插值到原始尺寸
    upsampled_feature = F.interpolate(
        feature_map,
        size=tuple(input_data.shape[2:]),
        mode='trilinear',
        align_corners=False
    ).squeeze()

    # 创建可视化画布
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    plt.set_cmap('jet')

    # 遍历三个轴向
    for idx, axis in enumerate(['x', 'y', 'z']):
        # 原始切片
        axes[idx, 0].imshow(original_slices[axis], cmap='gray')
        axes[idx, 0].set_title(f'Original {axis.upper()}-Slice {slice_idx}')
        # 特征图切片
        if axis == 'x':
            feature_slice = upsampled_feature[slice_idx, :, :]
        elif axis == 'y':
            feature_slice = upsampled_feature[:, slice_idx, :]
        else:
            feature_slice = upsampled_feature[:, :, slice_idx]

        # 归一化
        feature_slice = (feature_slice - feature_slice.min()) / (feature_slice.max() - feature_slice.min())

        # 叠加可视化
        axes[idx, 1].imshow(original_slices[axis], cmap='gray')
        im = axes[idx, 1].imshow(feature_slice, alpha=0.5)
        plt.colorbar(im, ax=axes[idx, 1])
        axes[idx, 1].set_title(f'Feature Map {axis.upper()}-Slice {slice_idx}')
    plt.tight_layout()
    plt.show()
from setting import parse_opts
def main():
    sets = parse_opts()
    sets.batch_size = 16
    sets.gpu_id = 2
    model_best_path = '/extra/shilei/tongji_newdata/Med3D/09-11-best/fold_1_best.pth'
    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    model.load_state_dict(torch.load(model_best_path), strict=False)
    print(model)
    model.eval()

    # 对应特征层
    target_layers = [model.feature_extractor[2]]

    # load image
    dwi_path = "/extra/shilei/tongji_newdata/Train/DWI/FS00357800.nii.gz"
    flair_path = "/extra/shilei/tongji_newdata/Train/DWI/FS00357800.nii.gz"
    dwi_nii = nibabel.load(dwi_path)
    flair_nii = nibabel.load(flair_path)
    # 获取 numpy 数据，通常 shape 是 [H, W, D]，需转换为 [D, H, W]
    dwi_np = np.array(dwi_nii.get_fdata(), dtype=np.float32)
    flair_np = np.array(flair_nii.get_fdata(), dtype=np.float32)

    # 转换为 [D, H, W]
    dwi_np = np.transpose(dwi_np, (2, 0, 1))
    flair_np = np.transpose(flair_np, (2, 0, 1))
    # 替换对应权重
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 0
    grayscale_cam = cam(input_tensor=input_data, target_category=target_category)
    # 获取不同轴上的切片
    slice_idx=80
    visualize_comparison(input_data, grayscale_cam, slice_idx)
if __name__ == '__main__':
    main()
