import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import torch
import scipy.ndimage as ndimage
from skimage.transform import resize
from matplotlib import pyplot as plt
import nibabel
from model import generate_model
from setting import parse_opts

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 加载数据 label 1
MRI_path = "/extra/shilei/tongji_newdata/Train/DWI/Partial-5.nii.gz"
flair_path = "/extra/shilei/tongji_newdata/Train/FLAIRS/Partial-5.nii.gz"
# MRI_path = "/extra/shilei/tongji_newdata/Train/DWI/FS00107511.nii.gz"
# flair_path = "/extra/shilei/tongji_newdata/Train/FLAIRS/FS00107511.nii.gz"
# MRI_path = "/extra/shilei/tongji_newdata/Test/DWI/Partial-104.nii.gz"
# flair_path = "/extra/shilei/tongji_newdata/Test/FLAIRS/Partial-104.nii.gz"
###---导入模型----###
# MRI_path = 'data.nii.gz'
model_path = '/extra/shilei/tongji_newdata/Med3D/09-11-best/fold_1_best.pth'
# 读取数据
MRI = nibabel.load(MRI_path)
MRI_2 = nibabel.load(flair_path)
MRI_array = MRI.get_fdata()
MRI_array = MRI_array.astype('float32')
MRI_2_array = MRI_2.get_fdata()
MRI_2_array = MRI_2_array.astype('float32')
# data preprocess
max_value = MRI_array.max()
MRI_array = MRI_array / max_value
max_2_value = MRI_2_array.max()
MRI_2_array = MRI_2_array / max_value
MRI_tensor = torch.FloatTensor(MRI_array).unsqueeze(0).unsqueeze(0)
MRI_2_tensor = torch.FloatTensor(MRI_array).unsqueeze(0).unsqueeze(0)
# print('origin MRI shape: ', MRI_tensor.shape)
MRI_tensor = torch.concat([MRI_tensor, MRI_2_tensor], axis = 1)
MRI_tensor = MRI_tensor.cuda()
sets = parse_opts()
model, parameters = generate_model(sets)
grad_model = model.cuda()


# use register_forward_hook() to gain the features map
class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        # 获取model.features中某一层的output

    def hook_fn(self, module, MRI_tensorut, output):
        self.features = output.cpu()

    def remove(self):  ## remove hook
        self.hook.remove()


# load model
grad_model.load_state_dict(torch.load(model_path), strict = False)
grad_model.eval()

# Instantiate, get the i_th layer (second argument) of each convolution
# conv_out = LayerActivations(grad_model.Conv2.conv,0) # train
conv_out = LayerActivations(grad_model.module.layer4, 0)  # test

output = grad_model(MRI_tensor)
cam = conv_out.features  # gain the ith output
# cam = output # gain the latest output
conv_out.remove  # delete the hook

###---lAYER-Name--to-visualize--###
# Create a graph that outputs target convolution and output
print('cam.shape1', cam.shape)
cam = cam.cpu().detach().numpy().squeeze()
print('cam.shape2', cam.shape)
cam = cam[1]
print('cam.shape3', cam.shape)

capi = resize(cam, (MRI_tensor.shape[2], MRI_tensor.shape[3], MRI_tensor.shape[4]))
# print(capi.shape)
capi = np.maximum(capi, 0)
heatmap = (capi - capi.min()) / (capi.max() - capi.min())
f, axarr = plt.subplots(3, 3, figsize=(12, 12))

f.suptitle('CAM_3D_medical_image', fontsize=30)

axial_slice_count = 11
coronal_slice_count = 150
sagittal_slice_count = 150

sagittal_MRI_img = np.squeeze(MRI_2_array[sagittal_slice_count, :, :])
sagittal_grad_cmap_img = np.squeeze(heatmap[sagittal_slice_count, :, :])
print("MRI-shape:", MRI_array.shape)
axial_MRI_img = np.squeeze(MRI_2_array[:, :, axial_slice_count])
axial_grad_cmap_img = np.squeeze(heatmap[:, :, axial_slice_count])

coronal_MRI_img = np.squeeze(MRI_2_array[:, coronal_slice_count, :])
coronal_grad_cmap_img = np.squeeze(heatmap[:, coronal_slice_count, :])

# 关键修改：添加翻转操作
# 方法1：水平翻转（左右镜像）
sagittal_MRI_img = np.fliplr(sagittal_MRI_img)
sagittal_grad_cmap_img = np.fliplr(sagittal_grad_cmap_img)
axial_MRI_img = np.fliplr(axial_MRI_img)
axial_grad_cmap_img = np.fliplr(axial_grad_cmap_img)
coronal_MRI_img = np.fliplr(coronal_MRI_img)
coronal_grad_cmap_img = np.fliplr(coronal_grad_cmap_img)

# 方法2：如果需要垂直翻转（上下翻转），取消下面的注释
# sagittal_MRI_img = np.flipud(sagittal_MRI_img)
# sagittal_grad_cmap_img = np.flipud(sagittal_grad_cmap_img)
# axial_MRI_img = np.flipud(axial_MRI_img)
# axial_grad_cmap_img = np.flipud(axial_grad_cmap_img)
# coronal_MRI_img = np.flipud(coronal_MRI_img)
# coronal_grad_cmap_img = np.flipud(coronal_grad_cmap_img)

# 方法3：如果需要进行180度旋转（水平+垂直翻转），取消下面的注释
# sagittal_MRI_img = np.rot90(sagittal_MRI_img, 2)  # 旋转180度
# sagittal_grad_cmap_img = np.rot90(sagittal_grad_cmap_img, 2)
# axial_MRI_img = np.rot90(axial_MRI_img, 2)
# axial_grad_cmap_img = np.rot90(axial_grad_cmap_img, 2)
# coronal_MRI_img = np.rot90(coronal_MRI_img, 2)
# coronal_grad_cmap_img = np.rot90(coronal_grad_cmap_img, 2)

# Sagittal view
img_plot = axarr[0, 0].imshow(np.rot90(sagittal_MRI_img, 1), cmap='gray')
axarr[0, 0].axis('off')
axarr[0, 0].set_title('Sagittal MRI', fontsize=25)

img_plot = axarr[0, 1].imshow(np.rot90(sagittal_grad_cmap_img, 1), cmap='jet')
axarr[0, 1].axis('off')
axarr[0, 1].set_title('Weight-CAM', fontsize=25)

# Zoom in ten times to make the weight map smoother
sagittal_MRI_img = ndimage.zoom(sagittal_MRI_img, (1, 1), order=3)
# Overlay the weight map with the original image
sagittal_overlay = cv2.addWeighted(sagittal_MRI_img, 0.3, sagittal_grad_cmap_img, 0.6, 0)

img_plot = axarr[0, 2].imshow(np.rot90(sagittal_overlay, 1), cmap='jet')
axarr[0, 2].axis('off')
axarr[0, 2].set_title('Overlay', fontsize=25)

# Axial view
img_plot = axarr[1, 0].imshow(np.rot90(axial_MRI_img, 1), cmap='gray')
axarr[1, 0].axis('off')
axarr[1, 0].set_title('Axial MRI', fontsize=25)

img_plot = axarr[1, 1].imshow(np.rot90(axial_grad_cmap_img, 1), cmap='jet')
axarr[1, 1].axis('off')
axarr[1, 1].set_title('Weight-CAM', fontsize=25)

axial_MRI_img = ndimage.zoom(axial_MRI_img, (1, 1), order=3)
axial_overlay = cv2.addWeighted(axial_MRI_img, 0.3, axial_grad_cmap_img, 0.6, 0)

img_plot = axarr[1, 2].imshow(np.rot90(axial_overlay, 1), cmap='jet')
axarr[1, 2].axis('off')
axarr[1, 2].set_title('Overlay', fontsize=25)

# coronal view
img_plot = axarr[2, 0].imshow(np.rot90(coronal_MRI_img, 1), cmap='gray')
axarr[2, 0].axis('off')
axarr[2, 0].set_title('Coronal MRI', fontsize=50)

img_plot = axarr[2, 1].imshow(np.rot90(coronal_grad_cmap_img, 1), cmap='jet')
axarr[2, 1].axis('off')
axarr[2, 1].set_title('Weight-CAM', fontsize=50)

coronal_ct_img = ndimage.zoom(coronal_MRI_img, (1, 1), order=3)
Coronal_overlay = cv2.addWeighted(coronal_ct_img, 0.3, coronal_grad_cmap_img, 0.6, 0)

img_plot = axarr[2, 2].imshow(np.rot90(Coronal_overlay, 1), cmap='jet')
axarr[2, 2].axis('off')
axarr[2, 2].set_title('Overlay', fontsize=50)

plt.colorbar(img_plot, shrink=0.5)  # color bar if need

plt.savefig('./grad_cam_res/Partial-5-12.png')
print("save successful")