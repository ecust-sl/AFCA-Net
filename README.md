# Project Code and Data Overview

## 1. Pretrained Model Link
- https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34/blob/main/resnet_34_23dataset.pth

## 2. Training Scripts for Four Fusion Methods
- `train_attention_direct.py`
- `train_img_text_add.py`
- `train_img_text_att.py`
- `train_img_text_cat.py`

## 3. Script for Generating 3D Grad-CAM Visualization
- `Grad_CAM_3D.py`

## 4. Script for Generating Heatmaps
- `draw_map.py`

## 5. Five-Fold Training Data Path
- `./KFData/kfold_data_all_final_dwi_flair`
