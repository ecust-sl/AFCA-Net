'''
Dataset for training
Written by Whalechen
'''

import math
import os
import random

import numpy as np
from torch.utils.data import Dataset
import nibabel
import torch
from scipy import ndimage
import shlex
class BrainS18Dataset(Dataset):

    def __init__(self, root_dir, img_list, sets):
        with open(img_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        print("Processing {} datas".format(len(self.img_list)))
        # print(f'root-dir:{root_dir}')
        self.root_dir = root_dir
        self.input_D = sets.input_D
        self.input_H = sets.input_H
        self.input_W = sets.input_W
        self.phase = sets.phase

    def __nii2tensorarray__(self, data):
        [z, y, x] = data.shape
        new_data = np.reshape(data, [1, z, y, x])
        new_data = new_data.astype("float32")
            
        return new_data
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        
        if self.phase == "train":
            # read image and labels
            ith_info = shlex.split(self.img_list[idx])
            # print(f'root-----------------------------------------{self.root_dir}')

            img_name = os.path.join(self.root_dir, ith_info[0])
            flair_name = os.path.join(self.root_dir, ith_info[1])
            label_name = os.path.join(self.root_dir, ith_info[2])
            raw = ith_info[6]
            split_token = "The FLAIR"
            idx = raw.find(split_token)
            if idx != -1:
                dwi_text = raw[:idx].strip()  # “The DWI image shows no …”
                flair_text = raw[idx:].strip()  # “The FLAIR image shows no …”
            else:
                # 万一没找到 "The FLAIR"，就当成整句 DWI，FLAIR 置空
                dwi_text, flair_text = raw, ""
            text = ith_info[5]
            # dwi_text = ith_info[6]
            # flair_text = ith_info[7]
            self.label_name = label_name
            class_array = int(ith_info[3])
            class_array_2 = int(ith_info[4])
            class_array = torch.tensor(class_array, dtype=torch.long)
            class_array_2 = torch.tensor(class_array_2, dtype=torch.long)
            # print(f'image-name-----------------------------------------------{img_name}')
            assert os.path.isfile(img_name)
            assert os.path.isfile(label_name)
            assert os.path.isfile(flair_name)
            img = nibabel.load(img_name)  # We have transposed the data from WHD format to DHW
            assert img is not None
            flair = nibabel.load(flair_name)
            assert flair is not None
            mask = nibabel.load(label_name)
            assert mask is not None
            
            # data processing
            img_array, flair_array, mask_array = self.__training_data_process__(img, flair, mask)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            flair_array = self.__nii2tensorarray__(flair_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            assert img_array.shape ==  mask_array.shape, "img shape:{} is not equal to mask shape:{}".format(img_array.shape, mask_array.shape)
            assert img_array.shape == flair_array.shape, "dwi shape:{} is not equal to flair shape:{}".format(img_array.shape, flair_array.shape)
            return img_array, flair_array, mask_array, class_array, class_array_2, text, dwi_text, flair_text
        
        elif self.phase == "test":
            # read image
            ith_info = shlex.split(self.img_list[idx])
            img_name = os.path.join(self.root_dir, ith_info[0])
            flair_name = os.path.join(self.root_dir, ith_info[1])
            label_name = os.path.join(self.root_dir, ith_info[2])
            class_array = int(ith_info[3])
            class_array = torch.tensor(class_array, dtype=torch.long)
            class_array_2 = int(ith_info[4])
            class_array = torch.tensor(class_array_2, dtype=torch.long)
            raw = ith_info[6]
            split_token = "The FLAIR"
            idx = raw.find(split_token)
            if idx != -1:
                dwi_text = raw[:idx].strip()  # “The DWI image shows no …”
                flair_text = raw[idx:].strip()  # “The FLAIR image shows no …”
            else:
                # 万一没找到 "The FLAIR"，就当成整句 DWI，FLAIR 置空
                dwi_text, flair_text = raw, ""
            text = ith_info[5]
            # dwi_test = ith_info[6]
            # flair_test = ith_info[7]
            # print(img_name)
            assert os.path.isfile(img_name)
            img = nibabel.load(img_name)
            assert img is not None
            assert os.path.isfile(flair_name)
            flair = nibabel.load(flair_name)
            assert flair is not None
            assert os.path.isfile(label_name)
            label = nibabel.load(label_name)
            assert label is not None

            # data processing
            img_array, flair_array, mask_array = self.__testing_data_process__(img, flair, label)

            # 2 tensor array
            img_array = self.__nii2tensorarray__(img_array)
            flair_array =self.__nii2tensorarray__(flair_array)
            mask_array = self.__nii2tensorarray__(mask_array)

            return img_array, flair_array, mask_array, class_array, class_array_2, text, dwi_text, flair_text
            

    def __drop_invalid_range__(self, volume, flair, label=None):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        if label is not None:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w], flair[min_z:max_z, min_h:max_h, min_w:max_w], label[min_z:max_z, min_h:max_h, min_w:max_w]
        else:
            return volume[min_z:max_z, min_h:max_h, min_w:max_w]


    def __random_center_crop__(self, data, flair_data, label):
        from random import random
        """
        Random crop
        """
        # print(f'label_name:{self.label_name}')
        target_indexs = np.where(label>0)
        if all(idx.size == 0 for idx in target_indexs):
            print(f'image-name:{self.label_name}')
            print(f'size::{len(target_indexs)}')
            return None
            # raise ValueError("Error: All indices are empty (no voxels meet the condition).")
        [img_d, img_h, img_w] = data.shape
        [max_D, max_H, max_W] = np.max(np.array(target_indexs), axis=1)
        [min_D, min_H, min_W] = np.min(np.array(target_indexs), axis=1)
        [target_depth, target_height, target_width] = np.array([max_D, max_H, max_W]) - np.array([min_D, min_H, min_W])
        Z_min = int((min_D - target_depth*1.0/2) * random())
        Y_min = int((min_H - target_height*1.0/2) * random())
        X_min = int((min_W - target_width*1.0/2) * random())
        
        Z_max = int(img_d - ((img_d - (max_D + target_depth*1.0/2)) * random()))
        Y_max = int(img_h - ((img_h - (max_H + target_height*1.0/2)) * random()))
        X_max = int(img_w - ((img_w - (max_W + target_width*1.0/2)) * random()))
       
        Z_min = np.max([0, Z_min])
        Y_min = np.max([0, Y_min])
        X_min = np.max([0, X_min])

        Z_max = np.min([img_d, Z_max])
        Y_max = np.min([img_h, Y_max])
        X_max = np.min([img_w, X_max])
 
        Z_min = int(Z_min)
        Y_min = int(Y_min)
        X_min = int(X_min)
        
        Z_max = int(Z_max)
        Y_max = int(Y_max)
        X_max = int(X_max)

        return data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], flair_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], label[Z_min: Z_max, Y_min: Y_max, X_min: X_max]



    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        out_random = np.random.normal(0, 1, size = volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out

    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data


    def __crop_data__(self, data, flair_data, label):
        """
        Random crop with different methods:
        """ 
        # random center crop
        data, flair_data, label = self.__random_center_crop__ (data, flair_data, label)
        
        return data, flair_data, label

    def __training_data_process__(self, data, flair_data, label):
        # crop data according net input size
        data = data.get_fdata()
        flair_data = flair_data.get_fdata()
        label = label.get_fdata()
        
        # drop out the invalid range
        data, flair_data, label = self.__drop_invalid_range__(data, flair_data, label)
        
        # crop data
        data, flair_data, label = self.__crop_data__(data, flair_data, label)

        # resize data
        data = self.__resize_data__(data)
        flair_data = self.__resize_data__(flair_data)
        label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        flair_data = self.__itensity_normalize_one_volume__(flair_data)

        return data, flair_data, label


    def __testing_data_process__(self, data, flair_data, label):
        # crop data according net input size
        data = data.get_fdata()
        flair_data = flair_data.get_fdata()
        label = label.get_fdata()

        # drop out the invalid range
        data, flair_data, label = self.__drop_invalid_range__(data, flair_data, label)

        # crop data
        data, flair_data, label = self.__crop_data__(data, flair_data, label)

        # resize data
        data = self.__resize_data__(data)
        flair_data = self.__resize_data__(flair_data)
        label = self.__resize_data__(label)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)
        flair_data = self.__itensity_normalize_one_volume__(flair_data)

        return data, flair_data, label