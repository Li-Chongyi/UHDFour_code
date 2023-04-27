# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import os

# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, dataset_name,val_data_dir):
        super().__init__() 
        self.dataset_name = dataset_name
        val_list = os.path.join(val_data_dir, 'data_list.txt')
        with open(val_list) as f:
            contents = f.readlines()
            lowlight_names = [i.strip() for i in contents]
            if self.dataset_name=='UHD' or self.dataset_name=='LOLv1' or self.dataset_name=='LOLv2':
                gt_names = lowlight_names #
            else:
                gt_names = None 
                print('The dataset is not included in this work.')  
        self.lowlight_names = lowlight_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        padding = 8
        # build the folder of validation/test data in our way
        if os.path.exists(os.path.join(self.val_data_dir, 'input')):
            lowlight_img = Image.open(os.path.join(self.val_data_dir, 'input', lowlight_name))
            if os.path.exists(os.path.join(self.val_data_dir, 'gt')) :
                gt_name = self.gt_names[index]
                gt_img = Image.open(os.path.join(self.val_data_dir, 'gt', gt_name)) ##   
                a = lowlight_img.size

                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = gt_img.crop((0, 0, 0 + a_1, 0+a_0))
                transform_lowlight = Compose([ToTensor()])
                transform_gt = Compose([ToTensor()])
                lowlight_img = transform_lowlight(lowlight_crop_img)
                gt_img = transform_gt(gt_crop_img)
            else: 
                # the inputs is used to calculate PSNR.
                a = lowlight_img.size
                a_0 =a[1] - np.mod(a[1],padding)
                a_1 =a[0] - np.mod(a[0],padding)            
                lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
                gt_crop_img = lowlight_crop_img
                transform_lowlight = Compose([ToTensor() ])
                transform_gt = Compose([ToTensor()])
                lowlight_img = transform_lowlight(lowlight_crop_img)
                gt_img = transform_gt(gt_crop_img) 
        # Any folder containing validation/test images
        else:
            lowlight_img = Image.open(os.path.join(self.val_data_dir, lowlight_name))
            a = lowlight_img.size
            a_0 =a[1] - np.mod(a[1],padding)
            a_1 =a[0] - np.mod(a[0],padding)            
            lowlight_crop_img = lowlight_img.crop((0, 0, 0 + a_1, 0+a_0))
            gt_crop_img = lowlight_crop_img
            transform_lowlight = Compose([ToTensor()])
            transform_gt = Compose([ToTensor()])
            lowlight_img = transform_lowlight(lowlight_crop_img)
            gt_img = transform_gt(gt_crop_img)           
        return lowlight_img, gt_img, lowlight_name


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.lowlight_names)
