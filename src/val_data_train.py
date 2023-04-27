# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Validation/test dataset --- #
class ValData_train(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'data_list.txt'#'final_test_datalist.txt'eval15/datalist.txt
        with open(val_list) as f:
            contents = f.readlines() 
            lowlight_names = [i.strip() for i in contents]
            gt_names = lowlight_names#[i.split('_')[0] + '.png' for i in lowlight_names]
 
        self.lowlight_names = lowlight_names
        self.gt_names = gt_names 
        self.val_data_dir = val_data_dir
        self.data_list=val_list
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index] 
        gt_name = self.gt_names[index]
        lowlight_img = Image.open(self.val_data_dir + 'input/' + lowlight_name)#eval15/low/
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name) #eval15/high/
        transform_lowlight = Compose([ToTensor()])
        transform_gt = Compose([ToTensor()])     
        lowlight = transform_lowlight(lowlight_img)
        gt = transform_gt(gt_img)       
        return lowlight, gt,lowlight_name #

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self): 
        return len(self.lowlight_names)
 