# --- Imports --- #
import torch.utils.data as data
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import imghdr
import random
import torch
import numpy as np 
from basicsr.utils import DiffJPEG, USMSharp
from skimage import io, color 
import PIL
import torchvision
 
# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        torch.multiprocessing.set_start_method('spawn', force=True)
        
        train_list = train_data_dir +'data_list.txt'  #'datalist.txt''train_list_recap.txt' 'fitered_trainingdata.txt'
        with open(train_list) as f:
            contents = f.readlines()
            lowlight_names = [i.strip() for i in contents]
            gt_names = lowlight_names#[i.split('_')[0] for i in lowlight_names]
             
        self.lowlight_names = lowlight_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.size_w = crop_size[0]
        self.size_h = crop_size[1]
        self.train_data_dir = train_data_dir
    def get_images(self, index):
        lowlight_name = self.lowlight_names[index]
        gt_name = self.gt_names[index]
 

        lowlight = Image.open(self.train_data_dir + 'input/' + lowlight_name).convert('RGB') #'input_unprocess_aligned/' v
        clear = Image.open(self.train_data_dir + 'gt/' + gt_name ).convert('RGB')  #'gt_unprocess_aligned/''high/'

        

        if not isinstance(self.crop_size,str):
            i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(self.size_w,self.size_h))
            # i,j,h,w=tfs.RandomCrop.get_params(lowlight,output_size=(2160,3840))
            lowlight=FF.crop(lowlight,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w) 


        data, target=self.augData(lowlight.convert("RGB") ,clear.convert("RGB") )

        return data, target #, lowlight.resize((width/8, height/8)),gt.resize((width/8, height/8))#,factor
    def augData(self,data,target):
        #if self.train:
        if 1:  
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data) 
        target=tfs.ToTensor()(target)   
     
        return  data, target
    def __getitem__(self, index):
        res = self.get_images(index) 
        return res  

    def __len__(self):  
        return len(self.lowlight_names)


    def cutblur(self, im1, im2, prob=1.0, alpha=1.0):
        if im1.size() != im2.size():
            raise ValueError("im1 and im2 have to be the same resolution.")

        if alpha <= 0 or np.random.rand(1) >= prob:
            return im1, im2

        cut_ratio = np.random.randn()* 0.1+ alpha

        h, w = im2.size(0), im2.size(1)
        ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
        cy = np.random.randint(0, h-ch+1)
        cx = np.random.randint(0, w-cw+1) 

    # apply CutBlur to inside or outside
        if np.random.random() > 0.3: #0.5
            im2[cy:cy+ch, cx:cx+cw,:] = im1[cy:cy+ch, cx:cx+cw,:]
        
        return im1, im2
        
    def tensor_to_image(self,tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)