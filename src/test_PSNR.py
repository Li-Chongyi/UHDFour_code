# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data import ValData
from utils import validation_PSNR, generate_filelist
from thop import profile
import os
from torchsummaryX import summary
# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='PyTorch implementation of dehamer from Li et al. (2022)')
parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['UHD', 'LOLv1', 'LOLv2','our_test'], default='our_test')
parser.add_argument('-t', '--test-image-dir', help='test images path', default='./data/classic_test_image/')
parser.add_argument('-c', '--ckpts-dir', help='ckpts path', default='./ckpts/UHD_checkpoint.pt')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
args = parser.parse_args()


val_batch_size = args.val_batch_size
dataset_name = args.dataset_name
# import pdb;pdb.set_trace()
# --- Set dataset-specific hyper-parameters  --- #
if dataset_name == 'UHD':
    val_data_dir = './data/UHD-LL/testing_set/'
    ckpts_dir = './ckpts/UHD_checkpoint.pt'
elif dataset_name == 'LOLv1': 
    val_data_dir = './data/LOL-v1/eval15/'
    ckpts_dir = './ckpts/LOLv1_checkpoint.pt'
elif dataset_name == 'LOLv2': 
    val_data_dir = './data/LOL-v2/Test/'
    ckpts_dir = './ckpts/LOLv2_checkpoint.pt'
else:
    val_data_dir = args.test_image_dir
    ckpts_dir =  args.ckpts_dir

# prepare .txt file
if not os.path.exists(os.path.join(val_data_dir, 'data_list.txt')):
    generate_filelist(val_data_dir, valid=True)
# --- Gpu device --- # 
device_ids =  [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(dataset_name,val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Define the network --- #
if dataset_name == 'LOLv1' or dataset_name == 'LOLv2':
    from EnhanceN_arch_LOL import InteractNet as UHD_Net
else:
    from EnhanceN_arch import InteractNet as UHD_Net
net = UHD_Net()

  
# --- Multi-GPU --- # 
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)
net.load_state_dict(torch.load(ckpts_dir), strict=False)


# --- Use the evaluation model in testing --- #
net.eval() 
print('--- Testing starts! ---') 
start_time = time.time()
val_psnr, val_ssim = validation_PSNR(net, val_data_loader, device, dataset_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))