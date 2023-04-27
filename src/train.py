#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from UHDFour_model import UHDFour
from argparse import ArgumentParser
from train_data_aug_local import TrainData #############
from val_data import ValData
from val_data_train import ValData_train


def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of UHDFour from Li et al. (2023)')

    # Data parameters

    parser.add_argument('-d', '--dataset-name', help='name of dataset',choices=['UHD', 'LOLv1', 'LOLv2'], default='UHD')
    parser.add_argument('-t', '--train-dir', help='training set path', default='./../data/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='./../data/valid')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='./../ckpts')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--ckpt-load-path', help='start training with a pretrained model',default=None)
    parser.add_argument('--report-interval', help='batch report interval', default=1, type=int)
    parser.add_argument('-ts', '--train-size',nargs='+', help='size of train dataset',default=[512,512], type=int) 
    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.0001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=8, type=int)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('--cuda', help='use cuda', action='store_true')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_true')    
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
 
    return parser.parse_args()  
 
  
if __name__ == '__main__':    
    """Trains UHDFour."""   
     
    # Parse training parameters  
    params = parse_args()     

 
  
# --- Load training data and validation/test data --- # params.train_size
    train_loader = DataLoader(TrainData(params.train_size, params.train_dir), batch_size=params.batch_size, shuffle=True, num_workers=0) 
    valid_loader = DataLoader(ValData_train(params.valid_dir), batch_size=1, shuffle=False, num_workers=0) 
    UHDFour = UHDFour(params, trainable=True)
    UHDFour.train(train_loader, valid_loader)


