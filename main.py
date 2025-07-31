
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import sys
from scipy import stats
from tqdm import tqdm
import argparse
import numpy as np
import glob
from PIL import Image
import datetime
import pickle as pkl
import logging
from models import InceptionI3d,DAE
from dataloader import load_image_train,load_image,VideoDataset,get_dataloaders
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from config import get_parser
from util import get_logger,log_and_print,loss_function,loss_function_v2
from feature_extracor import extract_i3d_features_10,extract_tcn_features
from X3D import *
sys.path.append('../')
torch.backends.cudnn.enabled = True
i3d_pretrained_path = r"/home/lab1015/programmes/lfydir/data/rgb_i3d_pretrained.pt"
x3d_pretrained_path = r"/home/lab1015/programmes/lfydir/Trampoline_experiment/X3D_M_extract_features.pth"
feature_dim = 1024

if __name__ == '__main__':

    args = get_parser().parse_known_args()[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp_folder = f'./{timestamp}'
    if not os.path.exists(timestamp_folder):
        os.makedirs(timestamp_folder)

    if not os.path.exists(f'./{timestamp_folder}/exp'):
        os.mkdir(f'./{timestamp_folder}/exp')
    if not os.path.exists('./{timestamp_folder}/rho'):
        os.mkdir(f'./{timestamp_folder}/rho')
    torch.cuda.empty_cache()

    base_logger = get_logger(f'{timestamp_folder}//exp//DAE.log', args.log_info)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    torch.cuda.set_device(3)
    #i3d = InceptionI3d().cuda()
    #i3d.load_state_dict(torch.load(i3d_pretrained_path))

    x3d = create_x3d(
            input_clip_length=16, 
            input_crop_size=224, 
            width_factor=2.0,
            depth_factor=2.2,
            head_dim_out=2048,
            model_num_class=400,  
    ) # X3D_M
    pretrained_path = "X3D_M_extract_features.pth"
    state_dict = torch.load(pretrained_path)
    x3d.load_state_dict(state_dict)
    x3d.cuda()
  
    model = ECLA().cuda()
    dataloaders = get_dataloaders(args)

    optimizer = torch.optim.Adam([*x3d.parameters()] + [*ECLA.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)
    epoch_best = 0
    rho_epoch_best = 0
    rho_best = 0.85

    train_loop(x3d,model,dataloaders['train'],optimizer,epoch_best,rho_epoch_best,base_logger)
    test_loop(x3d,model,dataloaders['test'],optimizer,epoch_best,rho_epoch_best,base_logger)
 
