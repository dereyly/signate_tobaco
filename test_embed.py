import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import model.multiclass_models as multi_models
#from  mafat_release.utils.dataset import *
from tqdm import tqdm
# from collections import OrderedDict
from collections import defaultdict
import pickle as pkl
from utils.dataset_pkl import  DatasetPkl
from utils.data_transforms import data_transforms
import numpy as np
import json
seed=2323
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

data_root = 'data/imgs_tst_2/'
dir_weights= 'weights/'
name_weights = ['se_resnext50_32x4d_multi_fold_4__3_.pth.tar',]
tta_num=5
dir_out='result/features'
os.makedirs(dir_out,exist_ok=True)
path_out=dir_out+'/feats_test_tta_%dx%d.pkl' % (tta_num, len(name_weights))

predict_batch_size = 64

trans_list=['val_em']
data_names=['test_crop']
folds=[1]
idx2cat=json.load(open('configs/idx2cat.json','r'))
remap=pkl.load(open('configs/image_folder_idx.pkl','rb'))
invert={}
for key,val in remap.items():
    invert[val]=int(key)

n_models=len(name_weights)

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    num_classes=[224]
    models=[None]

    models[0] = multi_models.__dict__['se_resnext50_32x4d_multi'](pretrained=False, num_classes=num_classes, is_embed=True)

    for z in range(n_models):
        if models[z] is not None:
            models[z] = torch.nn.DataParallel(models[z]).cuda()
            path_weights = dir_weights + name_weights[z]
            if os.path.isfile(path_weights):
                print("=> loading checkpoint '{}'".format(path_weights))
                checkpoint = torch.load(path_weights)

                if 'state_dict' in checkpoint:
                    models[z].load_state_dict(checkpoint['state_dict'])
                else:
                    models[z].load_state_dict(checkpoint)
                models[z].eval()


            else:
                print("=> no checkpoint found at '{}'".format(path_weights))
                return


    feats_all = defaultdict(dict)

    for z in range(n_models):
        datas = name_weights[z].split('__')[-1][:-8]
        data_path=data_root+data_names[z]+'/'
        pkl_path=data_root+data_names[z]+'_list.pkl'
        test_dataset  = DatasetPkl(data_path,pkl_path,transform=data_transforms[trans_list[z]])
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=predict_batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=8,
                                                  pin_memory=True)

        with torch.no_grad():
            for tta_iter in range(tta_num):
                for i, data in tqdm(enumerate(test_loader)):
                    # try:
                    input = data[0]
                    meta = data[1]

                    feats = models[z](input.cuda())
                    norm = feats.norm(p=2, dim=1, keepdim=True)
                    feats_norm = feats.div(norm.expand_as(feats))
                    if z==0 and tta_iter==0:
                        for k in range(len(input)):
                            feats_all[meta[0][k]][int(meta[1][k])]=[]

                    for k in range(len(input)):
                        feats_all[meta[0][k]][int(meta[1][k])].append([meta[2][k].cpu().numpy(),feats_norm[k].cpu().clone()])



    pkl.dump(feats_all,open(path_out,'wb'))
if __name__ == '__main__':
    main()