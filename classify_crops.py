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

seed=93 #2323 #1093
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

data_root = 'data/imgs_tst_2/'

dir_weights= 'weights/'
name_weights = ['InceptionResNetV2_multi_fold_3__3_.pth.tar',
                'se_resnext50_32x4d_multi_fold_4__3_.pth.tar',
                'se_resnext50_32x4d_multi_fold_1__3_.pth.tar',
                'InceptionResNetV2_multi_fold_2__3_.pth.tar'
                ]

tta_num=25
path_out='result/res_all_tta_2_%dx%d.pkl' % (tta_num, len(name_weights))

predict_batch_size = 64

trans_list=['val_v3','val_v3','val_v3','val_v4']
data_names=['test_crop','test_crop','test_crop_sq','test_crop_sq']
n_models=len(name_weights)

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    num_classes=[224]

    models=[None for z in range(n_models)]

    models[0] = multi_models.__dict__['InceptionResNetV2_multi'](pretrained=False, num_classes=num_classes)
    models[1] = multi_models.__dict__['se_resnext50_32x4d_multi'](pretrained=False, num_classes=num_classes)
    models[2] = multi_models.__dict__['se_resnext50_32x4d_multi'](pretrained=False, num_classes=num_classes)
    models[3] = multi_models.__dict__['InceptionResNetV2_multi'](pretrained=False, num_classes=num_classes)

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





    cudnn.benchmark = True
    softmax = torch.nn.Softmax(dim=1)
    # len_dataset=11879
    cls_res_all = defaultdict(dict)


    for z in range(n_models):
        datas = name_weights[z].split('__')[-1][:-8]
        data_path=data_root+data_names[z]+'/'
        pkl_path=data_root+data_names[z]+'_list.pkl'
        test_dataset  = DatasetPkl(data_path,pkl_path,transform=data_transforms[trans_list[z]])
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=predict_batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=5,
                                                  pin_memory=True)

        with torch.no_grad():
            for tta_iter in range(tta_num):
                for i, data in tqdm(enumerate(test_loader)):

                    input = data[0]
                    meta=data[1]
                    output = models[z](input.cuda())
                    res=softmax(output[0])

                    K=1
                    if z>=2:
                        K=1.1
                    if z==0 and tta_iter==0:
                        for k in range(len(input)):
                            cls_res_all[meta[0][k]][int(meta[1][k])]=[meta[2][k].cpu().numpy(),res[k]]

                    else:
                        for k in range(len(input)):

                            cls_res_all[meta[0][k]][int(meta[1][k])][1]+=K*res[k]



    cls_out = defaultdict(dict)
    for key, val in cls_res_all.items():

        for id, res in val.items():

            cls_res=res[1].cpu().numpy()/ (tta_num*n_models)
            cls_id=cls_res.argmax()
            prob = cls_res.max()
            cls_out[key][id]=[res[0],cls_res, prob, cls_id]
    pkl.dump(cls_out,open(path_out,'wb'))
if __name__ == '__main__':
    main()