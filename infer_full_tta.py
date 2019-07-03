import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import os
from tqdm import tqdm
import pickle as pkl
import cv2

import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '/mmdetection/'))


import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import os
from tqdm import tqdm
import pickle as pkl
import cv2
from mmdet.ops import nms
import numpy as np
import time
import sys
sys.path.insert(0,'data_utils')
import boxes_mask as box_utils

is_vis = False

full_sz=[4000,4000,4000,4000,3000,4000,4000]
im_sz = [1600,1700,1800,1800,1800,2000,2000]
stretch = [1.1,1.5,1,1.2,1.3,1,1.4]
flips=[0,1,0,1,1,1,0]

# im_sz = [1800,]
# stretch = [1.3]
# flips=[1]

model_name='configs/cascade_rcnn_r50_fpn_full.py'
model_weight='weights/epoch_50.pth'
dir_tst='data/test_images_2/'
for z in range(len(im_sz)):
    dir_out='result/detect_all/'
    name_out=dir_out+'cascade50_full_%d_%d' % (im_sz[z],10*stretch[z])
    # if is_vis:
    os.makedirs(dir_out,exist_ok=True)
    names = os.listdir(dir_tst)
    cfg = mmcv.Config.fromfile(model_name)
    cfg.model.pretrained = None
    cfg['data']['test']['flip_ratio'] = flips[z]
    cfg['data']['test']['img_scale']=(full_sz[z], im_sz[z])


    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, model_weight)

    data_out={}
    count=0
    for name in tqdm(os.listdir(dir_tst)):
        count+=1
        path=dir_tst+name
        img = mmcv.imread(path)
        sz=img.shape

        img=cv2.resize(img,(int(stretch[z]*sz[1]),sz[0]))
        result = inference_detector(model, img, cfg)
        data_out[name]=result
        if stretch[z]!=1:
            for k in range(len(result)):
                if len(result[k])>0:
                    result[k][:,(0,2)]/=stretch[z]
        # show_result(img, result[0])
        if count % 200==0:
            pkl.dump(data_out, open(dir_out + '%d.pkl' % count, 'wb'))
            data_out={}

        if is_vis:
            img=cv2.imread(path)

            n_ob=0
            for k in range(len(result)):
                color=(0,255,0)
                for bb in result[k]:
                    n_ob+=1
                    if bb[4] > 0.1:
                        bb = bb.astype(int)
                        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 3)
            print(n_ob)
            os.makedirs(dir_out,exist_ok=True)
            im_path_out = dir_out + '/' + name + '.jpg'
            print(im_path_out)
            cv2.imwrite(im_path_out, img)
    pkl.dump(data_out,open(name_out+'.pkl','wb'))

