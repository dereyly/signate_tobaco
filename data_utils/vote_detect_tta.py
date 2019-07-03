import pickle as pkl
import os
import numpy as np
import sys

import cv2
import boxes_mask as box_utils
import json
import time
import argparse
from mmdet.ops import nms,soft_nms
from collections import defaultdict
import cv2
print(np.__version__)
print(cv2.__version__)
num_cls = 1
nms_th=0.7
nms_all_th=0.6
cls_th = 0.1
cls_th_v2=0.24

is_border=True
# if is_border:
#     dir_data='data/test_images_2/'




start0 = time.time()

def get_data(dir_in):
    if os.path.isdir(dir_in):
        data_props_bins = []
        names = os.listdir(dir_in)
        print(names)

        data_all = defaultdict()
        for z, fname in enumerate(os.listdir(dir_in)):
            path_name=dir_in+'/' + fname
            data_loc=pkl.load(open(path_name, 'rb'))

            for key,val in data_loc.items():
                if z==0:
                    data_all[key]=val.copy()
                else:
                    for k in range(num_cls):
                        if len(val[k])>0:
                            val[k]=np.array(val[k], np.float32)
                            if len(data_all[key])==0:
                                data_all[key][k]=val[k].copy()
                            else:
                                if len(val[k])>0:
                                    data_all[key][k] = np.vstack((data_all[key][k], val[k].copy()))
                                    # val[k][:, 5]*=1.2
                                    # data_all[key][k]=np.vstack((data_all[key][k],val[k][:,:5].copy()))

    else:
        names=['']
        data_all=pkl.load(open(dir_in,'rb'))
    return data_all, names

def max_class_per_position(dets):
    # boxes_all=[]
    boxes_all = np.zeros((0,6))
    for cls, det in enumerate(dets):
        if len(det)>0:
            det_out=np.zeros((len(det),6))
            det_out[:,:5]=det
            det_out[:,5]=cls
            #boxes_all.append(det_out)
            boxes_all=np.vstack((boxes_all,det_out))
    # boxes_all=np.array(boxes_all)
    boxes_all=boxes_all[boxes_all[:,4]>cls_th]
    _, idx = nms(boxes_all[:,:5], nms_all_th)
    boxes_max=boxes_all[idx]
    boxes_max_cls=[[] for cls in range(num_cls)]
    for cls in range(num_cls):
        idx_cls=boxes_max[:,5]==cls
        boxes_max_cls[cls]=boxes_max[idx_cls]
    return boxes_max_cls


def merge_detects_all(data_all, img_dir=None,is_soft=False, is_vote=False, is_small_extend=False, beta=5):
    data_vote={}
    max_dets={}
    start = time.time()
    print('==> data load',start-start0)
    count=0

    for key, bxs in data_all.items():
        if is_border:
            img=cv2.imread(img_dir+key)
            sz=img.shape
        count+=1
        if count%5==0:
            end = time.time()
            print(count, 'time=%0.2f' % (end - start))
        if not key in data_vote:
            data_vote[key] = [np.empty((0, 5), np.float32) for cls in range(num_cls)]


        result = []
        if img_dir is not None:
            im=cv2.imread(os.path.join(img_dir,key))
            im_sz=im.shape
            # bxs=get_merged_box(bxs, im_sz)
        #
        for cls in range(num_cls):
            if len(bxs[cls])==0:
                continue
            # dets_all= bxs[cls].astype(np.float32).copy()
            dets_all=np.array(bxs[cls],np.float32)
            dets_nms, _ = nms(dets_all, nms_th)
            group_coef = 0.75
            if is_vote:
                vote_dets = box_utils.box_voting(dets_nms, dets_all, group_coef, scoring_method='IOU_WAVG', beta=beta)
            else:
                vote_dets=dets_nms

            # print(vote_dets[:,4])
            if is_soft:
                idx = np.argsort(-vote_dets[:, 4])
                dim=min(4000,len(idx))
                vote_dets = vote_dets[idx[:dim]].copy()
                vote_dets, _ = soft_nms(vote_dets, 0.5, min_score=1e-9)


            # idx = np.argsort(-vote_dets[:, 4])
            # vote_dets = vote_dets[idx].copy()
            if is_border:
                vote_dets[vote_dets[:, 0] < 10,4] *= 0.7
                vote_dets[vote_dets[:, 1] < 10, 4] *= 0.7
                vote_dets[vote_dets[:, 3] > sz[0] - 10, 4] *= 0.7
                vote_dets[vote_dets[:, 2] > sz[1] - 10, 4] *= 0.7
            vote_dets = vote_dets[vote_dets[:, 4] > cls_th_v2]
            data_vote[key][cls]=vote_dets.copy()
            zz=0
        max_dets[key] = max_class_per_position(data_vote[key])

    return max_dets




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detects_dir', type=str, help='path to test images')
    parser.add_argument('--test_img_dir', type = str, help = 'path to test images',default='data/test_images_2/')
    parser.add_argument('--pkl_out', type=str, help='path to save results', default='')
    args = parser.parse_args()
    if len(args.pkl_out)==0:
        pkl_out = args.detects_dir+ '.pkl'
    else:
        pkl_out=args.pkl_out
    data_all, names = get_data(args.detects_dir)
    # names=os.listdir(args.detects_dir)
    data_vote=merge_detects_all(data_all, args.test_img_dir,beta=len(names), is_vote=True)
    pkl.dump(data_vote, open(pkl_out, 'wb'))