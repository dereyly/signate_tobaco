import cv2
import os
import numpy as np
import pickle as pkl
from mmdet.ops import nms,soft_nms


is_vis=False
is_img=True
is_square=False
dir_imgs = 'data/test_images_2/'
pkl_path='result/detect_all.pkl'
root_out=dir_out = 'data/imgs_tst_2/'
os.makedirs(root_out,exist_ok=True)
state_names=['test_crop','test_crop_sq']


def crop_img(img, bb, K_add=1.1):
    mx = (bb[0] + bb[2]) / 2
    my = (bb[1] + bb[3]) / 2
    w = bb[2] - bb[0]
    h = bb[3] - bb[1]
    S = int(K_add * max(w, h))
    if is_square:
        bb_add = np.array([mx - K_add * w / 2, my - K_add * h / 2, mx + K_add * w / 2, my + K_add * h / 2], int)
    else:
        bb_add = np.array([mx - S / 2, my - S / 2, mx + S / 2, my + S / 2], int)
    bb_add[bb_add < 0] = 0


    crop = img[bb_add[1]:bb_add[3], bb_add[0]:bb_add[2]]
    if is_square:
        crop = cv2.resize(crop, (250, 250))
    else:
        cr_sz = crop.shape
        if cr_sz[0] != S or cr_sz[1] != S:
            cc = 128 * np.ones((S, S, 3), np.uint8)
            x1 = max(0, int((S - cr_sz[0] - 1) / 2))
            y1 = max(0, int((S - cr_sz[1] - 1) / 2))
            x2 = min(S, x1 + cr_sz[0])
            y2 = min(S, y1 + cr_sz[1])

            cc[x1:x2, y1:y2] = crop.copy()
            crop = cc
    return crop


for st_name in state_names:
    K_add=1.1
    if st_name=='test_crop_sq':
        is_square=True
        K_add=1.0
    dir_out = root_out+ '/%s/' % st_name

    os.makedirs(dir_out,exist_ok=True)
    th_nms=0.6
    rsz=(250,250)

    data=pkl.load(open(pkl_path,'rb'))
    img_idx={}
    img_list=[]
    for name, bbox in data.items():
        bbox_all=[]
        img = cv2.imread(dir_imgs+name)
        for bbs in bbox:
            for bb in bbs:
                bbox_all.append(bb)
        bbox_all = np.array(bbox_all,np.float32)
        bbox_nms,_=nms(bbox_all, th_nms)
        for k,bb in enumerate(bbox_nms):
            crop=crop_img(img,bb,K_add=K_add)
            if rsz[0]>0:
                crop=cv2.resize(crop,rsz)
            img_path=dir_out+name[:-4]+'_%d.jpg' % k
            if is_img:
                cv2.imwrite(img_path,crop)
            img_idx[name] = img_path
            img_list.append([name[:-4]+'_%d.jpg' % k, [name,k,bb]])
        if is_vis:
            probs=bbox_nms[:,4].copy()
            bbox_nms = bbox_nms.astype(int)
            for k,bb in enumerate(bbox_nms):
                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0,0,255), 3)
                cv2.putText(img, '%.2f'%probs[k], (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0))
            cv2.imwrite('data/test_vis/'+name,img)
    pkl.dump(img_idx,open(dir_out[:-1]+'_index.pkl','wb'))
    pkl.dump(img_list,open(dir_out[:-1]+'_list.pkl','wb'))

