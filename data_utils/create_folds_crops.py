import os
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import cv2
# from my_utils import cut_image
# import aug_util as aug
import pickle as pkl
import argparse
from tqdm import tqdm

# data_root = '/media/dereyly/data_ssd/ImageDB/japan_ships/'
# dir_out='/media/dereyly/data_ssd/ImageDB/japan_ships/train_val/'


is_vis = True





def generate_crops(data_root, dir_out, fold, n_fold, is_full=False, K_add=1.1, is_square=False):
    # Get Annotations
    _train_images_path = os.path.join(data_root, 'train_images')
    _train_annotations_path = os.path.join(data_root, 'train_annotations')
    train_annotations_files = os.listdir(_train_annotations_path)
    train_images_files = os.listdir(_train_images_path)

    per_category = {}
    per_image = []
    for train_annotations_file in train_annotations_files:
        with open(os.path.join(_train_annotations_path, train_annotations_file)) as f:
            annotation = json.load(f)
        labels = annotation['labels']
        per_image.append(len(labels))
        for label in labels:
            if label['category'] in per_category:
                per_category[label['category']] += 1
            else:
                per_category[label['category']] = 1

    category_names = ()
    vals = ()
    for category in per_category:
        if category == 'unknown':
            continue
        category_names += (category,)
        vals += (per_category[category],)

    # Create COCO style classes
    count_im = 0
    count_ob = 0
    data = {'annotations': [], 'categories': [], 'images': []}
    data['categories'] = []
    cat2idx = {}
    idx2cat = json.load(open('configs/idx2cat.json', 'r'))
    for cls, cls_name in idx2cat.items():
        cls = int(cls)
        data['categories'].append({'supercategory': 'none', 'id': cls, 'name': cls_name})
        cat2idx[cls_name] = cls

    # MAin loop to crop images and create annotations
    if is_full:
        fold['train'] += fold['val']
        zz = 0
    for state, fold_idx in fold.items():
        dir_fold = dir_out + '/%s_%d/' % (state, n_fold)
        os.makedirs(dir_fold, exist_ok=True)
        path_ann_out = dir_out + '/%s_%d.json' % (state, n_fold)
        data['images'] = []
        data['annotations'] = []
        for id in fold_idx:
            name = train_images_files[id]
            path = _train_images_path + '/' + name
            annotation_path = _train_annotations_path + '/' + name.split('.')[0] + '.json'

            img = cv2.imread(path)
            sz = img.shape

            # print(im.shape)
            with open(annotation_path) as f:
                annotation = json.load(f)
            bbox = []
            cls_all = []
            for l in annotation['labels']:
                if l['category'] in category_names:
                    bb = l['box2d']
                    # bbox=[bb['y1'], bb['x1'], bb['y2'], bb['x2']]
                    bbox.append([bb['x1'], bb['y1'], bb['x2'], bb['y2']])
                    label = cat2idx[l['category']]
                    cls_all.append(label)
            bbox = np.array(bbox)

            cls_all = np.array(cls_all)

            im_name = name.split('.')[0] + '_%d.jpg' % (count_im)

            data_loc = []
            for k, bb in enumerate(bbox):
                mx = (bb[0] + bb[2]) / 2
                my = (bb[1] + bb[3]) / 2
                w = bb[2] - bb[0]
                h = bb[3] - bb[1]
                S=int(K_add*max(w,h))
                if is_square:
                    bb_add = np.array([mx - K_add*w / 2, my - K_add*h / 2, mx + K_add*w / 2, my + K_add*h / 2], int)
                else:
                    bb_add = np.array([mx - S / 2, my - S / 2, mx + S / 2, my + S / 2],int)
                bb_add[bb_add<0]=0
                #bb_add[2] = bb_add[2] if bb_add[2] <= sz[1] - 1 else sz[1] - 1
                #bb_add[3] = bb_add[3] if bb_add[3] <= sz[0] - 1 else sz[0] - 1
                #bb_add=bb_add.astype(int)
                crop=img[bb_add[1]:bb_add[3],bb_add[0]:bb_add[2]]
                cr_sz=crop.shape
                if is_square:
                    crop=cv2.resize(crop,(250,250))
                else:
                    if cr_sz[0]!=S or cr_sz[1]!=S:
                        cc=128*np.ones((S,S,3),np.uint8)
                        x1=max(0,int((S-cr_sz[0]-1)/2))
                        y1=max(0,int((S-cr_sz[1]-1)/2))
                        x2 = min(S, x1 + cr_sz[0])
                        y2 = min(S, y1 + cr_sz[1])
                        #print(S,cr_sz,x1,y1,x2-x1,y2-y1)
                        cc[x1:x2,y1:y2]=crop.copy()
                        crop=cc
                cls_fold=dir_fold+'/%d/' % cls_all[k]
                os.makedirs(cls_fold,exist_ok=True)
                cv2.imwrite(cls_fold + im_name,crop)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='dir to root dir with train images', default='../data')
    parser.add_argument('--dir_out', type=str, help='directory to save results', default='../data/train_val_crop')
    args = parser.parse_args()
    folds = pkl.load(open('configs/folds.pkl', 'rb'))
    adds = [1.06, 1.06, 1.15, 1.15]
    n_folds = [1,2,3,4]
    sq=[True,True,False,False]
    for k in tqdm(range(len(n_folds))):
        fold = folds[n_folds[k]]
        generate_crops(args.data_root, args.dir_out, fold, n_folds[k], K_add=adds[k], is_square=sq[k])



