import os
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import cv2
#from my_utils import cut_image
#import aug_util as aug
import pickle as pkl
import argparse



is_vis=False
is_img = False

def generate_crops(data_root, dir_out, fold,n_fold,is_full=False, is_one_cls=False):
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

    #Create COCO style classes
    count_im = 0
    count_ob = 0
    data = {'annotations': [], 'categories': [], 'images': []}
    data['categories'] = []
    cat2idx = {}

    if is_one_cls:
        data['categories'].append({'supercategory': 'none', 'id': 1, 'name': 'bb_tobac'})
    else:
        idx2cat = json.load(open(data_root + '/idx2cat.json', 'r'))
        for cls, cls_name in idx2cat.items():
            cls=int(cls)
            data['categories'].append({'supercategory': 'none', 'id': cls, 'name': cls_name})
            cat2idx[cls_name] = cls

    #MAin loop to crop images and create annotations
    if is_full:
        fold['train']+=fold['val']
        zz=0
    for state,fold_idx in fold.items():
        dir_fold = dir_out + '/%s_%d/' % (state,n_fold)
        os.makedirs(dir_fold,exist_ok=True)
        path_ann_out = dir_out+'/%s_%d.json' % (state,n_fold)
        data['images'] = []
        data['annotations'] = []
        for id in fold_idx:
            name=train_images_files[id]
            path=_train_images_path+'/'+name
            annotation_path=_train_annotations_path+'/'+name.split('.')[0]+'.json'

            img = cv2.imread(path)
            sz = img.shape


            # print(im.shape)
            with open(annotation_path) as f:
                annotation = json.load(f)
            bbox=[]
            cls_all=[]
            for l in annotation['labels']:
                if l['category'] in category_names:
                    bb = l['box2d']
                    #bbox=[bb['y1'], bb['x1'], bb['y2'], bb['x2']]
                    bbox.append([bb['x1'], bb['y1'], bb['x2'], bb['y2']])
                    if not is_one_cls:
                        label=cat2idx[l['category']]
                    else:
                        label=1
                    cls_all.append(label)
            bbox=np.array(bbox)

            cls_all=np.array(cls_all)

            #im_name = name.split('.')[0] + '_%d.jpg' % (count_im)
            im_name=name
            data_loc=[]
            for k, bb in enumerate(bbox):
                if is_vis:
                    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                    bb = bb.astype(int)
                    #cls = c_cls[k] - 1
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
                    cv2.putText(img, '%d' % cls_all[k], (bb[0], bb[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))


                count_ob += 1
                bb[[2, 3]] -= bb[[0, 1]]
                cls_out = int(cls_all[k].copy())
                if is_one_cls:
                    cls_out=1
                data_loc.append(
                    {'id': count_ob, 'bbox': bb.tolist(), 'segmentation': [], 'area': int(bb[2] * bb[3]),
                     'image_id': count_im, 'category_id': cls_out, 'ignore': 0, 'iscrowd': 0})

            if len(data_loc)>0:
                # for anns in
                data['annotations']+=data_loc
                data['images'].append(
                    {'height': sz[0], 'width': sz[1], 'file_name': im_name, 'id': count_im})
                if is_img:
                    cv2.imwrite(dir_fold + im_name, img)
                count_im += 1

        json.dump(data,open(path_ann_out,'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='dir to root dir with train images', default='../data')
    parser.add_argument('--dir_out', type=str, help='directory to save results', default='../data/train_val_single')
    args = parser.parse_args()
    folds = pkl.load(open('configs/folds.pkl', 'rb'))
    adds=[0,0,0,0,0]
    n_fold=3
    #for n_fold in range(5):
    fold = folds[n_fold]
    generate_crops(args.data_root, args.dir_out, fold, n_fold, is_full=True, is_one_cls=True)
    # break



