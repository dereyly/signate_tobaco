#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python data_utils/create_folds_full.py --data_root=data --dir_out=data/data_detect_single
python mmdetection/tools/train.py configs/cascade_rcnn_r50_fpn_full.py
python data_utils/create_folds_crops.py --data_root=data/ --dir_out=data/imgs_trainval
python data_utils/add_master.py
python tobaco_train.py data/imgs_trainval -b 32 --lr=0.00001 --arch=InceptionResNetV2_multi --dataset=_3 --fold=2
python tobaco_train.py data/imgs_trainval -b 32 --lr=0.00001 --arch=InceptionResNetV2_multi --dataset=_3 --fold=3
python tobaco_train.py data/imgs_trainval -b 64 --lr=0.00002 --arch=se_resnext50_32x4d_multi --dataset=_3 --fold=4
python tobaco_train.py data/imgs_trainval -b 64 --lr=0.00002 --arch=se_resnext50_32x4d_multi --dataset=_3 --fold=1