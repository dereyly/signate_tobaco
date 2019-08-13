#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python infer_full_tta.py
python data_utils/vote_detect_tta.py --detects_dir=result/detect_all --test_img_dir=data/test_images_2/
python data_utils/det2crops.py
python data_utils/resize_master.py
python classify_crops.py
python master_embed.py
python test_embed.py
python utils/kkNN_embedding_pkl.py
python submit_signate_cls_embed.py --pkl_name=result/res_all_tta_2_25x4.pkl --pkl_embed=result/embeded_tta9x5.pkl
