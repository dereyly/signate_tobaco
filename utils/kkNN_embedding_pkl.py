import torch
import torch.nn as nn
import numpy as np
import os
import sys
import pickle as pkl
import torch.nn.functional as F
import json
from tqdm import tqdm

master_data=pkl.load(open('result/features/feats_tta_9x1.pkl','rb'))
test_data=pkl.load(open('result/features/feats_test_tta_5x1.pkl','rb'))
name_out='result/embeded_tta9x5.pkl'
fold=1
master=[]
id2name=[]

max_id=224

vote_master=4
idx2cat=json.load(open('configs/idx2cat.json','r'))
remap=pkl.load(open('configs/image_folder_idx.pkl','rb'))
cat2idx={}
for key,val in idx2cat.items():
    cat2idx[val]=int(key)
reremap={}
for key,val in remap.items():
    reremap[val]=int(key)

for key, val in master_data[fold].items():
    for feat in val:
        master.append(feat)
        idd = cat2idx[key]
        iddd = remap[str(idd)]
        id2name.append(iddd)

sz_mast=len(master)
master=torch.stack(master)
acc=0
count=0
data_out={}
for key, objects in tqdm(test_data.items()):
    # feats=[]
    data_out[key]={}
    for id_ob,feat in objects.items():
        vote_matrix = np.zeros(max_id)
        feats=[]
        for k in range(len(feat)):
            feats.append(feat[k][1])
        feats=torch.stack(feats)


        res = torch.mm(master, feats.transpose(0, 1))

        idx_min=res.argmax(dim=0)
        idx_sort = torch.argsort(-res,dim=0)

        sz=len(idx_min)

        for i in range(sz):
            for j in range(vote_master):
                id = id2name[idx_sort[j][i]]
                dd=res[idx_sort[j][i],i]
                vote_matrix[id]+=1
        vote_matrix/=sz*vote_master
        cls_out=vote_matrix.argmax()

        prob=vote_matrix[cls_out]
        cls_name=str(cls_out)
        # print(cls_out,prob)
        bb = feat[0][0]
        bb_out = bb[:4].astype(int).tolist()

        data_out[key][id_ob]=[bb_out, vote_matrix, prob, cls_out]
        a=0

pkl.dump(data_out, open(name_out, 'wb'))
