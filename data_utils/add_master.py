import os
import pickle as pkl
import json
import shutil
import cv2
from tqdm import tqdm
dir_img='data/master_images/'
names_copy=['train_3','train_4']
dir_out='data/imgs_trainval'
for nc in names_copy:
    os.makedirs(dir_out+nc,exist_ok=True)

rsz=0.15
idx2cat=json.load(open('configs/idx2cat.json','r'))
remap=pkl.load(open('configs/image_folder_idx.pkl','rb'))
cat2idx={}
for key,val in idx2cat.items():
    cat2idx[val]=int(key)
reremap={}
for key,val in remap.items():
    reremap[val]=int(key)


for name in tqdm(os.listdir(dir_img)):
    id = int(name[:-4])
    idd = cat2idx[id]
    iddd = reremap[idd]
    img=cv2.imread(dir_img+name)
    sz=img.shape
    sz_new=(int(sz[1]*rsz),int(sz[0]*rsz))
    img=cv2.resize(img,sz_new)
    for nc in names_copy:
        dir_name= dir_out + '/%s/%d/' % (nc, idd)
        fname_out = dir_name+ name
        cv2.imwrite(fname_out,img)
