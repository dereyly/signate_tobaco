import os
import pickle as pkl
import json
import shutil
import cv2
from tqdm import tqdm
dir_img='data/master_images/'
dir_out='data/imgs_tst_2/master_resize/'

os.makedirs(dir_out,exist_ok=True)
rsz=0.15


for name in tqdm(os.listdir(dir_img)):
    img=cv2.imread(dir_img+name)
    sz=img.shape
    sz_new=(int(sz[1]*rsz),int(sz[0]*rsz))
    img=cv2.resize(img,sz_new)
    fname_out = dir_out + name
    cv2.imwrite(fname_out, img)

