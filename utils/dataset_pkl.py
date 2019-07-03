import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch

from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np
# from torchvision import transforms, utils

# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')




def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)

class DatasetPkl(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, pkl_name, transform=None): #, loader=default_loader):
        with open(pkl_name,'rb') as f_pkl:
            self.imgs=pkl.load(f_pkl)
        #self.imgs=self.imgs[0]+self.imgs[1]
        #self.imgs=list(self.imgs.items())
        self.root = root
        #self.classes = range(num_classes)
        self.transform = transform




    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        fname, meta = self.imgs[index]
        # print(',,,,,,,',params)
        path=self.root+'/' +fname
        img = Image.open(path) #self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, meta

    def __len__(self):
        return len(self.imgs)
