import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import pandas as pd
import cv2
from skimage import io
from PIL import Image
import os
import pickle


class MAFATDataset(Dataset):

    def __init__(self, csv_file, fold_num, root_dir='../data/train', train=True, transform=None):
        self.csv_frame = pd.read_csv(csv_file, header=0)
        self.fold_num = fold_num
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        if self.train:
            self.csv_frame = self.csv_frame[self.csv_frame['fold'] != self.fold_num]
        else:
            self.csv_frame = self.csv_frame[self.csv_frame['fold'] == self.fold_num]
        self.names = ['general_class', 'sub_class', 'sunroof', 'luggage_carrier',
                      'open_cargo_area', 'enclosed_cab', 'spare_wheel',
                      'wrecked', 'flatbed', 'ladder', 'enclosed_box', 'soft_shell_box',
                      'harnessed_to_a_cart', 'ac_vents', 'color']
        name_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data/name2value.pkl')
        with open(name_path, 'rb') as pklf:
            self.name2value = pickle.load(pklf)

    def __len__(self):
        return len(self.csv_frame)

    @staticmethod
    def __get_obj_sizes(row):
        bbox = row[['p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y']].values.astype(np.int32)
        cnt = [(bbox[idx], bbox[idx + 1]) for idx in range(0, len(bbox), 2)]
        rect = cv2.minAreaRect(np.array(cnt))
        return np.array(rect[1])

    def __getitem__(self, idx):
        row = self.csv_frame.iloc[idx]
        img_name = os.path.join(self.root_dir, str(row['tag_id']) + '.png')
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image}
        labels = [self.name2value[name][row[name]] for name in self.names]
        sample.update({'labels': labels})
        sample.update({'sizes': self.__get_obj_sizes(row)})
        return sample


class MAFATTestDataset(Dataset):

    def __init__(self, csv_file, root_dir, rotates=[0, 90, 180, 270], use_hflip=False, use_vflip=False, transform=None):
        self.csv_frame = pd.read_csv(csv_file, header=0)
        self.root_dir = root_dir
        self.transform = transform
        self.angles = rotates
        self.use_hflip = use_hflip
        self.use_vflip = use_vflip
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path+'/../data/name2value.pkl', 'rb') as pklf:
            self.name2value = pickle.load(pklf)

    def __len__(self):
        return len(self.csv_frame)

    @staticmethod
    def __get_obj_sizes(row):
        bbox = row[['p1_x', 'p_1y', ' p2_x', ' p2_y', ' p3_x', ' p3_y', ' p4_x', ' p4_y']].values.astype(np.int32)
        cnt = [(bbox[idx], bbox[idx + 1]) for idx in range(0, len(bbox), 2)]
        rect = cv2.minAreaRect(np.array(cnt))
        return np.array(rect[1])


    def __getitem__(self, idx):
        row = self.csv_frame.iloc[idx]
        img_name = os.path.join(self.root_dir, str(int(row['tag_id'])) + '.png')
        image = Image.open(img_name)
        sample = {'image': []}
        for angle in self.angles:
            rot_image = transforms.functional.rotate(image, angle)
            # for rot_crop_img in self.transform[0](rot_image):
            #     sample['image'].append(self.transform[1](rot_crop_img).unsqueeze(0))
            sample['image'].append(self.transform[1](self.transform[0](rot_image)).unsqueeze(0))
            if self.use_hflip:
                rot_flip_image = transforms.functional.hflip(rot_image)
                sample['image'].append(self.transform[1](self.transform[2](rot_flip_image)).unsqueeze(0))
            if self.use_vflip:
                rot_flip_image = transforms.functional.vflip(rot_image)
                sample['image'].append(self.transform[1](self.transform[0](rot_flip_image)).unsqueeze(0))
        sample['image'] = torch.cat(sample['image'])
        sample.update({'sizes': torch.FloatTensor([self.__get_obj_sizes(row)] * len(sample['image']))})
        # sample['tag_id'] = int(row['tag_id'])
        return sample


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomAffine(degrees=360, scale=(0.7, 1.3), shear=30), #fillcolor=128
        transforms.RandomCrop(196),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_v2': transforms.Compose([
            transforms.Resize(400),
            transforms.RandomAffine(degrees=360, shear=30),
            transforms.CenterCrop(280),
            transforms.RandomResizedCrop(196,scale=(0.5,1)),
            transforms.ColorJitter(brightness=0.3,contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_v2': transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_mask': transforms.Compose([
            transforms.RandomAffine(degrees=360), #, shear=30),
            transforms.RandomResizedCrop(196,scale=(0.5,1)),
            transforms.ColorJitter(brightness=0.3,contrast=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

    'val_mask': transforms.Compose([
        transforms.CenterCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': [
                transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(196),  #change FiveCrop
                 ]),
                transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                transforms.Compose([
                transforms.Resize(224),
                transforms.RandomCrop(196),
                 ]),
            ],
    'test_v2': [
            transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(196),  #change FiveCrop
             ]),
            transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
            transforms.Resize(400),
            transforms.RandomAffine(degrees=360),
            transforms.CenterCrop(280),
            transforms.RandomResizedCrop(196,scale = (0.7,1)),
             ]),
            ],
    'test_mask': [
            transforms.Compose([
            transforms.CenterCrop(196),  #change FiveCrop
             ]),
            transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
            transforms.RandomAffine(degrees=360),
            transforms.RandomResizedCrop(196,scale = (0.7,1)),
             ]),
            ]

}
