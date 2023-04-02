import numpy as np
import glob
import os
from os.path import join
import random
import h5py
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

class cam16ctp_ca(Dataset):
    '''
    dataset for casii (ctp features). 20x. output entire bag, keyset, label
    '''
    def __init__(self, train='train', transform=None, keys='', split=42, splitrate=0.1):

        self.img_dir = '../casii/data/feats/cam16CTP'

        self.split = split
        self.keys = keys
        self.splitrate = splitrate
        self.train = train
        
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))

        postrainlist, posvallist = train_test_split(postrainlist, test_size=self.splitrate, random_state=self.split)
        negtrainlist, negvallist = train_test_split(negtrainlist, test_size=self.splitrate, random_state=self.split)
        testnamelist = glob.glob(os.path.join(self.img_dir, 'test', '*', '*.npy'))

        if train == 'train':
            self.img_names = postrainlist + negtrainlist
        elif train == 'test':
            self.img_names = testnamelist
        elif train == 'val':
            self.img_names = posvallist + negvallist
            
        self.transform = transform
        self.keyset = np.load(join('./data/keys', self.keys))

    def __len__(self):
        return len(self.img_names)
    
    def get_weights(self):
        # get weights for weight random sampler (training only)
        if self.train != 'train':
            raise TypeError('WEIGHT SAMPLING FOR TRAINING SET ONLY')
        N = len(self.img_names)
        postrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'tumor', '*.npy'))
        negtrainlist = glob.glob(os.path.join(self.img_dir, 'train', 'normal', '*.npy'))
        w_per_cls = {'tumor': N/len(postrainlist), 'normal':N/len(negtrainlist)}
        weights = [w_per_cls[name.split('/')[-2]] for name in self.img_names]

        return torch.DoubleTensor(weights)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = np.load(img_path)

        label = img_path.split('/')[-2]
        if label == 'tumor':
            label = 1
        elif label == 'normal':
            label = 0
            
        return torch.Tensor(image), torch.Tensor(self.keyset), label

