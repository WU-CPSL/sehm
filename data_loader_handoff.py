#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data loader for training and testing

@author: dingwenli
"""

#import ujson as json
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, train=True):
        super(MySet, self).__init__()
        csv_path = 'absolute_path'+'del/'
        flag = 'train' if train else 'test'
        
        self.values = np.load(csv_path+'{}_forward_values.npy'.format(flag))
        self.labels = np.load(csv_path+'{}_labels.npy'.format(flag))
        self.masks = np.load(csv_path+'{}_forward_masks.npy'.format(flag))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (self.values[idx], self.labels[idx], self.masks[idx])

def collate_fn(recs):
    values = []
    labels = []
    masks = []
    
    for x in recs:
        values.append(np.reshape(x[0], (-1, 600*56)))
        labels.append(x[1])
        masks.append(np.reshape(x[2], (-1, 600*56)))
        
    return torch.FloatTensor(values), torch.FloatTensor(labels), torch.FloatTensor(masks)

def get_loader(batch_size = 64, train=True, shuffle = True):
    data_set = MySet(train)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              num_workers = 16, \
                              shuffle = shuffle, \
                              pin_memory = False, \
                              collate_fn = collate_fn
    )

    return data_iter
