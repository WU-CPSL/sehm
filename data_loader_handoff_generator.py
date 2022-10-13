#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data loader for larger dataset, such as pneumonia or AKI, which cannot be fitted into memory as a whole matrix.

@author: dingwenli
"""

#import ujson as json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class MySet(Dataset):
    def __init__(self, train=True):
        super(MySet, self).__init__()
        
        self.csv_path = 'absolute_path'+'pneumonia/'
        if train:
            self.indices = np.where(np.load(self.csv_path+'is_train.npy')==1)[0]
        else:
            self.indices = np.where(np.load(self.csv_path+'is_train.npy')==0)[0]
 
        self.labels = np.load(self.csv_path+'labels.npy')
        

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        forward_values = np.load(self.csv_path+'forward_values_'+str(self.indices[idx])+'.npy')
        return (forward_values, self.labels[self.indices[idx]])

def collate_fn(recs):
    forward_values = []
    labels = []
    
    for x in recs:
        forward_values.append(np.reshape(x[0], (-1, 600*56)))
        labels.append(x[1])

    return torch.FloatTensor(forward_values), torch.FloatTensor(labels)

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
