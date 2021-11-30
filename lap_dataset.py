#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data loader for one-by-one loading

@author: dingwenli
"""
import numpy as np

import torch
from torch.utils.data import Dataset

class MySet(Dataset):
    def __init__(self, train=True):
        super(MySet, self).__init__()
        csv_path = 'absolute_path'
        flag = 'train' if train else 'test'
        
        self.values = np.load(csv_path+'{}_forward_values.npy'.format(flag))
        self.masks = np.load(csv_path+'{}_forward_masks.npy'.format(flag))
        self.labels = np.load(csv_path+'{}_labels.npy'.format(flag))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return (torch.FloatTensor(np.reshape(self.values[idx], (1, 600*56))), torch.FloatTensor(np.reshape(self.masks[idx], (1, 600*56))), torch.FloatTensor(self.labels[idx]))
