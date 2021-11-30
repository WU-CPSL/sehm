#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dingwenli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import numpy as np
import time

torch.autograd.set_detect_anomaly(True)

T = 600
D = 56

class Model(nn.Module):
    def __init__(self, rnn_hid_size):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size

        self.build()

    def build(self):
        self.rnn = nn.GRU(D, self.rnn_hid_size, num_layers=1)
        self.out = nn.Linear(self.rnn_hid_size, 1)  

    def forward(self, data, labels):
        values = torch.reshape(data, (data.shape[0], T, D))
        values = values.transpose(0,1)
        
        _, h = self.rnn(values)
        
        y_h = self.out(torch.squeeze(h[0]))
        y_loss = F.binary_cross_entropy_with_logits(y_h, labels)
    
        y_h = torch.sigmoid(y_h)

        return y_loss, y_h, 1.0

    def run_on_batch(self, data, labels, optimizer, epoch = None):
        ret = self(data, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            ret[0].backward()
            optimizer.step()

        return ret
