#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dingwenli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, rnn_hid_size, seq_len=600, dim=56, **kwargs):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.T = seq_len
        self.D = dim

        self.build()

    def build(self):
        self.cnn = nn.Conv1d(self.D, self.D, 30, 15)
        self.rnn = nn.GRU(self.D, self.rnn_hid_size)
        self.out = nn.Linear(self.rnn_hid_size, 1)  

    def forward(self, data, labels):
        values = torch.reshape(data, (data.shape[0], self.T, self.D))
        values = values.transpose(1,2)
        values = self.cnn(values)
        values = values.transpose(1,2)
        values = values.transpose(0,1)
        _, h = self.rnn(values)
        
        y_h = self.out(torch.squeeze(h))
        y_loss = F.binary_cross_entropy_with_logits(y_h, labels)
    
        y_h = torch.sigmoid(y_h)

        return y_loss, y_h, 1.0

    def run_on_batch(self, data, labels, optimizer):
        ret = self(data, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            ret[0].backward()
            optimizer.step()

        return ret
