#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi scale CNN with multiple conv layers of different kernel sizes in parallel

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
        self.cnn3 = nn.Conv1d(self.D, 13, 30, 15)
        self.cnn4 = nn.Conv1d(self.D, 13, 40, 20, 100)
        self.cnn5 = nn.Conv1d(self.D, 13, 50, 25, 200)
        self.cnn6 = nn.Conv1d(self.D, 13, 60, 30, 300)
        self.rnn = nn.LSTM(4*13, self.rnn_hid_size)
        self.out = nn.Linear(self.rnn_hid_size, 1)  

    def forward(self, data, labels):
        values = torch.reshape(data, (data.shape[0], self.T, self.D))
        values = values.transpose(1,2)
        v3 = self.cnn3(values)
        v3 = v3.transpose(1,2)
        v3 = v3.transpose(0,1)
        v4 = self.cnn4(values)
        v4 = v4.transpose(1,2)
        v4 = v4.transpose(0,1)
        v5 = self.cnn5(values)
        v5 = v5.transpose(1,2)
        v5 = v5.transpose(0,1)
        v6 = self.cnn6(values)
        v6 = v6.transpose(1,2)
        v6 = v6.transpose(0,1)
        values = torch.cat((v3, v4, v5, v6), -1)
        _, (h, c) = self.rnn(values)
        
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
