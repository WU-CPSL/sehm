#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dingwenli

Kernelized local attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernelized_attention import Attention
    
class SelfAttention(nn.Module):
    def __init__(self, Tot, D, C, nH):
        self.Tot = Tot
        self.D = D
        self.C = C
        self.nH = nH
        super(SelfAttention, self).__init__()
        self.attn = Attention(dim=D, heads=nH, dim_head=D)
        
    def forward(self, x):   
        # reshape input x: N*T*D to x': (NL)*C*D where T=LC
        x = torch.reshape(x, (x.shape[0]*self.Tot//self.C, self.C, self.D))
        return self.attn(x)

class Model(nn.Module):
    def __init__(self, rnn_hid_size, seq_len=600, dim=56, nei=30, nH=4):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.nei = nei
        self.T = seq_len
        self.D = dim
        self.L = self.T // self.nei
        self.nH = nH

        self.build()

    def build(self):
        self.rnn = nn.GRU(self.nH*self.D, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.rnn_hid_size, 1)
        
        self.selfatt = SelfAttention(self.T, self.D, self.nei, self.nH)
        self.reduced = nn.Linear(self.nei, 1)
        # batchnorm over generated intermediate inputs
        self.bn = nn.BatchNorm1d(self.nH*self.D)

    def forward(self, data, labels):
        values = data
        
        # kernelized local attention
        values = self.selfatt(values)
        # (NL)*C*HD -> (NL)*HD*C
        values = self.reduced(torch.transpose(values, 1, 2))
        values = torch.squeeze(values)
        # reshape (NL)*HD to N*L*HD
        values = torch.reshape(values, (data.shape[0], values.shape[0]//data.shape[0], values.shape[-1]))
        values = self.bn(values.transpose(1,2))
        values = values.transpose(1,2)
        values = values.transpose(0,1)
        
        # RNN
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
