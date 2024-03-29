#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dingwenli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

import math

    
class SelfAttention(nn.Module):
    def __init__(self, Tot, D, Dk, C, nH):
        self.Tot = Tot
        self.D = D
        self.Dk = Dk
        self.C = C
        self.nH = nH
        super(SelfAttention, self).__init__()
        self.build()
        
    def build(self):
        for i in range(self.nH):
            setattr(self, 'W_q%d'%(i+1), Parameter(torch.Tensor(self.C, self.D, self.Dk)))
            setattr(self, 'W_k%d'%(i+1), Parameter(torch.Tensor(self.D, self.Dk)))

        self.reset_parameters()
        
    def reset_parameters(self):
        for i in range(self.nH):
            W_qi = getattr(self, 'W_q%d'%(i+1))
            W_ki = getattr(self, 'W_k%d'%(i+1))
            stdv = 1. / math.sqrt(W_qi.size(0))
            W_qi.data.uniform_(-stdv, stdv)
            stdv = 1. / math.sqrt(W_ki.size(0))
            W_ki.data.uniform_(-stdv, stdv)

    def forward(self, x):   
        # reshape input x: N*T*D to x': N*L*C*D
        x = torch.reshape(x, (x.shape[0], self.Tot//self.C, self.C, self.D))
        attns = []
        for i in range(self.nH):
            W_qi = getattr(self, 'W_q%d'%(i+1))
            W_ki = getattr(self, 'W_k%d'%(i+1))
            # transpose x': N*L*C*D to N*L*D*C
            # Q: N*L*D*Dk
            Q = torch.einsum('nlcd,cdt->nlt', x, W_qi)
            # K: N*L*C*Dk
            K = torch.einsum('nlcd,dt->nlct', x, W_ki)
            # V: N*L*C*D
            # perform batched matrix multiplication on K and Q
            # QK: N*L*C*D
            QK = torch.einsum('nlct,nlt->nlc', K, Q)
            QK = torch.div(QK, self.Dk**0.5)
            # softmax layer
            # score: N*L*C*D
            score = F.softmax(QK, dim=2)
            attn = torch.einsum('nlc,nlcd->nld', score, x)
            attns.append(attn)
        
        return torch.cat(attns, -1)

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
        
        self.selfatt = SelfAttention(self.T, self.D, 30, self.nei, self.nH)
        # batchnorm over generated intermediate inputs
        self.bn = nn.BatchNorm1d(self.nH * self.D)

    def forward(self, data, labels):
        values = data
        
        # locality-based attention
        values = self.selfatt(values)
        values = self.bn(values.transpose(1,2))
        values = values.transpose(1,2)
        values = values.transpose(0,1)
        
        # RNN
        _, h = self.rnn(values)
        
        y_h = self.out(torch.squeeze(h))
        
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
