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

from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()


SEQ_LEN = 20
S1 = 20
S2 = 120
D = 56
T = 600
ep = 0

torch.autograd.set_detect_anomaly(True)
    
class SelfAttention(nn.Module):
    def __init__(self, Tot, D, Dk, C):
        self.Tot = Tot
        self.D = D
        self.Dk = Dk
        self.C = C
        super(SelfAttention, self).__init__()
        self.build(D, Dk, C)
        
    def build(self, D, Dk, C):
        self.W_q1 = Parameter(torch.Tensor(C, D, Dk))
        self.W_k1 = Parameter(torch.Tensor(D, Dk))
        self.W_q2 = Parameter(torch.Tensor(C, D, Dk))
        self.W_k2 = Parameter(torch.Tensor(D, Dk))
        self.W_q3 = Parameter(torch.Tensor(C, D, Dk))
        self.W_k3 = Parameter(torch.Tensor(D, Dk))
        self.W_q4 = Parameter(torch.Tensor(C, D, Dk))
        self.W_k4 = Parameter(torch.Tensor(D, Dk))
        self.W_q5 = Parameter(torch.Tensor(C, D, Dk))
        self.W_k5 = Parameter(torch.Tensor(D, Dk))


        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_q1.size(0))
        self.W_q1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.W_k1.size(0))
        self.W_k1.data.uniform_(-stdv, stdv)
        
        stdv = 1. / math.sqrt(self.W_q2.size(0))
        self.W_q2.data.uniform_(-stdv, stdv)  
        stdv = 1. / math.sqrt(self.W_k2.size(0))
        self.W_k2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.W_q3.size(0))
        self.W_q3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.W_k3.size(0))
        self.W_k3.data.uniform_(-stdv, stdv)
        
        stdv = 1. / math.sqrt(self.W_q4.size(0))
        self.W_q4.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.W_k4.size(0))
        self.W_k4.data.uniform_(-stdv, stdv)
        
        stdv = 1. / math.sqrt(self.W_q5.size(0))
        self.W_q5.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.W_k5.size(0))
        self.W_k5.data.uniform_(-stdv, stdv)
        
    def forward(self, x):   
        # reshape input x: N*T*D to x': N*L*C*D
        x = torch.reshape(x, (x.shape[0], self.Tot//self.C, self.C, self.D))
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*D*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q1)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k1)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C*D
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C*D
        score1 = F.softmax(QK, dim=2)
        att1 = torch.einsum('nlc,nlcd->nld', score1, x)
        
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*D*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q2)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k2)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C*D
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C*D
        score2 = F.softmax(QK, dim=2)
        att2 = torch.einsum('nlc,nlcd->nld', score2, x)
        
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*D*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q3)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k3)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C*D
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C*D
        score3 = F.softmax(QK, dim=2)
        att3 = torch.einsum('nlc,nlcd->nld', score3, x)
        
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*D*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q4)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k4)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C*D
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C*D
        score4 = F.softmax(QK, dim=2)
        att4 = torch.einsum('nlc,nlcd->nld', score4, x)
        
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*D*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q5)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k5)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C*D
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C*D
        score5 = F.softmax(QK, dim=2)
        att5 = torch.einsum('nlc,nlcd->nld', score5, x)
        
        return torch.cat((att1, att2, att3, att4, att5),-1)

class Model(nn.Module):
    def __init__(self, rnn_hid_size, nei):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.nei = nei

        self.build()

    def build(self):
        self.rnn = nn.LSTM(5*D, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.rnn_hid_size, 1)
        
        self.selfatt = SelfAttention(T, D, 30, self.nei)
        # batchnorm over generated intermediate inputs
        self.bn = nn.BatchNorm1d(5*D)

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
