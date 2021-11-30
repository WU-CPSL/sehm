#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluating explanation: use this model for generating explanations for evaluation

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
        # Q: N*L*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q1)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k1)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C
        score1 = F.softmax(QK, dim=2)
        att1 = torch.einsum('nlc,nlcd->nld', score1, x)
        
        # Q: N*L*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q2)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k2)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C
        score2 = F.softmax(QK, dim=2)
        att2 = torch.einsum('nlc,nlcd->nld', score2, x)
        
        # Q: N*L*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q3)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k3)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C
        score3 = F.softmax(QK, dim=2)
        att3 = torch.einsum('nlc,nlcd->nld', score3, x)
        
        # transpose x': N*L*C*D to N*L*D*C
        # Q: N*L*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q4)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k4)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C
        score4 = F.softmax(QK, dim=2)
        att4 = torch.einsum('nlc,nlcd->nld', score4, x)
        
        # Q: N*L*Dk
        Q = torch.einsum('nlcd,cdt->nlt', x, self.W_q5)
        # K: N*L*C*Dk
        K = torch.einsum('nlcd,dt->nlct', x, self.W_k5)
        # V: N*L*C*D
        # perform batched matrix multiplication on K and Q
        # QK: N*L*C
        QK = torch.einsum('nlct,nlt->nlc', K, Q)
        QK = torch.div(QK, self.Dk**0.5)
        # softmax layer
        # score: N*L*C
        score5 = F.softmax(QK, dim=2)
        att5 = torch.einsum('nlc,nlcd->nld', score5, x)
        
        return torch.cat((att1, att2, att3, att4, att5),-1), torch.cat((score1, score2, score3, score4, score5),-1)

class Model(nn.Module):
    def __init__(self, rnn_hid_size):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.build()

    def build(self):
        self.rnn = nn.LSTM(5*D, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.rnn_hid_size, 1)
        
        self.selfatt = SelfAttention(T, D, 30, 30)
        # batchnorm over generated intermediate inputs
        self.bn = nn.BatchNorm1d(5*D)
        # linear approximation
        self.la1 = nn.Linear(SEQ_LEN*5*D, D)
        self.la2 = nn.Linear(D, SEQ_LEN*5*D)
        

    def forward(self, data):
        values = data
        
        # locality-based attention
        values, weights = self.selfatt(values)
        values = self.bn(values.transpose(1,2))
        values = values.transpose(1,2)

        # RNN
        _, (h, c) = self.rnn(values.transpose(0,1))
        
        y_h = self.out(torch.squeeze(h))
        
        y_h = torch.sigmoid(y_h) 
        
        # linear approximating network
        ap = self.la1(torch.reshape(values, (values.shape[0], values.shape[1]*values.shape[2])))
        ap = self.la2(ap)
        ap = torch.reshape(ap, (values.shape[0], values.shape[1], values.shape[2]))
        
        y_p = torch.sigmoid(torch.einsum('nlt,nlt->n', values, ap))
        
        # ap: N*L*(5D)
        ap = torch.unsqueeze(ap, dim=2)
        extended_ap = ap.repeat(1, 1, 30, 1)
        
        # multiply the attention score matrix and the linear approximating matrix to get an end-to-end explanation
        explanations = extended_ap*weights
        explanations = torch.div(explanations[:,:,:,0:D]+explanations[:,:,:,D:2*D]+explanations[:,:,:,2*D:3*D]+explanations[:,:,:,3*D:4*D]+explanations[:,:,:,4*D:], 5.0)

        return y_h, explanations, y_p

    def run_on_batch(self, data):
        ret = self(data)

        return ret
