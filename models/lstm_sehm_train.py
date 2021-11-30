#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear approximating neural network training

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
lamb = 0.001

torch.autograd.set_detect_anomaly(True)

def linear_reg(gx, wx):
    return torch.norm(gx - wx, p=2)
    
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
    def __init__(self, rnn_hid_size, n_samples=10, pert_dist=10000.0):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.n_samples = n_samples
        self.pert_dist = pert_dist

        self.build()

    def build(self):
        self.rnn = nn.LSTM(5*D, self.rnn_hid_size)
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.rnn_hid_size, 1)
        
        self.selfatt = SelfAttention(T, D, 30, 30)
        self.la1 = nn.Linear(SEQ_LEN*5*D, D)
        self.la2 = nn.Linear(D, SEQ_LEN*5*D)
        # batchnorm over generated intermediate inputs
        self.bn = nn.BatchNorm1d(5*D)

    def forward(self, data, labels):
        values = data

        y_loss = 0.0
        
        # locality-based attention
        values = self.selfatt(values)
        values = self.bn(values.transpose(1,2))
        values = values.transpose(1,2)
        
        # add perturbation to the model
        values = values.repeat(self.n_samples,1,1)
        gx = torch.zeros(values.shape)
        pertur = torch.rand(values.shape)/self.pert_dist
        values = values + pertur.cuda()
        
        _, (h, c) = self.rnn(values.transpose(0,1))
        
        y_h = self.out(torch.squeeze(h))
        
        def processGrad(ii):
            gx[ii,:,:] = torch.autograd.grad(y_h[ii,0], values, retain_graph=True)[0].data[ii,:,:]
        
        Parallel(n_jobs=num_cores, prefer='threads')(delayed(processGrad)(i) for i in range(y_h.shape[0]))
        
        gx = gx.cuda()
        gx = torch.reshape(gx, (gx.shape[0], gx.shape[1]*gx.shape[2]))

        values = torch.reshape(values, (values.shape[0], values.shape[1]*values.shape[2]))
        
        # linear approximating network
        ap = self.la1(values)
        ap = self.la2(ap)
        
        # objective function in Eq. 18
        res = torch.norm(self.la1.weight, p=2)*torch.norm(self.la2.weight, p=2)*torch.norm(values, p=2)
        y_loss = linear_reg(gx, ap) + lamb * res
    
        y_h = torch.sigmoid(y_h)

        return y_loss

    def run_on_batch(self, data, labels, optimizer):
        ret = self(data, labels)

        if optimizer is not None:
            optimizer.zero_grad()
            ret.backward()
            optimizer.step()

        return ret
