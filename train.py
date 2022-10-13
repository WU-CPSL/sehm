#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main script for performing the predictive performance evaluation and computational efficiency evaluation

This script includes the evaluation for SEHM(LSTM/GRU), LSTM/GRU, Conv+LSTM/GRU, multi scale CNN, RAIM

Here are the links to the implementation of the models used in our evaluation:

BRITS: https://github.com/caow13/BRITS
GRU-D: https://github.com/PeterChe1990/GRU-D (Keras), https://github.com/caow13/BRITS (PyTorch)
Latent-ODE: https://github.com/YuliaRubanova/latent_ode
TCN: https://github.com/Baichenjia/Tensorflow-TCN
SAnD: https://github.com/khirotaka/SAnD
Informer: https://github.com/zhouhaoyi/Informer2020
Performer: https://github.com/lucidrains/performer-pytorch
RAIM: https://github.com/yanboxu

@author: dingwenli
"""

import torch
import torch.optim as optim
import models
import models.gru
import models.conv_gru
import models.multi_scale_cnn
import models.gru_sehm
import models.gru_sehm_train
import models.gru_sehm_kernelized
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int)

args = parser.parse_args()

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    data = torch.rand(10, 4, 600, 56)
    labels = torch.empty(10, 4, 1).random_(2)
    

    for epoch in range(args.epochs):
        model.train()
        loss = []
        for i in range(data.size()[0]):
            input, label = data[i], labels[i]
            ret = model.run_on_batch(input, label, optimizer)
            loss.append(ret[0].data.cpu().numpy())
        print('epoch', epoch, 'loss', sum(loss)/data.size()[0])
            


def run(nei):
    model = getattr(models, args.model).Model(args.hid_size, nei=nei)
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    return train(model)


if __name__ == '__main__':
    nei = 30
    run(nei)
