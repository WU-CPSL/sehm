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
import numpy as np
import utils
import models
import models.gru
import models.conv_gru
import models.multi_scale_cnn
import models.gru_sehm
import models.gru_sehm_train
import models.gru_sehm_kernelized
import argparse
# load pneu or aki
import data_loader_handoff_generator
# load del
import data_loader_handoff
from sklearn import metrics
import time

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str)
parser.add_argument('--hid_size', type=int)
parser.add_argument('--flag', type=str)
args = parser.parse_args()

def train(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_data_iter = data_loader_handoff.get_loader(batch_size=args.batch_size, train=True)
    test_data_iter = data_loader_handoff.get_loader(batch_size=args.batch_size, train=False)
    
    times = []

    for epoch in range(args.epochs):
        model.train()
        elapsed = 0
        for idx, comb in enumerate(train_data_iter):
            data, label = comb[0], comb[1]
            data = utils.to_var(data)
            label = utils.to_var(label)
            # measure the start time
            start_epoch = time.time()
            model.run_on_batch(data, label, optimizer, epoch)
            torch.cuda.synchronize()
            # measure the end time
            end_epoch = time.time()
            elapsed += end_epoch - start_epoch
        
        times.append(elapsed)
        curr_AUROC, curr_AUPRC, curr_labels, curr_preds = evaluate(model, test_data_iter)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print('avg training time:', avg_time, 'std training time:', std_time)
    return avg_time, std_time

def evaluate(model, val_iter):
    model.eval()

    labels = []
    preds = []
    times = []
    
    for i in range(10):
        elapsed = 0
        for idx, comb in enumerate(val_iter):
            data, label = comb[0], comb[1]
            data = utils.to_var(data)
            label = utils.to_var(label)

            # measure the start time
            start_epoch = time.time()
            _, pred, _ = model.run_on_batch(data, label, None)
            torch.cuda.synchronize()
            # measure the end time
            end_epoch = time.time()
            elapsed += end_epoch - start_epoch

            pred = pred.data.cpu().numpy()
            label = label.data.cpu().numpy()

            # collect test label & prediction
            labels += label.tolist()
            preds += pred.tolist()
            
        times.append(elapsed)
        
    avg_time = np.mean(times)
    std_time = np.std(times)
    print('avg testing time (s):', avg_time, 'std testing time (s):', std_time)
    
    labels = np.asarray(labels).astype('int32')
    preds = np.asarray(preds)
    
    auroc = metrics.roc_auc_score(labels, preds)
    
    prec, rec, _ = metrics.precision_recall_curve(labels, preds)
    auprc = metrics.auc(rec, prec)
    
    print('Test AUROC {}'.format(auroc),'Test AUPRC {}'.format(auprc))

    return auroc, auprc, labels, preds


def run(nei):
    model = getattr(models, args.model).Model(args.hid_size, nei=nei)
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    return train(model)


if __name__ == '__main__':
    for i in range(10):
        nei = 30
        print('{}/10 run'.format(i+1))
        run(nei)
