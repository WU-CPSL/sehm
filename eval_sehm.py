#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate the quantitative metrics: local accuracy, AOPC, estimated Lipschitz continuity

@author: dingwenli
"""

import torch
import torch.optim as optim
import numpy as np
import utils
import models
import models.lstm_sehm_quan as lstm_sehm_quan
import warnings
from numpy import linalg as LA
from sklearn.metrics import mean_absolute_error
import data_loader_handoff
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
args = parser.parse_args()

csv_path = 'absolute_path'
test_set_size = 3226
pert_size = 100

def calcLocalAcc(model, val_iter):
    preds = []
    yps = []
    
    for idx, comb in enumerate(val_iter):
        data, label = comb[0], comb[1]
        data = utils.to_var(data)
        label = utils.to_var(label)
        model.eval()
        pred, _, yp = model.run_on_batch(data, None)
        preds += pred.data.cpu().numpy().tolist()
        yps += yp.data.cpu().numpy().tolist()
        
    preds, yps = np.array(preds), np.array(yps)
    
    print('MSE:', LA.norm(preds-yps)**2/preds.shape[0])
        
def calcLC(model, grapset):
    mu, sigma = 0, 0.0001
    lc = []
    for i in range(test_set_size):
        data, mask, label = grapset[i]
        data = utils.to_var(data)
        label = utils.to_var(label)
        model.eval()
        
        pred, exp, _ = model(data)
        exp = exp.data.cpu().numpy()
        # normalization
        exp = np.divide(exp - np.amin(exp), np.amax(exp) - np.amin(exp))
        
        data = data.data.cpu().numpy()
        
        tlc = 0
        for t in range(pert_size):
            s = np.random.normal(mu, sigma, size=data.shape)
            rdata = data + s*mask.data.cpu().numpy()
            rdata = torch.FloatTensor(rdata).cuda()

            rpred, rexp, _ = model(rdata)
            rexp = rexp.data.cpu().numpy()
            rdata = rdata.data.cpu().numpy()
            # normalization
            rexp = np.divide(rexp - np.amin(rexp), np.amax(rexp) - np.amin(rexp))

            if (rpred-0.5)*(pred-0.5)>=0:
                tlc = max(tlc, LA.norm(rexp - exp) / LA.norm(rdata - data))
            
        lc.append(tlc)

    print('est. Lipschitz continuity:', np.mean(lc))
    

def calcAOPC(model, grapset):
    pos_trends = []
    neg_trends = []
    for i in range(test_set_size):
        data, _, label = grapset[i]
        data = utils.to_var(data)
        label = utils.to_var(label)
        model.eval()
        
        new_preds = []
        pred, exp, _ = model.run_on_batch(data, None)
        new_preds.append(pred.data.cpu().numpy()[0])
        exp = exp.data.cpu().numpy()
        if pred<0.5:
            sorted_test_indices = np.argsort(exp.flatten()*data.data.cpu().numpy(), axis=None)
        else:
            sorted_test_indices = np.argsort(-exp.flatten()*data.data.cpu().numpy(), axis=None)

        for j in range(20):
            data = data.data.cpu().numpy()
            test_indices = sorted_test_indices[j*100:(j+1)*100]
            data[0,test_indices] = np.random.random(test_indices.shape)
            data = torch.FloatTensor(data).cuda()
            new_pred, _, _ = model.run_on_batch(data, None)
            new_pred = new_pred.data.cpu().numpy()[0]
            new_preds.append(new_pred)
            
        if pred<0.5:
            neg_trends.append(new_preds)
        else:
            pos_trends.append(new_preds)
    
    neg_trends = np.array(neg_trends)
    pos_trends = np.array(pos_trends)
    aopc = []
    for i in range(1,20):
        pos_aopc = np.sum(pos_trends[:,0] - np.mean(pos_trends[:,i], axis=-1))/pos_trends.shape[0]
        neg_aopc = np.sum(np.mean(neg_trends[:,i], axis=-1) - neg_trends[:,0])/neg_trends.shape[0]
        print('AOPC:', pos_aopc+neg_aopc)
        aopc.append(pos_aopc+neg_aopc)
    np.save(csv_path+'exp_aopc.npy', aopc)

def calcMetrics(model, mode):
    if mode=='AOPC':
        grapset = MySet(train=False)
        calcAOPC(model, grapset)
    elif mode=='LC':
        grapset = MySet(train=False)
        calcLC(model, grapset)
    else:
        test_data_iter = data_loader_handoff.get_loader(batch_size=args.batch_size, train=False)
        calcLocalAcc(model, test_data_iter)

if __name__ == '__main__':
    # load pre-trained model for evaluation
    model = lstm_sehm_quan.Model(108)
    model.load_state_dict(torch.load(csv_path+'sehm_model'), strict=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    calcMetrics(model, args.mode)