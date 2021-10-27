# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:49:23 2021

@author: Kaima
"""
import numpy as np
import copy
import torch
import torch.optim as optim
from datasets import metr_La_processing, TraDataLoader, TeDataLoader
from model import general_satcn
from sample_gene import sadj_transform, sample_processing, ZerooneScaler
from loss import masked_rmse, masked_mae


'''
Hyper parameters
'''
device = 'cuda'
training_rate = 0.5 # training time ratio
sample_rate = 0.5 # number of missing node
missing_rate = 0# MIssing data in the observed sensors
seq_length = 6 # INput length of the sequence

layers = 1 # number of unmasked spatial aggregation and temporal convolution
t_kernel = 2 # the length of temporal convolution kernel
least_k = 2 # the number of neighbors
N_v = 20 # the number of masked nodes for training

batch_size = 8 # training batch size
episode = 20 # max training epoch
channels = 128

#scalers = ['identity']


'''
Generate Dataset
'''
print("generate dataset...",flush=True)
metr_La_processing(training_rate, sample_rate, missing_rate, seq_length, (layers + 1) *  (t_kernel - 1), least_k)

Tradata = np.load('metr/training_data.npz')
Tedata = np.load('metr/test_data.npz')
realy = torch.Tensor(Tedata['y'][:, :, :, Tedata['u']]).to('cuda')

Atra2 = sadj_transform(Tradata['A2'], least_k)
Ate2 = sadj_transform(Tedata['A2'], least_k)
avg_d = {}
avg_d['log'] = torch.tensor(np.mean(np.log(np.sum(Atra2, axis = 0) +1)))

Traloader = TraDataLoader(Tradata['x'], Tradata['A1'], batch_size)
Teloader = TeDataLoader(Tedata['x'], Tedata['y'], Tedata['A1'])

Scaler = ZerooneScaler(Tradata['x'].max())

'''
Define model
'''
metr_kriging = general_satcn(avg_d, device, layers = layers, t_kernel = t_kernel, channels = channels) # very simple model, without tuning feature number aggregation, scaler
metr_kriging.to(device)
optimizer = optim.Adam(metr_kriging.parameters(), lr=0.001, weight_decay=0.0001)

'''
Training and test
'''
Atra2 = torch.Tensor(Atra2).to(device)
Ate2 = torch.Tensor(Ate2).to(device)
print("start training...",flush=True)
MAE_list = []
for i in range(1, episode+1):
    train_loss = []
    Traloader.shuffle()
    for iter, (x, A) in enumerate(Traloader.get_iterator()):
        x, y, Atra1 = sample_processing(x, A, least_k, N_v, (layers + 1) *  (t_kernel - 1))
        x, y = Scaler.transform(x), Scaler.transform(y) 
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        Atra1 = torch.Tensor(Atra1).to(device)
        metr_kriging.train()
        optimizer.zero_grad()
        pre = metr_kriging(trainx, Atra2, Atra1) 
        
        loss = masked_mae(pre, trainy, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(metr_kriging.parameters(), 2)
        optimizer.step()
        train_loss.append(loss.item())
        
        
        #if iter % 50 == 0:
    output = []
    
    print("start testing...",flush=True)
    for iter, (x, A, y) in enumerate(Teloader.get_iterator()):
        x, y = Scaler.transform(x), Scaler.transform(y) 
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        Ate1 = torch.Tensor(A).to(device)
        metr_kriging.eval()
        with torch.no_grad():
            pred = metr_kriging(testx, Ate2, Ate1)
            pred = Scaler.inverse_transform(pred)
        output.append(pred[:, :, :, Tedata['u']])
        
    yhat = torch.cat(output,dim=0)
    yhat = yhat[:realy.size(0),...]
    
    test_mae = masked_mae(yhat, realy, 0)
    test_rmse = masked_rmse(yhat, realy, 0)
    
    log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(test_mae,test_rmse))
    
    MAE_list.append(test_mae)
    if test_mae == min(MAE_list):
        print('saved best model at' + str(i))
        best_model = copy.deepcopy(metr_kriging.state_dict())

metr_kriging.load_state_dict(best_model)
torch.save(metr_kriging, 'metr_la' + str(training_rate) + str(sample_rate) + str(layers) + str(least_k) + str(missing_rate) + str(channels) + '.pth')
            
            
b = torch.ones(8,)
for i in range(8):
    b[i] = MAE_list[12+i]
   
MAE_mean = torch.mean(b)
MAE_std = torch.std(b)

log = 'mean MAE: {:.4f}, std MAE: {:.4f}'
print(log.format(MAE_mean,MAE_std))

metr_kriging = torch.load('metr_la0.50.5120128.pth')

output = []
print("start testing...",flush=True)
for iter, (x, A, y) in enumerate(Teloader.get_iterator()):
    x, y = Scaler.transform(x), Scaler.transform(y) 
    testx = torch.Tensor(x).to(device)
    testy = torch.Tensor(y).to(device)
    Ate1 = torch.Tensor(A).to(device)
    metr_kriging.eval()
    with torch.no_grad():
        pred = metr_kriging(testx, Ate2, Ate1)
        pred = Scaler.inverse_transform(pred)
    output.append(pred[:, :, :, Tedata['u']])
    
yhat = torch.cat(output,dim=0)
yhat = yhat[:realy.size(0),...]

test_mae = masked_mae(yhat, realy, 0)
test_rmse = masked_rmse(yhat, realy, 0)

log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}'
print(log.format(test_mae,test_rmse))  
            
            
            
        
