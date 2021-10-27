"""
Created on Sun May 30 18:45:41 2021

@author: Kaima
"""

import numpy as np
import copy
import torch
import torch.optim as optim
from datasets import load_udata, TraDataLoader, TeDataLoader, get_0_1_array
from model import general_satcn
from sample_gene import sadj_transform, sample_processing, ZerooneScaler, test_processing
from loss import masked_rmse, masked_mae

rand = np.random.RandomState(0)
'''
Hyper parameters
'''
device = 'cuda'
training_rate = 0.5 # training time ratio
sample_rate = 0.5 # number of missing node
missing_rate = 0.5# MIssing data in the observed sensors
seq_length = 7 # INput length of the sequence

layers = 1 # number of unmasked spatial aggregation and temporal convolution
t_kernel = 4 # the length of temporal convolution kernel
least_k = 3 # the number of neighbors
N_v = 150 # the number of masked nodes for training

batch_size = 4 # training batch size
episode = 50 # max training epoch




'''
Generate Dataset
'''
print("generate dataset...",flush=True)
#ushcn_processing(training_rate, sample_rate, missing_rate, seq_length, (layers + 1) *  (t_kernel - 1), least_k)

dist_mx,X,Omissing = load_udata()
d_max = 5000000
dist_mx = dist_mx / d_max
A_new = 1 - dist_mx
A_new[np.isinf(A_new)] = 0    
A_new = A_new #- np.eye(A_new.shape[0],)
A = A_new
# for i in range(A_new.shape[0]):
#     A_new[i, i] = 0    
X = X[:,:,:,0]
X = X.reshape(1218,120*12)
X = X/100    
Omissing = Omissing[:,:,:,0]
Omissing = Omissing.reshape(1218,120*12)
X[Omissing == 0] = 0

unknow_set = rand.choice(list(range(0,X.shape[0])),int(sample_rate * X.shape[0]),replace=False)
unknow_set = set(unknow_set)
full_set = set(range(0,X.shape[0]))        
know_set = full_set - unknow_set
split_line1 = int(X.shape[1] * training_rate)
training_set = X[list(know_set), :split_line1]
test_set = X[:, split_line1:]
test_truth = test_set.copy()
test_set[list(unknow_set), :] = 0

if missing_rate > 0:
    training_set = training_set * get_0_1_array(training_set, missing_rate)
    test_set = test_set * get_0_1_array(test_set, missing_rate)

A_tra2 = A[:, list(know_set)][list(know_set), :]    
A_te1 = A.copy()
A_te1[:, list(unknow_set)] = 0
# ensure every time point has been evaluated in the test dataset
tseq_length = seq_length - (layers + 1) *  (t_kernel - 1)
A_tra1_base = np.expand_dims(A_tra2, 0).repeat(seq_length, 0)
Atra2 = sadj_transform(A_tra2, least_k)
Ate2 = sadj_transform(A, least_k)
avg_d = {}
avg_d['log'] = torch.tensor(np.mean(np.log(np.sum(Atra2, axis = 0) + 1)))

A_te1_base = np.expand_dims(A_te1, 0).repeat(seq_length, 0)

Scaler = ZerooneScaler(training_set.max())

'''
Define model
'''
ushcn_kriging = general_satcn(avg_d, device, layers = layers, t_kernel = t_kernel) # very simple model, without tuning feature number aggregation, scaler
ushcn_kriging.to(device)
optimizer = optim.Adam(ushcn_kriging.parameters(), lr=0.001, weight_decay=0.0001)

'''
Training and test
'''
Atra2 = torch.Tensor(Atra2).to(device)
Ate2 = torch.Tensor(Ate2).to(device)
print("start training...",flush=True)
MAE_list = []
for i in range(1, episode+1):
    train_loss = []

    for t in range((training_set.shape[1]  -  (layers + 1) *  (t_kernel - 1) - tseq_length)//(tseq_length * batch_size)):
        t_random = np.random.randint(0, high=(training_set.shape[1]  -  (layers + 1) *  (t_kernel - 1) - tseq_length), size=batch_size, dtype='l')
        x = []
        A = []
        for j in range(batch_size):
            A_temp = A_tra1_base.copy()
            x.append(training_set[:, t_random[j]:(layers + 1) *  (t_kernel - 1) + t_random[j] + tseq_length])
            pos = np.where(training_set[:, t_random[j]:(layers + 1) *  (t_kernel - 1) + t_random[j] + tseq_length] == 0)
            A_temp[pos[1], :, pos[0]] = 0
            A.append(A_temp)
            
        x = np.stack(x, axis=0)
        x = np.expand_dims(x, 1).transpose([0, 1, 3, 2])
        A = np.stack(A, axis=0)
        x, y, Atra1 = sample_processing(x, A, least_k, N_v, (layers + 1) *  (t_kernel - 1))
        x, y = Scaler.transform(x), Scaler.transform(y) 
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        Atra1 = torch.Tensor(Atra1).to(device)
        ushcn_kriging.train()
        optimizer.zero_grad()
        pre = ushcn_kriging(trainx, Atra2, Atra1) 
        
        loss = masked_mae(pre, trainy, 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ushcn_kriging.parameters(), 2)
        optimizer.step()
        train_loss.append(loss.item())
        
        
        #if iter % 50 == 0:
    output = []
    realy = []
    print("start testing...",flush=True)
    for t in range((test_set.shape[1] - (layers + 1) *  (t_kernel - 1) - tseq_length)//tseq_length):
        x = test_set[:, t*tseq_length:(layers + 1) *  (t_kernel - 1) + t*tseq_length + tseq_length]
        y = test_truth[:, (layers + 1) *  (t_kernel - 1) + t*tseq_length:(layers + 1) *  (t_kernel - 1) + t*tseq_length + tseq_length]
        x = np.expand_dims(x, [0, 1]).transpose([0, 1, 3, 2])
        y = np.expand_dims(y, [0, 1]).transpose([0, 1, 3, 2])
        A_temp = A_te1_base.copy()
        pos = np.where(test_set[:, t:(layers + 1) *  (t_kernel - 1) + t + tseq_length] == 0)
        A_temp[pos[1], :, pos[0]] = 0
        A_temp = test_processing(A_temp, least_k)
        A = np.expand_dims(A_temp, 0)
        x = Scaler.transform(x)
        testx = torch.Tensor(x).to(device)
        testy = torch.Tensor(y).to(device)
        Ate1 = torch.Tensor(A).to(device)
        ushcn_kriging.eval()
        with torch.no_grad():
            pred = ushcn_kriging(testx, Ate2, Ate1)
            pred = Scaler.inverse_transform(pred)
        output.append(pred[:, :, :, list(unknow_set)])
        realy.append(testy[:, :, :, list(unknow_set)])
    yhat = torch.cat(output,dim=0)
    realy = torch.cat(realy,dim=0)
    # yhat = yhat[:realy.size(0),...]
    
    test_mae = masked_mae(yhat, realy, 0)
    test_rmse = masked_rmse(yhat, realy, 0)
    
    log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(test_mae,test_rmse))
    
    MAE_list.append(test_mae)
    if test_mae == min(MAE_list):
        print('saved best model at' + str(i))
        best_model = copy.deepcopy(ushcn_kriging.state_dict())

ushcn_kriging.load_state_dict(best_model)
torch.save(ushcn_kriging, 'ushcn' + str(training_rate) + str(sample_rate) + str(layers) + str(least_k) + str(missing_rate) + '.pth')
            
