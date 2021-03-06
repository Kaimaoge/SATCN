# -*- coding: utf-8 -*-
"""
Created on Tue May 18 23:36:09 2021

@author: Kaima
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import os
import joblib
import zipfile

from sample_gene import test_processing
from math import radians, cos, sin, asin, sqrt

rand = np.random.RandomState(0)

def get_0_1_array(array,rate=0.2):

    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    rand.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000

def load_metr_La_adj():
    with open('metr/graph_sensor_ids.txt') as f:
        sensor_ids = f.read().strip().split(',')

    distance_df = pd.read_csv('metr/distances_la_2012.csv', dtype={'from': 'str', 'to': 'str'})
    
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    
    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
    
    for i in range(dist_mx.shape[0]):
        for j in range(dist_mx.shape[1]):
            dist_mx[i, j] = np.min((dist_mx[i, j], dist_mx[j, i]))
    
    d_max = 12000
    dist_mx = dist_mx / d_max
    
    A = 1 - dist_mx
    A[np.isinf(A)] = 0
    
    A_new = (A + A.transpose())/2
    return A_new

def generate_ushcn_data():
    pos = []
    Utensor = np.zeros((1218, 120, 12, 2))
    Omissing = np.ones((1218, 120, 12, 2))
    with open("ushcn/Ulocation", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open("ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9*i:22 + 9*i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp
                        temp = temp + 1   
            with open("ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.tavg", "r") as ft:
                temp = 0
                for linet in ft.readlines():
                    if int(linet[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linet[17 + 9*i:22 + 9*i]
                            t_temp = int(str_temp)
                            if t_temp == -9999:
                                Omissing[loc, temp, i, 1] = 0
                            else:
                                Utensor[loc, temp, i, 1] = t_temp
                        temp = temp + 1    
            loc = loc + 1
            
    latlon =np.loadtxt("ushcn/latlon.csv",delimiter=",")
    sim = np.zeros((1218,1218))

    for i in range(1218):
        for j in range(1218):
            sim[i,j] = haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0]) #RBF
    #sim = np.exp(-sim/10000/10) # directly read USHCN data

    joblib.dump(Utensor,'ushcn/Utensor.joblib')
    joblib.dump(Omissing,'ushcn/Omissing.joblib')
    joblib.dump(sim,'ushcn/sim.joblib')  

def load_udata():
    if (not os.path.isfile("ushcn/Utensor.joblib")
            or not os.path.isfile("ushcn/sim.joblib")):
        with zipfile.ZipFile("ushcn/ushcn.v2.5.5.20191231.zip", 'r') as zip_ref:
            zip_ref.extractall("ushcn/ushcn.v2.5.5.20191231/")
        generate_ushcn_data()
    X = joblib.load('ushcn/Utensor.joblib')
    A = joblib.load('ushcn/sim.joblib')
    Omissing = joblib.load('ushcn/Omissing.joblib')
    X = X.astype(np.float32)
    return A,X,Omissing


def metr_La_processing(training_rate, sample_rate, missing_rate, seq_length, t_offset, least_k):
    X = np.load("metr/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)[:, 0, :]
    A = load_metr_La_adj()
    
 # Fixed random output
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
    tseq_length = seq_length - t_offset
    
    xtra = []
    Atra1 = []
    A_tra1_base = np.expand_dims(A_tra2, 0).repeat(training_set.shape[1], 0)
    for t in range((training_set.shape[1]  - t_offset - tseq_length)//tseq_length):
        xtra.append(training_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length])
        A_temp = A_tra1_base[t*tseq_length:t_offset + t*tseq_length + tseq_length]

        pos = np.where(training_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length] == 0)
        A_temp[pos[1], :, pos[0]] = 0
        Atra1.append(A_temp)
    
    xtra = np.stack(xtra, axis=0)
    xtra = np.expand_dims(xtra, 1).transpose([0, 1, 3, 2])
    Atra1 = np.stack(Atra1, axis=0)
    np.savez_compressed(
            'metr/training_data.npz',
            x=xtra,
            A1=Atra1,
            A2=A_tra2
        )
    
    xte = []
    Ate1 = []
    yte = []
    A_te1_base = np.expand_dims(A_te1, 0).repeat(test_set.shape[1], 0)

    
    
    for t in range((test_set.shape[1] - t_offset - tseq_length)//tseq_length):
        xte.append(test_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length])
        yte.append(test_truth[:, t_offset + t*tseq_length:t_offset + t*tseq_length + tseq_length])
        A_temp = A_te1_base[t*tseq_length:t_offset + t*tseq_length + tseq_length]

        pos = np.where(test_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length] == 0)
        A_temp[pos[1], :, pos[0]] = 0
        A_temp = test_processing(A_temp, least_k) # avoid ranking during test
        Ate1.append(A_temp) 
    
    xte = np.stack(xte, axis=0)
    xte = np.expand_dims(xte, 1).transpose([0, 1, 3, 2])
    yte = np.stack(yte, axis=0)
    yte = np.expand_dims(yte, 1).transpose([0, 1, 3, 2])
    Ate1 = np.stack(Ate1, axis=0)
    np.savez_compressed(
            'metr/test_data.npz',
            x=xte,
            y=yte,
            u=list(unknow_set),
            A1=Ate1,
            A2=A
        )
    
    
def ushcn_processing(training_rate, sample_rate, missing_rate, seq_length, t_offset, least_k):
    dist_mx,X,Omissing = load_udata()
    d_max = 5000000
    dist_mx = dist_mx / d_max
    A_new = 1 - dist_mx
    A_new[np.isinf(A_new)] = 0    
    A_new = A_new - np.eye(A_new.shape[0],)
    A = A_new
    for i in range(A_new.shape[0]):
        A_new[i, i] = 0    
    X = X[:,:,:,0]
    X = X.reshape(1218,120*12)
    X = X/100    
    Omissing = Omissing[:,:,:,0]
    Omissing = Omissing.reshape(1218,120*12)
    X[Omissing == 0] = 0

    
 # Fixed random output
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
    tseq_length = seq_length - t_offset
    
    xtra = []
    Atra1 = []
    A_tra1_base = np.expand_dims(A_tra2, 0).repeat(training_set.shape[1], 0)
    for t in range((training_set.shape[1]  - t_offset - tseq_length)//tseq_length):
        xtra.append(training_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length])
        A_temp = A_tra1_base[t*tseq_length:t_offset + t*tseq_length + tseq_length]

        pos = np.where(training_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length] == 0)
        A_temp[pos[1], :, pos[0]] = 0
        Atra1.append(A_temp)
    
    xtra = np.stack(xtra, axis=0)
    xtra = np.expand_dims(xtra, 1).transpose([0, 1, 3, 2])
    Atra1 = np.stack(Atra1, axis=0)
    np.savez_compressed(
            'ushcn/training_data.npz',
            x=xtra,
            A1=Atra1,
            A2=A_tra2
        )
    
    del Atra1
    del A_tra2
    
    xte = []
    Ate1 = []
    yte = []
    A_te1_base = np.expand_dims(A_te1, 0).repeat(test_set.shape[1], 0)

    
    
    for t in range((test_set.shape[1] - t_offset - tseq_length)//tseq_length):
        xte.append(test_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length])
        yte.append(test_truth[:, t_offset + t*tseq_length:t_offset + t*tseq_length + tseq_length])
        A_temp = A_te1_base[t*tseq_length:t_offset + t*tseq_length + tseq_length]

        pos = np.where(test_set[:, t*tseq_length:t_offset + t*tseq_length + tseq_length] == 0)
        A_temp[pos[1], :, pos[0]] = 0
        A_temp = test_processing(A_temp, least_k) # avoid ranking during test
        Ate1.append(A_temp) 
    
    xte = np.stack(xte, axis=0)
    xte = np.expand_dims(xte, 1).transpose([0, 1, 3, 2])
    yte = np.stack(yte, axis=0)
    yte = np.expand_dims(yte, 1).transpose([0, 1, 3, 2])
    Ate1 = np.stack(Ate1, axis=0)
    np.savez_compressed(
            'ushcn/test_data.npz',
            x=xte,
            y=yte,
            u=list(unknow_set),
            A1=Ate1,
            A2=A
        )
    
    
        
class TraDataLoader(object):
    def __init__(self, xs, As, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            A_padding = np.repeat(As[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            As = np.concatenate([As, A_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.As = As


    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, As = self.xs[permutation], self.As[permutation]
        self.xs = xs
        self.As = As


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                A_i = self.As[start_ind: end_ind, ...]
                yield (x_i, A_i)
                self.current_ind += 1

        return _wrapper()    
    
    
class TeDataLoader(object):
    def __init__(self, xs, ys, As):
        self.current_ind = 0
        self.size = len(xs)
        self.xs = xs
        self.As = As
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.size:
                start_ind =  self.current_ind
                end_ind = min(self.size, (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                A_i = self.As[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, A_i, y_i)
                self.current_ind += 1

        return _wrapper()    
       
    
    
    
    
    
    
    



