# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:48:50 2021

@author: Kaima
"""

from __future__ import division
import heapq
import random

def sadj_transform(adj, least_k):
    (N, _) = adj.shape
    for i in range(N):
        small_k = heapq.nlargest(least_k, adj[i, :])[-1]
        adj[i, :][adj[i, :]<small_k] = 0
            
    return adj


def sample_processing(x, A1, least_k, N_v, t_reduce):
    (B, _, T, N) = x.shape
    y = x.copy()
    for b in range(B):
        for t in range(T):
            if N_v > 0:
                unknow_mask = random.sample(range(0,N),N_v) 
                x[b, :, t, list(unknow_mask)] = 0
                A1[b, t, :, list(unknow_mask)] = 0
            A1[b, t, :, :] = sadj_transform(A1[b, t, :, :], least_k)
           
    return x, y[:, :, t_reduce:,:], A1

def test_processing(A1, least_k):
    (T, _, _) = A1.shape

    for t in range(T):
       A1[t, :, :] = sadj_transform(A1[t, :, :], least_k)
           
    return A1



class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
    
class ZerooneScaler():
    """
    Standard the input
    """

    def __init__(self, max_):
        self.max = max_

    def transform(self, data):
        return data / self.max

    def inverse_transform(self, data):
        return data * self.max 
    

    