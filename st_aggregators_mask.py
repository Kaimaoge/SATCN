# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:54:42 2021

@author: Kaima
"""

import math
import torch

EPS = 1e-5

# masked spatial aggregation, adj is T x N x N
# each aggregator is a function taking as input X (B x T x N x Din), adj (B x T x N x N), device
# returning the aggregated value of X (B x T x N x Din) for each dimension

def aggregate_mean(X, adj, device = 'cpu'):
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True)

    X_sum =  torch.einsum("btji,btoj->btoi", [X, adj_])
    X_mean = torch.div(X_sum, D)
    return X_mean

def aggregate_normalised_mean(X, adj, device='cpu'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    D = torch.sum(adj, -1, keepdim=True)

    X_sum =  torch.einsum("btji,btoj->btoi", [X, adj])
    X_mean = torch.div(X_sum, D)
    return X_mean

def aggregate_d(X, adj, device='cpu'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, 1]).cuda()
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True)
#    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device))  # D^{-1/2]
#    adj = torch.matmul(torch.matmul(rD, adj), rD)
    X_sum =  torch.einsum("btji,btoj->btoi", [P, adj])
    X_mean = torch.div(X_sum, D)
    return X_mean

def aggregate_d_var(X, adj, device='cpu'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, 1]).cuda()
    D = torch.sum(adj, -1, keepdim=True)

    X_sum = torch.einsum("btji,btoj->btoi", [P * P, adj])
    X_sum = torch.div(X_sum, D) 
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var

def aggregate_d_std(X, adj, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_d_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_var(X, adj, device='cpu'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    D = torch.sum(adj, -1, keepdim=True)

    X_sum = torch.einsum("btji,btoj->btoi", [X * X, adj])
    X_sum = torch.div(X_sum, D) 
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var

def aggregate_std(X, adj, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std

def aggregate_sum(X, adj, device='cpu'):
    # A * X    i.e. the mean of the neighbours

    X_sum =  torch.einsum("btji,btoj->btoi", [X, adj])
    return X_sum

def aggregate_softmax(X, adj, device='cpu'):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    X_sum =  torch.einsum("btji,btoj->btoi", [X, adj])
    softmax = torch.nn.functional.softmax(X_sum, dim = 2)
    return softmax

def aggregate_softmin(X, adj, device='cpu'):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, device=device)

AGGREGATORS_MASK = {'mean': aggregate_mean, 'sum': aggregate_sum, 
               'std': aggregate_std, 'var': aggregate_var,
               'normalised_mean': aggregate_normalised_mean, 'softmax': aggregate_softmax, 'softmin': aggregate_softmin, 'distance': aggregate_d, 'd_std': aggregate_d_std}