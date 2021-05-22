# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:10:42 2021

@author: Kaima
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from st_aggregators_new import AGGREGATORS
from st_aggregators_mask import AGGREGATORS_MASK
from scaler import SCALERS

#A_to_A = A[list(unknow_set), :][:, list(know_set)]
#torch.set_default_tensor_type(torch.cuda.FloatTensor)

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
    

class tcn_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="linear", dropout = 0.1): # not necessary to use GLU for Kriging, actually linear activation is slightly better.
        super(tcn_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)
        self.dropout = dropout
            

    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, num_variables, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_features, num_timesteps - kt, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            h = (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            return F.dropout(h, self.dropout, training=self.training)
        if self.act == "sigmoid":
            h = torch.sigmoid(self.conv(x) + x_in)
            return F.dropout(h, self.dropout, training=self.training)
        h = self.conv(x) + x_in
        return F.dropout(h, self.dropout, training=self.training)
    
class STower(nn.Module):
    """
    Spatil aggragation layer applies principle aggragation on the spatial dimension
    """      
    def __init__(self, in_features, out_features, aggregators, scalers,  avg_d,
                 device = 'cuda', masking = False, dropout = 0.1):
        super(STower, self).__init__()
        
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.aggregators = aggregators
        self.scalers = scalers
        
        self.Theta_po = nn.Parameter(torch.FloatTensor(len(aggregators) * len(scalers) * self.in_features, self.out_features)).cuda()
        self.bias_po = nn.Parameter(torch.FloatTensor(self.out_features)).cuda() 
        self.reset_parameters()
        self.avg_d = avg_d   
        self.masking = masking
        self.dropout = dropout
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta_po.shape[1])
        self.Theta_po.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.bias_po.shape[0])
        self.bias_po.data.uniform_(-stdv, stdv)
        
    def forward(self, X, adj):
        """
        :param X: Input data of shape (batch_size, in_features, num_timesteps, in_nodes)
        :adj: The adjacency (num_nodes, num_nodes) missing_nodes (The kriging target nodes )
        :return: Output data of shape (batch_size, num_nodes, num_timesteps, out_features)
        """ 
        I = X.permute([0, 2, 3, 1])
        if self.masking:
            (_, _, N, _) = adj.shape
        else:
            (N, _) = adj.shape
        adj = adj 
        if self.masking:
            m = torch.cat([AGGREGATORS_MASK[aggregate](I, adj, device = self.device) for aggregate in self.aggregators], dim=3)
            m[torch.isnan(m)] = 6
            m[torch.isinf(m)] = 0
            m[m > 6] = 6          
        else:
            m = torch.cat([AGGREGATORS[aggregate](I, adj, device = self.device) for aggregate in self.aggregators], dim=3)              
        m = torch.cat([SCALERS[scale](m, adj, avg_d=self.avg_d) for scale in self.scalers], dim=3)        
        out = torch.einsum("btji,io->btjo", [m, self.Theta_po])
        out += self.bias_po
        out = F.dropout(out, self.dropout, training=self.training)
        return out.permute([0, 3, 1, 2])
    

class SAGE(nn.Module):
    """
    A simple graphSAGE layer
    """ 
    def __init__(self, in_features, out_features, aggregators = ['mean'],
                 device = 'cuda'):
        super(SAGE, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.aggregators = aggregators
        self.Theta_po = nn.Parameter(torch.FloatTensor(self.in_features, self.out_features)).cuda()
        self.bias_po = nn.Parameter(torch.FloatTensor(self.out_features)).cuda() 
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta_po.shape[1])
        self.Theta_po.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.bias_po.shape[0])
        self.bias_po.data.uniform_(-stdv, stdv)
        
    def forward(self, X, adj):
        """
        :param X: Input data of shape (batch_size, in_features, num_timesteps, in_nodes)
        :adj: The adjacency (num_nodes, num_nodes) missing_nodes (The kriging target nodes )
        :return: Output data of shape (batch_size, num_nodes, num_timesteps, out_features)
        """ 
        I = X.permute([0, 2, 3, 1])
        (N, _) = adj.shape
        m = AGGREGATORS[self.aggregators[0]](I, adj, device = self.device)               
        #m = torch.cat([SCALERS[scale](m, adj, avg_d=self.avg_d) for scale in self.scalers], dim=3)        
        out = torch.einsum("btji,io->btjo", [m, self.Theta_po])
        out += self.bias_po
        return torch.cat((out.permute([0, 3, 1, 2]), X), dim = 1)
    
class double_t_pna(nn.Module):
    """
    :param c_k [a, b, c, d, e], a: input_feature: k (in_variables + 1)
    b: out features of the first TCN layer
    c: out features of tbe GAT layer
    d: out features of the second TCN layer
    e: in_variables
    time_steps: length of input time length
    kt [f, h]: convolution length of the tcn layers
    ks: convolution length of GAT layer
    aggragators, scalers
    """  
    def __init__(self, c_k, kt, aggragators, scalers, avg_d, device, masking = True):
        super(double_t_pna, self).__init__()
        self.s_layer0 = STower(1, c_k[1], aggragators, ['identity'], avg_d, device, masking)
        self.t_layer0 = tcn_layer(kt[1], c_k[0], c_k[2])
        self.s_layer = STower(c_k[2], c_k[3], aggragators, scalers, avg_d, device)
        self.t_layer1 = tcn_layer(kt[1], c_k[3], c_k[4])
        self.out_conv = nn.Conv2d(c_k[4], c_k[5], (1, 1), 1)
        
    def forward(self, x, Lk, Lk_mask):
        x = self.s_layer0(x, Lk_mask)
        x = self.t_layer0(x)
        x = self.s_layer(x, Lk)
        x = self.t_layer1(x)
        y = self.out_conv(x)
        return y
    
    
class general_satcn(nn.Module):

    def __init__(self, avg_d, device, in_variables = 1, layers = 1, channels=32, t_kernel = 2, aggragators = ['mean', 'softmin', 'softmax', 'normalised_mean', 'std'], 
                 scalers = ['identity', 'amplification', 'attenuation'], masking = True, dropout = 0):
        super(general_satcn, self).__init__()
        self.s_layer0 = STower(in_variables, channels, aggragators + ['distance','d_std'], ['identity'],avg_d, device, masking, dropout)
        self.t_layer0 = tcn_layer(t_kernel, channels, channels, dropout)
        self.s_convs = nn.ModuleList()
        self.t_convs = nn.ModuleList()
        self.layers = layers
        for i in range(layers):
            self.s_convs.append(STower(channels, channels, aggragators, scalers, avg_d, device, False, dropout))
            self.t_convs.append(tcn_layer(t_kernel, channels, channels, dropout))
        self.out_conv = nn.Conv2d(channels, in_variables, (1, 1), 1)
        
    def forward(self, x, Lk, Lk_mask):
        x = self.s_layer0(x, Lk_mask)
        x = self.t_layer0(x)
        for i in range(self.layers):
            x = self.s_convs[i](x, Lk)
            x = self.t_convs[i](x)
        y = self.out_conv(x)
        return y
