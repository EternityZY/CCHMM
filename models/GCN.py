# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class gconv(nn.Module):

    def __init__(self):
        super(gconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('hw, bwc->bhc', (A, x))
        return x.contiguous()


class linear(nn.Module):

    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = nn.Linear(c_in, c_out, bias)

    def forward(self, x):
        return F.relu(self.mlp(x), inplace=True)

class mixpropGCN(nn.Module):
    '''
    '''

    def __init__(self, in_dim, out_dim, gdep, dropout_prob=0, alpha=0.3, norm_adj=None):
        super(mixpropGCN, self).__init__()
        self.nconv = gconv()
        self.mlp = linear((gdep + 1) * in_dim, out_dim)
        self.gdep = gdep
        self.dropout_prob = dropout_prob
        self.alpha = alpha
        self.norm_adj = norm_adj

    def forward(self, x, norm_adj=None):
        if norm_adj == None:
            norm_adj = self.norm_adj
        h = x
        out = [x]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(norm_adj, h)
            out.append(h)
        ho = torch.cat(out, dim=-1)
        ho = self.mlp(ho)
        if self.dropout_prob > 0:
            ho = F.dropout(ho, self.dropout_prob)
        return ho

