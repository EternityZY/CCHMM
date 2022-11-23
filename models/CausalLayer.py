# !/usr/bin/env python
# -*- coding:utf-8 -*-

import math

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
import networkx as nx

from tools.utils import _h_A


class NonlinearTransforms(nn.Module):
    """docstring for InvertiblePrior

        1/(1+exp{-(wx+b)})
    """

    def __init__(self, in_dim, out_dim, activation):
        super(NonlinearTransforms, self).__init__()

        self.FC1 = nn.Linear(in_dim, out_dim//4)
        self.FC2 = nn.Linear(out_dim//4, out_dim)

        # self.activation = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, eps):
        '''

        @param eps:
        @return:
        '''
        # o = F.linear(eps, self.W, self.bias)
        # o = 1/(1+torch.exp(-o))

        o = self.FC2(self.activation(self.FC1(eps)))
        # o = self.sigmoid(self.FC1(eps))
        return o

class SCM(nn.Module):
    def __init__(self, in_dim, hidden_dim, hidden_num, scm_type='linear',
                 nonlinear_activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num  # num_label
        self.nonlinear_activation = nonlinear_activation
        self.Weight_DAG = nn.Parameter(torch.randn((hidden_num, hidden_num)), requires_grad=True)
        self.alpha = 3

        # Elementwise nonlinear mappings
        if scm_type == 'linear':
            transforms = nn.Identity()
        elif scm_type == 'nonlinear':
                transforms = NonlinearTransforms
        else:
            raise NotImplementedError("Not supported prior network.")

        for i in range(self.hidden_num):
            setattr(self, "transforms%d" % i, transforms(in_dim, hidden_dim, self.nonlinear_activation))

    def generate_z(self, eps):
        '''
        h = (I-A.T)^{-1}*eps

        z = f(h)
        @param eps: [batch, num, dim]
        @return:
        '''
        # to amplify the value of A and accelerate convergence.
        self.amplif_Weight_DAG = F.relu(torch.tanh(self.alpha*self.Weight_DAG))

        I = torch.eye(self.amplif_Weight_DAG.shape[0], device=self.Weight_DAG.device)
        DAG_normalized = torch.inverse(I - self.amplif_Weight_DAG.t())
        h = torch.matmul(DAG_normalized, eps)
        h = torch.split(h, 1, dim=2)
        zs = []
        for i in range(self.hidden_num):
            zs.append(getattr(self, "transforms%d" % i)(h[i]))

        z = torch.cat(zs, dim=2)

        return z

    def cal_loss(self):

        loss = 0
        # add A loss
        one_adj_A = self.amplif_Weight_DAG

        # compute h(A)
        h_A = _h_A(one_adj_A, one_adj_A.shape[0])
        loss += h_A + 0.5 * h_A * h_A + 100. * torch.trace(
                one_adj_A * one_adj_A)  # +  0.01 * torch.sum(variance * variance)

        return loss

    def forward(self, eps):
        '''

        @param eps:
        @param z:
        @return:
        '''
        z = self.generate_z(eps)           # n x d  （B*4）
        # nonlinear transform
        return z
