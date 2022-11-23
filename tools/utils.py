# !/usr/bin/env python
# -*- coding:utf-8 -*-


from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import math
import mkl_random
from scipy.sparse import linalg
from torch.optim.lr_scheduler import MultiStepLR
import colorsys
import random


class StepLR2(MultiStepLR):
    """StepLR with min_lr"""

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6):
        """

        :optimizer: TODO
        :milestones: TODO
        :gamma: TODO
        :last_epoch: TODO
        :min_lr: TODO

        """
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate


class StandardScaler:
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


class MinMaxNormalization:
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    def __init__(self, max, min):
        self.max = max
        self.min = min

    def transform(self, data):
        data = 1. * (data - self.min) / (self.max - self.min)
        data = 2. * data - 1.
        return data

    def inverse_transform(self, data):
        data = (data + 1) / 2
        data = data * (self.max - self.min) + self.min
        return data

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sparse.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def kl_normal_log(mu, logvar, mu_prior, logvar_prior):
    var = logvar.exp()
    var_prior = logvar_prior.exp()

    element_wise = 0.5 * (
                torch.log(var_prior) - torch.log(var) + var / var_prior + (mu - mu_prior).pow(2) / var_prior - 1)
    kl = element_wise.mean(-1)

    kl = torch.mean(kl, dim=1)
    kl = torch.mean(kl, dim=0)

    return kl

def matrix_poly(matrix, d):
    x = torch.eye(d).to(matrix.device) + torch.div(matrix, d)
    return torch.matrix_power(x, d)


def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


def make_saved_dir(saved_dir, use_time=3):
    """
    :param saved_dir:
    :return: {saved_dir}/{%m-%d-%H-%M-%S}
    """
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if use_time == 1:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d_%H:%M'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
    elif use_time == 2:
        saved_dir = os.path.join(saved_dir, datetime.now().strftime('%m-%d'))
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

    return saved_dir
