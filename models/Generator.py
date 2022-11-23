# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Generator, self).__init__()

        self.start = nn.Linear(in_dim, hidden_dim)

        self.block1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim//4, hidden_dim))

        self.FC = nn.Linear(hidden_dim, out_dim)


    def forward(self, z):
        batch, node, hidden_dim = z.shape
        # x = z.reshape(-1, hidden_dim)
        xs = self.start(z)

        x1 = self.block1(xs)
        out = x1 + xs

        out = self.FC(out)

        return out