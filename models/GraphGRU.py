# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from models.GCN import mixpropGCN

class GraphGRU(nn.Module):
    '''
    '''

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 norm_adj,
                 gcn_depth=2,
                 dropout_type='zoneout',
                 dropout_prob=0.3,
                 alpha=0.3):
        super(GraphGRU, self).__init__()

        self.in_channels = in_dim
        self.hidden_dim = hidden_dim
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

        self.norm_adj = norm_adj

        self.GCN_update = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha)
        self.GCN_reset = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha)
        self.GCN_cell = mixpropGCN(in_dim+hidden_dim, hidden_dim, gcn_depth, dropout_prob, alpha=alpha)

        self.layerNorm = nn.LayerNorm([self.hidden_dim])


    def forward(self, inputs, hidden_state=None):


        batch_size, node_num, in_dim = inputs.shape
        if hidden_state == None:
            hidden_state = torch.zeros((batch_size, node_num, self.hidden_dim)).to(inputs.device)

        combined = torch.cat((inputs, hidden_state), dim=-1)

        update_gate = torch.sigmoid(self.GCN_update(combined, self.norm_adj))

        reset_gate = torch.sigmoid(self.GCN_reset(combined, self.norm_adj))

        temp = torch.cat((inputs, torch.mul(reset_gate, hidden_state)), dim=-1)
        cell_State = torch.tanh(self.GCN_cell(temp, self.norm_adj))

        next_Hidden_State = torch.mul(update_gate, hidden_state) + torch.mul(1.0 - update_gate, cell_State)

        next_hidden = self.layerNorm(next_Hidden_State)

        output = next_hidden
        if self.dropout_type == 'zoneout':
            next_hidden = self.zoneout(prev_h=hidden_state,
                                       next_h=next_hidden,
                                       rate=self.dropout_prob,
                                       training=self.training)

        return output, next_hidden



    def zoneout(self, prev_h, next_h, rate, training=True):
        """TODO: Docstring for zoneout.
        :returns: TODO

        """
        if training:
            # bernoulli: draw a value 1.
            # p = 1 -> d = 1 -> return prev_h
            # p = 0 -> d = 0 -> return next_h
            d = torch.zeros_like(next_h).bernoulli_(rate)
            next_h = d * prev_h + (1 - d) * next_h
        else:
            next_h = rate * prev_h + (1 - rate) * next_h

        return next_h