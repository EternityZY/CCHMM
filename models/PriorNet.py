# !/usr/bin/env python
# -*- coding:utf-8 -*-


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.GraphGRU import GraphGRU


class PriorNet(nn.Module):
    def __init__(self,
                 tpos_dim,
                 POI_dim,
                 weather_dim,
                 hidden_num=4,
                 hidden_dim=64,
                 gcn_depth=2,
                 norm_adj=None,
                 SCM_model=None,
                 Prior_type='GraphGRU',
                 var_type=False,
                 dropout_prob=0.3,
                 mu_type=False,
                 activation=nn.ReLU(inplace=True),
                 ):
        super(PriorNet, self).__init__()

        # self.tpos_encoder = tpos_encoder
        # self.POI_encoder = POI_encoder
        # self.weather_encoder = weather_encoder

        self.tpos_dim = tpos_dim
        self.POI_dim = POI_dim
        self.weather_dim = weather_dim

        self.SCM_model = SCM_model
        self.var_type = var_type
        self.mu_type = mu_type
        self.Prior_type = Prior_type
        # prior inference for eps_t via z_(t-1) and context
        if Prior_type == 'GraphGRU':

            self.poi_feature_fusion = nn.Linear(self.tpos_dim + self.POI_dim, hidden_dim)
            self.bike_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.taxi_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.bus_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.speed_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)

            self.prior_eps_poi_GRU = GraphGRU(hidden_dim,
                                              hidden_dim,
                                              norm_adj,
                                              gcn_depth=gcn_depth,
                                              dropout_type='None',
                                              dropout_prob=dropout_prob,
                                              alpha=0.3)

            self.prior_eps_bike_GRU = GraphGRU(hidden_dim,
                                               hidden_dim,
                                               norm_adj,
                                               gcn_depth=gcn_depth,
                                               dropout_type='None',
                                               dropout_prob=dropout_prob,
                                               alpha=0.3)

            self.prior_eps_taxi_GRU = GraphGRU(hidden_dim,
                                               hidden_dim,
                                               norm_adj,
                                               gcn_depth=gcn_depth,
                                               dropout_type='None',
                                               dropout_prob=dropout_prob,
                                               alpha=0.3)
            self.prior_eps_bus_GRU = GraphGRU(hidden_dim,
                                              hidden_dim,
                                              norm_adj,
                                              gcn_depth=gcn_depth,
                                              dropout_type='None',
                                              dropout_prob=dropout_prob,
                                              alpha=0.3)
            self.prior_eps_speed_GRU = GraphGRU(hidden_dim,
                                                hidden_dim,
                                                norm_adj,
                                                gcn_depth=gcn_depth,
                                                dropout_type='None',
                                                dropout_prob=dropout_prob,
                                                alpha=0.3)

        elif Prior_type == 'GRU':
            self.poi_feature_fusion = nn.Linear(self.tpos_dim + self.POI_dim, hidden_dim)
            self.bike_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim, hidden_dim)
            self.taxi_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim, hidden_dim)
            self.bus_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim, hidden_dim)
            self.speed_feature_fusion = nn.Linear(self.tpos_dim + self.weather_dim, hidden_dim)
            self.prior_eps_poi_GRU = nn.GRUCell(hidden_dim,
                                                hidden_dim)
            self.prior_eps_bike_GRU = nn.GRUCell(hidden_dim,
                                                 hidden_dim, )
            self.prior_eps_taxi_GRU = nn.GRUCell(hidden_dim,
                                                 hidden_dim, )
            self.prior_eps_bus_GRU = nn.GRUCell(hidden_dim,
                                                hidden_dim, )
            self.prior_eps_speed_GRU = nn.GRUCell(hidden_dim,
                                                  hidden_dim, )

        elif Prior_type == 'FC':

            self.prior_eps_poi_FC = nn.Linear(self.tpos_dim + self.POI_dim + hidden_dim, hidden_dim)
            self.prior_eps_bike_FC = nn.Linear(self.tpos_dim + self.weather_dim + hidden_dim, hidden_dim)
            self.prior_eps_taxi_FC = nn.Linear(self.tpos_dim + self.weather_dim + hidden_dim, hidden_dim)
            self.prior_eps_bus_FC = nn.Linear(self.tpos_dim + self.weather_dim + hidden_dim, hidden_dim)
            self.prior_eps_speed_FC = nn.Linear(self.tpos_dim + self.weather_dim + hidden_dim,
                                                hidden_dim)

        # prior inference for mu, logvar via z_t and eps_t
        if self.mu_type == 'share':
            self.prior_mu_FC = nn.Linear(hidden_dim, hidden_dim)

        elif self.mu_type == 'split':
            self.prior_poi_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_bike_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_taxi_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_bus_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_speed_mu_FC = nn.Linear(hidden_dim, hidden_dim)

        if self.var_type == 'share':
            self.prior_logvar_FC = nn.Linear(hidden_dim, hidden_dim)

        elif self.var_type == 'split':
            self.prior_poi_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_bike_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_taxi_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_bus_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.prior_speed_logvar_FC = nn.Linear(hidden_dim, hidden_dim)

        self._init_parameters('kaiming')

    def _init_parameters(self, args):

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if 'kaiming' in args:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif 'xavier' in args:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRUCell):
                if 'kaiming' in args:
                    nn.init.kaiming_normal_(m.weight_hh.data)
                    nn.init.kaiming_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)
                elif 'xavier' in args:
                    nn.init.xavier_normal_(m.weight_hh.data)
                    nn.init.xavier_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)

    def forward(self, tpos_cur_features, POI_features, weather_cur_features, z_pr_last):
        batch, node, _ = tpos_cur_features.shape
        # context as input
        eps_poi_input = torch.cat([POI_features, tpos_cur_features], dim=-1)
        eps_bike_input = torch.cat([POI_features, weather_cur_features, tpos_cur_features], dim=-1)
        eps_taxi_input = torch.cat([POI_features, weather_cur_features, tpos_cur_features], dim=-1)
        eps_bus_input = torch.cat([POI_features, weather_cur_features, tpos_cur_features], dim=-1)
        eps_speed_input = torch.cat([POI_features, weather_cur_features, tpos_cur_features], dim=-1)

        # calculate eps_cur
        if self.Prior_type == 'GRU':
            eps_poi_fusion = self.poi_feature_fusion(eps_poi_input)
            eps_bike_fusion = self.bike_feature_fusion(eps_bike_input)
            eps_taxi_fusion = self.taxi_feature_fusion(eps_taxi_input)
            eps_bus_fusion = self.bus_feature_fusion(eps_bus_input)
            eps_speed_fusion = self.speed_feature_fusion(eps_speed_input)

            eps_poi_cur = poi_hidden = self.prior_eps_poi_GRU(eps_poi_fusion.reshape(batch * node, -1),
                                                              z_pr_last[:, :, 0, :].reshape(batch * node, -1))
            eps_bike_cur = bike_hideen = self.prior_eps_bike_GRU(eps_bike_fusion.reshape(batch * node, -1),
                                                                 z_pr_last[:, :, 1, :].reshape(batch * node, -1))
            eps_taxi_cur = taxi_hidden = self.prior_eps_taxi_GRU(eps_taxi_fusion.reshape(batch * node, -1),
                                                                 z_pr_last[:, :, 2, :].reshape(batch * node, -1))
            eps_bus_cur = bus_hidden = self.prior_eps_bus_GRU(eps_bus_fusion.reshape(batch * node, -1),
                                                              z_pr_last[:, :, 3, :].reshape(batch * node, -1))
            eps_speed_cur = speed_hidden = self.prior_eps_speed_GRU(eps_speed_fusion.reshape(batch * node, -1),
                                                                    z_pr_last[:, :, 4, :].reshape(batch * node, -1))

            eps_poi_cur, poi_hidden = eps_poi_cur.reshape(batch, node, -1), poi_hidden.reshape(batch, node, -1)
            eps_bike_cur, bike_hideen = eps_bike_cur.reshape(batch, node, -1), bike_hideen.reshape(batch, node, -1)
            eps_taxi_cur, taxi_hidden = eps_taxi_cur.reshape(batch, node, -1), taxi_hidden.reshape(batch, node, -1)
            eps_bus_cur, bus_hidden = eps_bus_cur.reshape(batch, node, -1), bus_hidden.reshape(batch, node, -1)
            eps_speed_cur, speed_hidden = eps_speed_cur.reshape(batch, node, -1), speed_hidden.reshape(batch, node, -1)

        elif self.Prior_type == 'GraphGRU':
            eps_poi_fusion = self.poi_feature_fusion(eps_poi_input)
            eps_bike_fusion = self.bike_feature_fusion(eps_bike_input)
            eps_taxi_fusion = self.taxi_feature_fusion(eps_taxi_input)
            eps_bus_fusion = self.bus_feature_fusion(eps_bus_input)
            eps_speed_fusion = self.speed_feature_fusion(eps_speed_input)

            eps_poi_cur, poi_hidden = self.prior_eps_poi_GRU(eps_poi_fusion, z_pr_last[:, :, 0, :])
            eps_bike_cur, bike_hideen = self.prior_eps_bike_GRU(eps_bike_fusion, z_pr_last[:, :, 1, :])
            eps_taxi_cur, taxi_hidden = self.prior_eps_taxi_GRU(eps_taxi_fusion, z_pr_last[:, :, 2, :])
            eps_bus_cur, bus_hidden = self.prior_eps_bus_GRU(eps_bus_fusion, z_pr_last[:, :, 3, :])
            eps_speed_cur, speed_hidden = self.prior_eps_speed_GRU(eps_speed_fusion, z_pr_last[:, :, 4, :])

        elif self.Prior_type == 'FC':
            eps_poi_cur = poi_hidden = self.prior_eps_poi_FC(torch.cat([eps_poi_input, z_pr_last[:, :, 0, :]], dim=-1))
            eps_bike_cur = bike_hideen = self.prior_eps_bike_FC(
                torch.cat([eps_bike_input, z_pr_last[:, :, 1, :]], dim=-1))
            eps_taxi_cur = taxi_hidden = self.prior_eps_taxi_FC(
                torch.cat([eps_taxi_input, z_pr_last[:, :, 2, :]], dim=-1))
            eps_bus_cur = bus_hidden = self.prior_eps_bus_FC(torch.cat([eps_bus_input, z_pr_last[:, :, 3, :]], dim=-1))
            eps_speed_cur = speed_hidden = self.prior_eps_speed_FC(
                torch.cat([eps_speed_input, z_pr_last[:, :, 4, :]], dim=-1))

        eps = torch.cat([eps_poi_cur.unsqueeze(dim=2),
                         eps_bike_cur.unsqueeze(dim=2),
                         eps_taxi_cur.unsqueeze(dim=2),
                         eps_bus_cur.unsqueeze(dim=2),
                         eps_speed_cur.unsqueeze(dim=2), ], dim=2)

        z = self.SCM_model(eps)

        # calculate mu
        if self.mu_type == 'share':
            z_mu = self.prior_mu_FC(z)

        elif self.mu_type == 'split':
            poi_mu = self.prior_poi_mu_FC(z[:, :, 0, :]).unsqueeze(dim=2)
            bike_mu = self.prior_bike_mu_FC(z[:, :, 1, :]).unsqueeze(dim=2)
            taxi_mu = self.prior_taxi_mu_FC(z[:, :, 2, :]).unsqueeze(dim=2)
            bus_mu = self.prior_bus_mu_FC(z[:, :, 3, :]).unsqueeze(dim=2)
            speed_mu = self.prior_speed_mu_FC(z[:, :, 4, :]).unsqueeze(dim=2)
            z_mu = torch.cat([poi_mu, bike_mu, taxi_mu, bus_mu, speed_mu], dim=2)


        if self.var_type == 'share':
            z_logvar = self.prior_logvar_FC(z)

        elif self.var_type == 'split':
            poi_logvar = self.prior_poi_logvar_FC(z[:, :, 0, :]).unsqueeze(dim=2)
            bike_logvar = self.prior_bike_logvar_FC(z[:, :, 1, :]).unsqueeze(dim=2)
            taxi_logvar = self.prior_taxi_logvar_FC(z[:, :, 2, :]).unsqueeze(dim=2)
            bus_logvar = self.prior_bus_logvar_FC(z[:, :, 3, :]).unsqueeze(dim=2)
            speed_logvar = self.prior_speed_logvar_FC(z[:, :, 4, :]).unsqueeze(dim=2)

            z_logvar = torch.cat([poi_logvar, bike_logvar, taxi_logvar, bus_logvar, speed_logvar], dim=2)

        else:
            z_logvar = torch.zeros_like(z_mu)

        hidden = torch.cat([poi_hidden.unsqueeze(dim=2),
                            bike_hideen.unsqueeze(dim=2),
                            taxi_hidden.unsqueeze(dim=2),
                            bus_hidden.unsqueeze(dim=2),
                            speed_hidden.unsqueeze(dim=2)], dim=2)

        return eps, z_logvar, z_mu, z_logvar, hidden
