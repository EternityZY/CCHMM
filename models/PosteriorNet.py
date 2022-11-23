# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.GCN import mixpropGCN
from models.GraphGRU import GraphGRU


class PosteriorNet(nn.Module):
    def __init__(self,
                 tpos_dim,
                 POI_dim,
                 weather_dim,
                 road_dim=None,
                 input_channels=[2, 2, 2, 1],
                 output_channels=[2, 2, 2, 1],
                 hidden_num=4,
                 hidden_dim=64,
                 norm_adj=None,
                 gcn_depth=2,
                 SCM_model=None,
                 Posterior_type='GCN',
                 dropout_prob=0.3,
                 var_type=False,
                 use_reparameterize=True,
                 mu_type=False,
                 activation=nn.ReLU(inplace=True),
                 ):
        super(PosteriorNet, self).__init__()

        self.bike_flow_channel = input_channels[0]
        self.taxi_flow_channel = input_channels[1]
        self.bus_flow_channel = input_channels[2]
        self.speed_channel = input_channels[3]

        self.tpos_dim = tpos_dim
        self.POI_dim = POI_dim
        self.weather_dim = weather_dim

        self.SCM_model = SCM_model

        self.var_type = var_type
        self.use_reparameterize = use_reparameterize
        self.mu_type = mu_type
        self.activation = activation
        self.Posterior_type = Posterior_type
        self.posterior_bike_encoder = nn.Sequential(nn.Linear(self.bike_flow_channel, hidden_dim),
                                                    )
        self.posterior_taxi_encoder = nn.Sequential(nn.Linear(self.taxi_flow_channel, hidden_dim),
                                                    )
        self.posterior_bus_encoder = nn.Sequential(nn.Linear(self.bus_flow_channel, hidden_dim)
                                                   )
        self.posterior_speed_encoder = nn.Sequential(nn.Linear(self.speed_channel, hidden_dim)
                                                     )

        if Posterior_type == 'FC':
            # posterior inference for eps_t
            self.poi_feature_fusion = nn.Linear(hidden_dim * 4 + self.tpos_dim + self.POI_dim, hidden_dim)
            self.bike_feature_fusion = nn.Linear(hidden_dim * 1+ self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.taxi_feature_fusion = nn.Linear(hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.bus_feature_fusion = nn.Linear(hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.speed_feature_fusion = nn.Linear(hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)

            self.posterior_eps_poi = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                self.activation
                )
            self.posterior_eps_bike = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                self.activation
                )
            self.posterior_eps_taxi = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                self.activation
                )
            self.posterior_eps_bus = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                self.activation
                )
            self.posterior_eps_speed = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                self.activation
            )

        elif Posterior_type == 'GCN':

            self.poi_feature_fusion = nn.Linear(hidden_dim * 4 + self.tpos_dim + self.POI_dim, hidden_dim)
            self.bike_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.taxi_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.bus_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.speed_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)

            self.posterior_eps_poi = mixpropGCN(hidden_dim*2,
                                                hidden_dim, gdep=gcn_depth, norm_adj=norm_adj,
                                                dropout_prob=dropout_prob)
            self.posterior_eps_bike = mixpropGCN(hidden_dim*2,
                                                 hidden_dim, gdep=gcn_depth, norm_adj=norm_adj,
                                                 dropout_prob=dropout_prob)
            self.posterior_eps_taxi = mixpropGCN(hidden_dim*2,
                                                 hidden_dim, gdep=gcn_depth, norm_adj=norm_adj,
                                                 dropout_prob=dropout_prob)
            self.posterior_eps_bus = mixpropGCN(hidden_dim*2,
                                                hidden_dim, gdep=gcn_depth, norm_adj=norm_adj,
                                                dropout_prob=dropout_prob)
            self.posterior_eps_speed = mixpropGCN(hidden_dim*2,
                                                hidden_dim, gdep=gcn_depth, norm_adj=norm_adj,
                                                dropout_prob=dropout_prob)
        elif Posterior_type == 'GraphGRU':

            self.poi_feature_fusion = nn.Linear(hidden_dim * 4 + self.tpos_dim + self.POI_dim, hidden_dim)
            self.bike_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.taxi_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.bus_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)
            self.speed_feature_fusion = nn.Linear(
                hidden_dim * 1 + self.tpos_dim + self.weather_dim + self.POI_dim, hidden_dim)

            self.posterior_eps_poi_GRU = GraphGRU(hidden_dim,
                                                  hidden_dim,
                                                  norm_adj,
                                                  gcn_depth=gcn_depth,
                                                  dropout_type='None',
                                                  dropout_prob=dropout_prob,
                                                  alpha=0.3)

            self.posterior_eps_bike_GRU = GraphGRU(hidden_dim,
                                                   hidden_dim,
                                                   norm_adj,
                                                   gcn_depth=gcn_depth,
                                                   dropout_type='None',
                                                   dropout_prob=dropout_prob,
                                                   alpha=0.3)

            self.posterior_eps_taxi_GRU = GraphGRU(hidden_dim,
                                                   hidden_dim,
                                                   norm_adj,
                                                   gcn_depth=gcn_depth,
                                                   dropout_type='None',
                                                   dropout_prob=dropout_prob,
                                                   alpha=0.3)

            self.posterior_eps_bus_GRU = GraphGRU(hidden_dim,
                                                  hidden_dim,
                                                  norm_adj,
                                                  gcn_depth=gcn_depth,
                                                  dropout_type='None',
                                                  dropout_prob=dropout_prob,
                                                  alpha=0.3)

            self.posterior_eps_speed_GRU = GraphGRU(hidden_dim,
                                                  hidden_dim,
                                                  norm_adj,
                                                  gcn_depth=gcn_depth,
                                                  dropout_type='None',
                                                  dropout_prob=dropout_prob,
                                                  alpha=0.3)

        # posterior inference for mu, logvar
        if self.mu_type == 'share':
            self.posterior_mu_FC = nn.Linear(hidden_dim, hidden_dim)

        elif self.mu_type == 'split':
            self.posterior_poi_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_bike_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_taxi_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_bus_mu_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_speed_mu_FC = nn.Linear(hidden_dim, hidden_dim)

        if self.var_type == 'share':
            self.posterior_logvar_FC = nn.Linear(hidden_dim, hidden_dim)

        elif self.var_type == 'split':
            self.posterior_poi_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_bike_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_taxi_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_bus_logvar_FC = nn.Linear(hidden_dim, hidden_dim)
            self.posterior_speed_logvar_FC = nn.Linear(hidden_dim, hidden_dim)

        # generate flow and speed via z_t and context
        self.posterior_bike_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4),
                                                    self.activation,
                                                    nn.Linear(hidden_dim // 4, output_channels[0]),
                                                    )
        self.posterior_taxi_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4),
                                                    self.activation,
                                                    nn.Linear(hidden_dim // 4, output_channels[1]),
                                                    )
        self.posterior_bus_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4),
                                                   self.activation,
                                                   nn.Linear(hidden_dim // 4, output_channels[2]),
                                                   )
        self.posterior_speed_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4),
                                                     self.activation,
                                                     nn.Linear(hidden_dim//4, output_channels[3]),
                                                     )


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

    def reparameterize(self, mu, logvar, test):

        if not test:
            std = torch.exp(0.5 * logvar)

            eps = torch.randn_like(mu)

            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, tpos_cur_features, POI_features, weather_cur_features,
                bike_flow_cur, taxi_flow_cur, bus_flow_cur, speed_cur, z_po_last,
                test=False):

        bike_features = self.posterior_bike_encoder(bike_flow_cur)
        taxi_features = self.posterior_taxi_encoder(taxi_flow_cur)
        bus_features = self.posterior_bus_encoder(bus_flow_cur)
        speed_features = self.posterior_speed_encoder(speed_cur)

        eps_poi_input = torch.cat([speed_features, bike_features, taxi_features, bus_features,
                                   POI_features, tpos_cur_features], dim=-1)
        eps_bike_input = torch.cat([bike_features, POI_features,
                                    weather_cur_features, tpos_cur_features], dim=-1)
        eps_taxi_input = torch.cat([taxi_features, POI_features,
                                    weather_cur_features, tpos_cur_features], dim=-1)
        eps_bus_input = torch.cat([bus_features, POI_features,
                                   weather_cur_features, tpos_cur_features], dim=-1)
        eps_speed_input = torch.cat([speed_features, POI_features,
                                     weather_cur_features, tpos_cur_features], dim=-1)

        eps_poi_fusion = self.poi_feature_fusion(eps_poi_input)
        eps_bike_fusion = self.bike_feature_fusion(eps_bike_input)
        eps_taxi_fusion = self.taxi_feature_fusion(eps_taxi_input)
        eps_bus_fusion = self.bus_feature_fusion(eps_bus_input)
        eps_speed_fusion = self.speed_feature_fusion(eps_speed_input)

        hidden = None
        # calculate eps_cur
        if self.Posterior_type == 'GraphGRU':
            eps_poi_cur, poi_hidden = self.posterior_eps_poi_GRU(eps_poi_fusion, z_po_last[:, :, 0, :])
            eps_bike_cur, bike_hideen = self.posterior_eps_bike_GRU(eps_bike_fusion, z_po_last[:, :, 1, :])
            eps_taxi_cur, taxi_hidden = self.posterior_eps_taxi_GRU(eps_taxi_fusion, z_po_last[:, :, 2, :])
            eps_bus_cur, bus_hidden = self.posterior_eps_bus_GRU(eps_bus_fusion, z_po_last[:, :, 3, :])
            eps_speed_cur, speed_hidden = self.posterior_eps_speed_GRU(eps_speed_fusion, z_po_last[:, :, 4, :])

            hidden = torch.cat([poi_hidden.unsqueeze(dim=2),
                                bike_hideen.unsqueeze(dim=2),
                                taxi_hidden.unsqueeze(dim=2),
                                bus_hidden.unsqueeze(dim=2),
                                speed_hidden.unsqueeze(dim=2)], dim=2)

        else:
            eps_poi_cur = self.posterior_eps_poi(torch.cat([eps_poi_fusion,
                                                            z_po_last[:, :, 0, :]], dim=-1))
            eps_bike_cur = self.posterior_eps_bike(torch.cat([eps_bike_fusion,
                                                              z_po_last[:, :, 1, :]], dim=-1))
            eps_taxi_cur = self.posterior_eps_taxi(torch.cat([eps_taxi_fusion,
                                                              z_po_last[:, :, 2, :]], dim=-1))
            eps_bus_cur = self.posterior_eps_bus(torch.cat([eps_bus_fusion,
                                                            z_po_last[:, :, 3, :]], dim=-1))
            eps_speed_cur = self.posterior_eps_speed(torch.cat([eps_speed_fusion, z_po_last[:, :, 4, :]], dim=-1))


        eps = torch.cat([eps_poi_cur.unsqueeze(dim=2),
                         eps_bike_cur.unsqueeze(dim=2),
                         eps_taxi_cur.unsqueeze(dim=2),
                         eps_bus_cur.unsqueeze(dim=2),
                         eps_speed_cur.unsqueeze(dim=2), ], dim=2)

        z = self.SCM_model(eps)

        # calculate mu
        if self.mu_type == 'share':
            z_mu = self.posterior_mu_FC(z)

        elif self.mu_type == 'split':
            poi_mu = self.posterior_poi_mu_FC(z[:, :, 0, :]).unsqueeze(dim=2)
            bike_mu = self.posterior_bike_mu_FC(z[:, :, 1, :]).unsqueeze(dim=2)
            taxi_mu = self.posterior_taxi_mu_FC(z[:, :, 2, :]).unsqueeze(dim=2)
            bus_mu = self.posterior_bus_mu_FC(z[:, :, 3, :]).unsqueeze(dim=2)
            speed_mu = self.posterior_speed_mu_FC(z[:, :, 4, :]).unsqueeze(dim=2)

            z_mu = torch.cat([poi_mu, bike_mu, taxi_mu, bus_mu, speed_mu], dim=2)

        if self.var_type == 'share':
            z_logvar = self.posterior_logvar_FC(z)

        elif self.var_type == 'split':
            poi_logvar = self.posterior_poi_logvar_FC(z[:, :, 0, :]).unsqueeze(dim=2)
            bike_logvar = self.posterior_bike_logvar_FC(z[:, :, 1, :]).unsqueeze(dim=2)
            taxi_logvar = self.posterior_taxi_logvar_FC(z[:, :, 2, :]).unsqueeze(dim=2)
            bus_logvar = self.posterior_bus_logvar_FC(z[:, :, 3, :]).unsqueeze(dim=2)
            speed_logvar = self.posterior_speed_logvar_FC(z[:, :, 4, :]).unsqueeze(dim=2)

            z_logvar = torch.cat([poi_logvar, bike_logvar, taxi_logvar, bus_logvar, speed_logvar], dim=2)

        else:
            z_logvar = torch.zeros_like(z_mu)


        if self.use_reparameterize:
            z_cur = self.reparameterize(z_mu, z_logvar, test)
        else:
            z_cur = z_mu

        bike_flow_rec = self.posterior_bike_decoder(z_cur[:, :, 1, :])
        taxi_flow_rec = self.posterior_taxi_decoder(z_cur[:, :, 2, :])
        bus_flow_rec = self.posterior_bus_decoder(z_cur[:, :, 3, :])
        speed_rec = self.posterior_speed_decoder(z_cur[:, :, 4, :])

        return eps, z_logvar, z_mu, z_logvar, z_cur, \
               bike_flow_rec, taxi_flow_rec, bus_flow_rec, speed_rec, \
               hidden
