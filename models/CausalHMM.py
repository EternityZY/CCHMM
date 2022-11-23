# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from models.CausalLayer import SCM
from models.PosteriorNet import PosteriorNet
from models.PriorNet import PriorNet
from tools.utils import kl_normal_log


class Attention(nn.Module):
    def __init__(self, hidden_dim, hidden_num, bias=False):
        super().__init__()
        self.M = nn.Parameter(
            torch.nn.init.normal_(torch.zeros(hidden_dim, hidden_dim), mean=0, std=1))
        self.FC = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, po_z, pr_z):
        a = po_z.matmul(self.M).matmul(pr_z.permute(0, 1, 3, 2))
        A = torch.softmax(a, dim=-1)
        out = torch.matmul(A, pr_z)
        return out, A

class CCausalHMM(nn.Module):
    '''

    '''

    def __init__(self,
                 context_channels=[3, 5, 5],
                 POI_vector=None,
                 context_dims=[32, 32, 32],
                 hidden_num=4,
                 hidden_dim=64,
                 input_channels=[2, 2, 2, 1],
                 output_channels=[2, 2, 2, 1],
                 gcn_depth=2,
                 norm_adj=None,
                 SCM_type='nonlinear',
                 nonlinear_activation='relu',
                 Prior_type='GraphGRU',
                 Posterior_type='GCN',
                 dropout_prob=0.3,
                 mu_type='split',
                 var_type='none',
                 use_reparameterize=True,
                 activation_type='relu',
                 pred_z_init='origin',
                 ):
        super(CCausalHMM, self).__init__()

        self.tpos_channel = context_channels[0]
        self.POI_channel = context_channels[1]
        self.weather_channel = context_channels[2]

        self.tpos_dim = context_dims[0]
        self.POI_dim = context_dims[1]
        self.weather_dim = context_dims[2]

        self.bike_flow_channel = input_channels[0]
        self.taxi_flow_channel = input_channels[1]
        self.bus_flow_channel = input_channels[2]
        self.speed_channel = input_channels[3]

        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.region_num = norm_adj.shape[0]
        self.var_type = var_type
        self.norm_adj = norm_adj

        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'elu':
            self.activation = nn.ELU(inplace=True)

        self.pred_z_init = pred_z_init
        self.att = Attention(hidden_dim, hidden_num)

        # inference for z_t via eps_t
        if nonlinear_activation == 'relu':
            nonlinear = nn.ReLU(inplace=True)
        elif nonlinear_activation == 'sigmoid':
            nonlinear = nn.Sigmoid()
        elif nonlinear_activation == 'elu':
            nonlinear = nn.ELU(inplace=True)
        elif nonlinear_activation == 'tanh':
            nonlinear = nn.Tanh()

        self.SCM = SCM(hidden_dim, hidden_dim, hidden_num,scm_type=SCM_type,
                       nonlinear_activation=nonlinear)

        self.POI_vector = torch.tensor(POI_vector).to(torch.float32)

        self.prior_tpos_FC = nn.Linear(self.tpos_channel, self.tpos_dim)
        self.prior_POI_FC = nn.Linear(self.POI_channel, self.POI_dim)
        self.prior_weather_FC = nn.Linear(self.weather_channel, self.weather_dim)

        # posterior encoder for context
        self.posterior_tpos_FC = nn.Linear(self.tpos_channel, self.tpos_dim)
        self.posterior_POI_FC = nn.Linear(self.POI_channel, self.POI_dim)
        self.posterior_weather_FC = nn.Linear(self.weather_channel, self.weather_dim)

        self.PriorNet = PriorNet(
            tpos_dim=self.tpos_dim,
            POI_dim=self.POI_dim,
            weather_dim=self.weather_dim,
            hidden_num=hidden_num,
            hidden_dim=hidden_dim,
            gcn_depth=gcn_depth,
            norm_adj=norm_adj,
            SCM_model=self.SCM,
            Prior_type=Prior_type,
            dropout_prob=dropout_prob,
            var_type=var_type,
            mu_type=mu_type,
            activation=self.activation,
        )

        self.PosteriorNet = PosteriorNet(
            tpos_dim=self.tpos_dim,
            POI_dim=self.POI_dim,
            weather_dim=self.weather_dim,
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_num=hidden_num,
            hidden_dim=hidden_dim,
            gcn_depth=gcn_depth,
            norm_adj=norm_adj,
            SCM_model=self.SCM,
            Posterior_type=Posterior_type,
            dropout_prob=dropout_prob,
            var_type=var_type,
            use_reparameterize=use_reparameterize,
            mu_type=mu_type,
            activation=self.activation,
        )

    def _init_papameters(self, args):

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRUCell):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal_(m.weight_hh.data)
                    nn.init.kaiming_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight_hh.data)
                    nn.init.xavier_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)

    def Generator_cur(self, pr_z_mu_cur, pr_z_logvar_cur, test=True):

        if self.PosteriorNet.use_reparameterize:
            z_cur = self.PosteriorNet.reparameterize(pr_z_mu_cur, pr_z_logvar_cur, test)
        else:
            z_cur = pr_z_mu_cur

        bike_flow_next = self.PosteriorNet.posterior_bike_decoder(z_cur[:, :, 1, :])
        taxi_flow_next = self.PosteriorNet.posterior_taxi_decoder(z_cur[:, :, 2, :])
        bus_flow_next = self.PosteriorNet.posterior_bus_decoder(z_cur[:, :, 3, :])
        speed_next = self.PosteriorNet.posterior_speed_decoder(z_cur[:, :, 4, :])

        return bike_flow_next, taxi_flow_next, bus_flow_next, speed_next

    def Generator_next(self, tpos_next, weather_next, POI_vector,
                       po_z_mu_last=None, po_z_logvar_last=None, z_last=None, test=True):
        '''

        :param tpos_next:
        :param weather_next:
        :return:
        '''


        batch, node, _ = tpos_next.shape
        tpos_next_feature = self.prior_tpos_FC(tpos_next)
        weather_next_feature = self.prior_weather_FC(weather_next)
        POI_feature = self.prior_POI_FC(POI_vector)


        pr_eps_mu, pr_eps_logvar, pr_z_mu, pr_z_logvar, pr_hidden = self.PriorNet(tpos_next_feature,
                                                                POI_feature,
                                                                weather_next_feature,
                                                                z_last)

        if self.pred_z_init == 'attention':
            input_z_mu, mu_att_map = self.att.attention(po_z_mu_last, pr_z_mu)
        elif self.pred_z_init == 'origin':
            input_z_mu = pr_z_mu

        input_z_logvar = pr_z_logvar

        if self.PosteriorNet.use_reparameterize:
            z_cur = self.PosteriorNet.reparameterize(input_z_mu, input_z_logvar, test)
        else:
            z_cur = input_z_mu


        bike_flow_pred = self.PosteriorNet.posterior_bike_decoder(z_cur[:, :, 1, :])
        taxi_flow_pred = self.PosteriorNet.posterior_taxi_decoder(z_cur[:, :, 2, :])
        bus_flow_pred = self.PosteriorNet.posterior_bus_decoder(z_cur[:, :, 3, :])
        speed_pred = self.PosteriorNet.posterior_speed_decoder(z_cur[:, :, 4, :])

        return bike_flow_pred, taxi_flow_pred, bus_flow_pred, speed_pred

    def forward(self, tpos, weather, bike_flow, taxi_flow, bus_flow, speed, test=False):

        batch, node, time, _ = bike_flow.shape

        POI_vector = torch.repeat_interleave(self.POI_vector.unsqueeze(dim=0), repeats=batch, dim=0).to(tpos.device)

        prior_POI_features = self.prior_POI_FC(POI_vector)
        posterior_POI_features = self.posterior_POI_FC(POI_vector)


        pr_hidden_last = torch.zeros((batch, node, self.hidden_num, self.hidden_dim), device=bike_flow.device)
        po_hidden_last = torch.zeros((batch, node, self.hidden_num, self.hidden_dim), device=bike_flow.device)
        po_z_mu_last = torch.zeros((batch, node, self.hidden_num, self.hidden_dim), device=bike_flow.device)
        po_z_logvar_last = torch.zeros((batch, node, self.hidden_num, self.hidden_dim), device=bike_flow.device)


        rec_bike_list = []
        rec_taxi_list = []
        rec_bus_list = []
        rec_speed_list = []

        gen_bike_list = []
        gen_taxi_list = []
        gen_bus_list = []
        gen_speed_list = []


        po_z_mu_list = []
        po_z_logvar_list = []
        po_z_list = []

        eps_all_kl_loss = torch.zeros(1, device=tpos.device)
        z_all_kl_loss = torch.zeros(1, device=tpos.device)


        for t in range(time - 1):
            prior_tpos_features = self.prior_tpos_FC(tpos[:, :, t, :])
            prior_weather_features = self.prior_weather_FC(weather[:, :, t, :])

            posterior_tpos_features = self.posterior_tpos_FC(tpos[:, :, t, :])
            posterior_weather_features = self.posterior_weather_FC(weather[:, :, t, :])

            pr_eps_mu, pr_eps_logvar, pr_z_mu, pr_z_logvar, pr_hidden = self.PriorNet(prior_tpos_features,
                                                                    prior_POI_features,
                                                                    prior_weather_features,
                                                                    pr_hidden_last)

            po_eps_mu, po_eps_logvar, po_z_mu, po_z_logvar, po_z_cur, \
            bike_flow_rec, taxi_flow_rec, bus_flow_rec, speed_rec, po_hidden = self.PosteriorNet(
                posterior_tpos_features,
                posterior_POI_features,
                posterior_weather_features,
                bike_flow[:, :, t, :],
                taxi_flow[:, :, t, :],
                bus_flow[:, :, t, :],
                speed[:, :, t, :],
                po_hidden_last,
                test=test)

            if t >= 1:

                if self.pred_z_init == 'attention':
                    input_z_mu, mu_att_map = self.att.attention(po_z_mu_last, pr_z_mu)
                elif self.pred_z_init == 'origin':
                    input_z_mu = pr_z_mu
                input_z_logvar = pr_z_logvar

                gen_bike_cur, gen_taxi_cur, \
                gen_bus_cur, gen_speed_cur = self.Generator_cur(input_z_mu, input_z_logvar, test=test)


                gen_bike_list.append(gen_bike_cur.unsqueeze(dim=2))
                gen_taxi_list.append(gen_taxi_cur.unsqueeze(dim=2))
                gen_bus_list.append(gen_bus_cur.unsqueeze(dim=2))
                gen_speed_list.append(gen_speed_cur.unsqueeze(dim=2))

            pr_hidden_last = po_z_mu
            po_hidden_last = po_z_mu

            po_z_mu_last = po_z_mu
            po_z_logvar_last = po_z_logvar


            rec_bike_list.append(torch.unsqueeze(bike_flow_rec, dim=2))
            rec_taxi_list.append(torch.unsqueeze(taxi_flow_rec, dim=2))
            rec_bus_list.append(torch.unsqueeze(bus_flow_rec, dim=2))
            rec_speed_list.append(torch.unsqueeze(speed_rec, dim=2))

            po_z_mu_list.append(po_z_mu)
            po_z_logvar_list.append(po_z_logvar)
            po_z_list.append(po_z_cur)

            eps_all_kl_loss += kl_normal_log(po_eps_mu, po_eps_logvar,
                                             pr_eps_mu, pr_eps_logvar).sum()

            z_all_kl_loss += kl_normal_log(po_z_mu, po_z_logvar,
                                           pr_z_mu, pr_z_logvar).sum()



        rec_bike = torch.cat(rec_bike_list, dim=2)
        rec_taxi = torch.cat(rec_taxi_list, dim=2)
        rec_bus = torch.cat(rec_bus_list, dim=2)
        rec_speed = torch.cat(rec_speed_list, dim=2)

        eps_kl_loss = eps_all_kl_loss / (time - 1)
        z_kl_loss = z_all_kl_loss / (time - 1)


        gen_bike_next, gen_taxi_next, \
        gen_bus_next, gen_speed_next = self.Generator_next(tpos[:, :, -1, :],
                                                           weather[:, :, -1, :],
                                                           POI_vector,
                                                           po_z_mu_last,
                                                           po_z_logvar_last,
                                                           po_hidden_last,
                                                           test=test
                                                           )

        gen_bike_list.append(gen_bike_next.unsqueeze(dim=2))
        gen_taxi_list.append(gen_taxi_next.unsqueeze(dim=2))
        gen_bus_list.append(gen_bus_next.unsqueeze(dim=2))
        gen_speed_list.append(gen_speed_next.unsqueeze(dim=2))

        gen_bike = torch.cat(gen_bike_list, dim=2)
        gen_taxi = torch.cat(gen_taxi_list, dim=2)
        gen_bus = torch.cat(gen_bus_list, dim=2)
        gen_speed = torch.cat(gen_speed_list, dim=2)

        return po_z_mu_list, po_z_logvar_list, po_z_list, \
               eps_kl_loss, z_kl_loss, \
               [rec_bike, rec_taxi, rec_bus, rec_speed], \
               [gen_bike, gen_taxi, gen_bus, gen_speed]
