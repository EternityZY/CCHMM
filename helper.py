# !/usr/bin/env python
# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim

from tools.metrics import masked_mae_torch, masked_mape_torch, masked_rmse_torch, metric, metric_all
from tools.utils import StepLR2


class Trainer():
    def __init__(self,
                 CausalHMM,
                 base_lr,
                 weight_decay,
                 milestones,
                 lr_decay_ratio,
                 min_learning_rate,
                 max_grad_norm,
                 num_for_target,
                 num_for_predict,
                 scaler,
                 device,
                 loss_weight,
                 DAG_loss_weight=5,
                 ):
        self.scaler = scaler
        self.CausalHMM = CausalHMM

        self.device = device
        self.max_grad_norm = max_grad_norm
        self.loss_weight = loss_weight
        self.DAG_loss_weight = DAG_loss_weight
        self.CausalHMM.to(device)
        self.CausalHMM_optimizer = optim.Adam(self.CausalHMM.parameters(), lr=base_lr, weight_decay=weight_decay)

        self.CausalHMM_scheduler = StepLR2(optimizer=self.CausalHMM_optimizer,
                                           milestones=milestones,
                                           gamma=lr_decay_ratio,
                                           min_lr=min_learning_rate)

        self.SmoothL1loss = nn.SmoothL1Loss(reduction='mean')
        self.scaler = scaler
        self.num_for_target = num_for_target
        self.num_for_predict = num_for_predict

    def train(self, tpos, weather, bike_flow, taxi_flow, bus_flow, speed):
        """
        :param input:  [batch node time hdim]
               output: [batch node time hdim]
        :param real_val:
        :return:
        """

        self.CausalHMM.train()
        self.CausalHMM_optimizer.zero_grad()

        po_z_mu_list, po_z_var_list, po_z_list, \
        eps_kl_loss, z_kl_loss, \
        rec_all, gen_all = self.CausalHMM(tpos, weather,
                                          self.scaler[0].transform(bike_flow),
                                          self.scaler[1].transform(taxi_flow),
                                          self.scaler[2].transform(bus_flow),
                                          self.scaler[3].transform(speed),
                                          test=False)

        rec_bike = self.scaler[0].inverse_transform(rec_all[0])
        rec_taxi = self.scaler[1].inverse_transform(rec_all[1])
        rec_bus = self.scaler[2].inverse_transform(rec_all[2])
        rec_speed = self.scaler[3].inverse_transform(rec_all[3])

        rec_loss = 0
        rec_loss += self.SmoothL1loss(rec_bike, bike_flow[:, :, :self.num_for_predict, :])
        rec_loss += self.SmoothL1loss(rec_taxi, taxi_flow[:, :, :self.num_for_predict, :])
        rec_loss += self.SmoothL1loss(rec_bus, bus_flow[:, :, :self.num_for_predict, :])
        rec_loss += self.SmoothL1loss(rec_speed, speed[:, :, :self.num_for_predict, :])


        rec_mae, rec_rmse, rec_mape = metric_all([rec_bike, rec_taxi, rec_bus, rec_speed],
                                                  [bike_flow[:, :, :self.num_for_predict, :],
                                                   taxi_flow[:, :, :self.num_for_predict, :],
                                                   bus_flow[:, :, :self.num_for_predict, :],
                                                   speed[:, :, :self.num_for_predict, :]])
        # rec_loss /= 4.

        pred_loss = 0
        gen_mae = {}
        gen_rmse = {}
        gen_mape = {}
        gen_bike = self.scaler[0].inverse_transform(gen_all[0])
        gen_taxi = self.scaler[1].inverse_transform(gen_all[1])
        gen_bus = self.scaler[2].inverse_transform(gen_all[2])
        gen_speed = self.scaler[3].inverse_transform(gen_all[3])


        pred_loss += self.SmoothL1loss(gen_bike, bike_flow[:, :, 1:self.num_for_predict + 1, :])
        pred_loss += self.SmoothL1loss(gen_taxi, taxi_flow[:, :, 1:self.num_for_predict + 1, :])
        pred_loss += self.SmoothL1loss(gen_bus, bus_flow[:, :, 1:self.num_for_predict + 1, :])
        pred_loss += self.SmoothL1loss(gen_speed, speed[:, :, 1:self.num_for_predict + 1, :])

        gen_mae, gen_rmse, gen_mape = metric_all([gen_bike, gen_taxi, gen_bus, gen_speed],
                                                 [bike_flow[:, :, 1:self.num_for_predict + 1, :],
                                                  taxi_flow[:, :, 1:self.num_for_predict + 1, :],
                                                  bus_flow[:, :, 1:self.num_for_predict + 1, :],
                                                  speed[:, :, 1:self.num_for_predict + 1, :]])

        HMM_loss = eps_kl_loss + z_kl_loss + rec_loss

        HMM_loss = HMM_loss + self.DAG_loss_weight * self.CausalHMM.SCM.cal_loss()

        total_loss = HMM_loss + self.loss_weight * pred_loss

        total_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.CausalHMM.parameters(), self.max_grad_norm)

        self.CausalHMM_optimizer.step()

        return total_loss.item(), eps_kl_loss.item(), z_kl_loss.item(), rec_loss.item(), pred_loss.item(), \
               rec_mae, rec_rmse, rec_mape, \
               gen_mae, gen_rmse, gen_mape

    def eval(self, tpos, weather, bike_flow, taxi_flow, bus_flow, speed):
        '''
        :param tpos:
        :param weather:
        :param bike_flow:
        :param taxi_flow:
        :param bus_flow:
        :param speed:
        :return:
        '''
        self.CausalHMM.eval()
        with torch.no_grad():
            po_z_mu_list, po_z_var_list, po_z_list, \
            eps_kl_loss, z_kl_loss, \
            rec_all, gen_all = self.CausalHMM(tpos, weather,
                                           self.scaler[0].transform(bike_flow),
                                           self.scaler[1].transform(taxi_flow),
                                           self.scaler[2].transform(bus_flow),
                                           self.scaler[3].transform(speed),
                                           test=True)

            rec_bike = self.scaler[0].inverse_transform(rec_all[0])
            rec_taxi = self.scaler[1].inverse_transform(rec_all[1])
            rec_bus = self.scaler[2].inverse_transform(rec_all[2])
            rec_speed = self.scaler[3].inverse_transform(rec_all[3])


            rec_loss = 0
            rec_loss += self.SmoothL1loss(rec_bike, bike_flow[:, :, :self.num_for_predict, :])
            rec_loss += self.SmoothL1loss(rec_taxi, taxi_flow[:, :, :self.num_for_predict, :])
            rec_loss += self.SmoothL1loss(rec_bus, bus_flow[:, :, :self.num_for_predict, :])
            rec_loss += self.SmoothL1loss(rec_speed, speed[:, :, :self.num_for_predict, :])

            rec_mae, rec_rmse, rec_mape = metric_all([rec_bike, rec_taxi, rec_bus, rec_speed],
                                                     [bike_flow[:, :, :self.num_for_predict, :],
                                                      taxi_flow[:, :, :self.num_for_predict, :],
                                                      bus_flow[:, :, :self.num_for_predict, :],
                                                      speed[:, :, :self.num_for_predict, :]])


            pred_loss = 0
            gen_mae = {}
            gen_rmse = {}
            gen_mape = {}
            pred = torch.zeros_like(speed[:, :, 1:self.num_for_predict+1, :])
            gen_bike = self.scaler[0].inverse_transform(gen_all[0])
            gen_taxi = self.scaler[1].inverse_transform(gen_all[1])
            gen_bus = self.scaler[2].inverse_transform(gen_all[2])
            gen_speed = self.scaler[3].inverse_transform(gen_all[3])

            pred = torch.cat([gen_bike, gen_taxi, gen_bus, gen_speed], dim=-1)

            pred_loss += self.SmoothL1loss(gen_bike, bike_flow[:, :, 1:self.num_for_predict + 1, :])
            pred_loss += self.SmoothL1loss(gen_taxi, taxi_flow[:, :, 1:self.num_for_predict + 1, :])
            pred_loss += self.SmoothL1loss(gen_bus, bus_flow[:, :, 1:self.num_for_predict + 1, :])
            pred_loss += self.SmoothL1loss(gen_speed, speed[:, :, 1:self.num_for_predict + 1, :])

            gen_mae, gen_rmse, gen_mape = metric_all([gen_bike, gen_taxi, gen_bus, gen_speed],
                                                     [bike_flow[:, :, 1:self.num_for_predict + 1, :],
                                                      taxi_flow[:, :, 1:self.num_for_predict + 1, :],
                                                      bus_flow[:, :, 1:self.num_for_predict + 1, :],
                                                      speed[:, :, 1:self.num_for_predict + 1, :]])


            HMM_loss = eps_kl_loss + z_kl_loss + rec_loss

            HMM_loss = HMM_loss + self.DAG_loss_weight * self.CausalHMM.SCM.cal_loss()

            total_loss = HMM_loss + self.loss_weight * pred_loss

        return total_loss.item(), eps_kl_loss.item(), z_kl_loss.item(), rec_loss.item(), pred_loss.item(), \
               rec_mae, rec_rmse, rec_mape, \
               gen_mae, gen_rmse, gen_mape, pred
