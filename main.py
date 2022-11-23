# !/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
from datetime import datetime
import torch
import pandas as pd
import numpy as np
import time
import sys
import os

from ModelTest import baseline_test
from ModelTrain import baseline_train
from models.CausalHMM import CCausalHMM

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from config.config import get_logger
from preprocess.datasets import load_dataset
from tools.utils import sym_adj, asym_adj
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == '__main__':


    config_filename = 'config/config_1.yaml'
    with open(config_filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)

    base_path = cfg['base_path']

    dataset_name = cfg['dataset_name']

    dataset_path = os.path.join(base_path, dataset_name)


    log_path = os.path.join('Results', cfg['model_name'], 'exp{:d}'.format(cfg['expid']), 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)


    save_path = os.path.join('Results', cfg['model_name'], 'exp{:d}'.format(cfg['expid']), 'ckpt')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)



    log_dir = log_path
    log_level = 'INFO'
    log_name = 'info_' + datetime.now().strftime('%m-%d_%H:%M') + '.log'
    logger = get_logger(log_dir, __name__, log_name, level=log_level)


    confi_name = 'config{:d}_'.format(cfg['expid']) + datetime.now().strftime('%m-%d_%H:%M') + '.yaml'
    with open(os.path.join(log_dir, confi_name), 'w+') as _f:
        yaml.safe_dump(cfg, _f)

    logger.info(cfg)
    logger.info(dataset_path)
    logger.info(log_path)

    torch.set_num_threads(3)
    device = torch.device(cfg['device'])


    dataloader = load_dataset(dataset_path,
                              cfg['data']['train_batch_size'],
                              cfg['data']['val_batch_size'],
                              cfg['data']['test_batch_size'],
                              logger=logger,
                              cfg=cfg
                              )


    if cfg['model']['adj']=='adj':
        geo_graph = np.load(os.path.join(base_path, 'graph', 'geo_adj.npy')).astype(np.float32)

    elif cfg['model']['adj']=='affinity':
        geo_graph = np.load(os.path.join(base_path, 'graph', 'geo_affinity.npy')).astype(np.float32)

    adjs = [geo_graph]

    if cfg['model']['norm_graph'] == 'sym':
        static_norm_adjs = [torch.tensor(sym_adj(adj)).to(device) for adj in adjs]
    elif cfg['model']['norm_graph'] == 'asym':
        static_norm_adjs = [torch.tensor(asym_adj(adj)).to(device) for adj in adjs]
    else:
        static_norm_adjs = [torch.tensor(adj).to(device) for adj in adjs]

    POI = pd.read_hdf(os.path.join(base_path, 'poi.h5')).values
    road_num_list = None

    DAG = torch.tensor(cfg['model']['init_DAG']).to(torch.float)
    DAG = DAG.to(device)

    model_name = cfg['model_name']

    val_loss_list = []
    val_mae_list = []
    val_mape_list = []
    val_rmse_list = []

    loss_list = []
    mae_list = []
    mape_list = []
    rmse_list = []
    for runid in range(cfg['runs']):

        if model_name == 'CausalHMM':
            # 0:tpos   1:POI   2:weather   3:road
            CausalHMM = CCausalHMM(context_channels=cfg['model']['context_channels'],
                                   POI_vector=POI,
                                   context_dims=cfg['model']['context_dims'],
                                   hidden_num=cfg['model']['hidden_num'],
                                   hidden_dim=cfg['model']['hidden_dim'],
                                   input_channels=cfg['model']['input_channels'],
                                   output_channels=cfg['model']['output_channels'],
                                   gcn_depth=cfg['model']['gcn_depth'],
                                   norm_adj=static_norm_adjs[0],
                                   SCM_type=cfg['model']['SCM_type'],
                                   nonlinear_activation=cfg['model']['nonlinear_activation'],
                                   Prior_type=cfg['model']['Prior_type'],
                                   Posterior_type=cfg['model']['Posterior_type'],
                                   dropout_prob=cfg['model']['dropout_prob'],
                                   mu_type=cfg['model']['mu_type'],
                                   var_type=cfg['model']['var_type'],
                                   use_reparameterize=cfg['model']['use_reparameterize'],
                                   activation_type=cfg['model']['activation_type'],
                                   pred_z_init=cfg['model']['pred_z_init'],
                                   )

        # model.apply(weight_init)

        logger.info(model_name)

        if cfg['test_only']:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_test(runid,
                                                           CausalHMM,
                                                           dataloader,
                                                           device,
                                                           logger,
                                                           cfg)
        else:
            mvalid_total_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
            mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = baseline_train(runid,
                                                                          CausalHMM,
                                                                          model_name,
                                                                          dataloader,
                                                                          device,
                                                                          logger,
                                                                          cfg)
        val_loss_list.append(mvalid_total_loss)
        val_mae_list.append(mvalid_pred_mae)
        val_mape_list.append(mvalid_pred_mape)
        val_rmse_list.append(mvalid_pred_rmse)

        loss_list.append(mtest_total_loss)
        mae_list.append(mtest_pred_mae)
        mape_list.append(mtest_pred_mape)
        rmse_list.append(mtest_pred_rmse)

    loss_list = np.array(loss_list)
    mae_list = np.array(mae_list)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    aloss = np.mean(loss_list, 0)
    amae = np.mean(mae_list, 0)
    amape = np.mean(mape_list, 0)
    armse = np.mean(rmse_list, 0)

    sloss = np.std(loss_list, 0)
    smae = np.std(mae_list, 0)
    smape = np.std(mape_list, 0)
    srmse = np.std(rmse_list, 0)

    logger.info('valid\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(val_loss_list), np.mean(val_mae_list), np.mean(val_rmse_list), np.mean(val_mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(val_loss_list), np.std(val_mae_list), np.std(val_rmse_list), np.std(val_mape_list)))
    logger.info('\n\n')

    logger.info('Test\tLoss\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.mean(loss_list), np.mean(mae_list), np.mean(rmse_list), np.mean(mape_list)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    logger.info(log.format(np.std(loss_list), np.std(mae_list), np.std(rmse_list), np.mean(mape_list)))
    logger.info('\n\n')
