# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm
from helper import Trainer
import os
import torch.nn.functional as F
from tools.metrics import record


def model_val(runid, engine, dataloader, device, logger, cfg, epoch):
    logger.info('Start validation phase.....')

    val_dataloder = dataloader['val']

    valid_total_loss = []
    valid_eps_kl_loss = []
    valid_z_kl_loss = []
    valid_rec_loss = []
    valid_pred_loss = []

    valid_rec_mape = {}
    valid_rec_rmse = {}
    valid_rec_mae = {}

    valid_rec_mae['bike'] = []
    valid_rec_mae['taxi'] = []
    valid_rec_mae['bus'] = []
    valid_rec_mae['speed'] = []

    valid_rec_rmse['bike'] = []
    valid_rec_rmse['taxi'] = []
    valid_rec_rmse['bus'] = []
    valid_rec_rmse['speed'] = []

    valid_rec_mape['bike'] = []
    valid_rec_mape['taxi'] = []
    valid_rec_mape['bus'] = []
    valid_rec_mape['speed'] = []

    valid_pred_mape = {}
    valid_pred_rmse = {}
    valid_pred_mae = {}

    valid_pred_mae['bike'] = []
    valid_pred_mae['taxi'] = []
    valid_pred_mae['bus'] = []
    valid_pred_mae['speed'] = []

    valid_pred_rmse['bike'] = []
    valid_pred_rmse['taxi'] = []
    valid_pred_rmse['bus'] = []
    valid_pred_rmse['speed'] = []

    valid_pred_mape['bike'] = []
    valid_pred_mape['taxi'] = []
    valid_pred_mape['bus'] = []
    valid_pred_mape['speed'] = []

    val_tqdm_loader = tqdm(enumerate(val_dataloder))
    for iter, (bike, bus, taxi, speed, pos) in val_tqdm_loader:
        tpos = pos[..., :56]
        weather = pos[..., 56:]
        bike_flow = bike
        taxi_flow = taxi
        bus_flow = bus

        tpos = tpos.to(device)
        weather = weather.to(device)
        bike_flow = bike_flow.to(device)
        taxi_flow = taxi_flow.to(device)
        bus_flow = bus_flow.to(device)
        speed = speed.to(device)

        total_loss, eps_kl_loss, z_kl_loss, rec_loss, pred_loss, \
        rec_mae, rec_rmse, rec_mape, \
        gen_mae, gen_rmse, gen_mape, speed_pred = engine.eval(tpos, weather, bike_flow, taxi_flow, bus_flow, speed)

        valid_total_loss.append(total_loss)
        valid_eps_kl_loss.append(eps_kl_loss)
        valid_z_kl_loss.append(z_kl_loss)
        valid_rec_loss.append(rec_loss)
        valid_pred_loss.append(pred_loss)

        record(valid_rec_mae, valid_rec_rmse, valid_rec_mape, rec_mae, rec_rmse, rec_mape)
        record(valid_pred_mae, valid_pred_rmse, valid_pred_mape, gen_mae, gen_rmse, gen_mape, only_last=True)

    mvalid_total_loss = np.mean(valid_total_loss)
    mvalid_eps_kl_loss = np.mean(valid_eps_kl_loss)
    mvalid_z_kl_loss = np.mean(valid_z_kl_loss)
    mvalid_rec_loss = np.mean(valid_rec_loss)
    mvalid_pred_loss = np.mean(valid_pred_loss)

    mvalid_rec_bike_mae = np.mean(valid_rec_mae['bike'])
    mvalid_rec_bike_mape = np.mean(valid_rec_mape['bike'])
    mvalid_rec_bike_rmse = np.mean(valid_rec_rmse['bike'])

    mvalid_rec_taxi_mae = np.mean(valid_rec_mae['taxi'])
    mvalid_rec_taxi_mape = np.mean(valid_rec_mape['taxi'])
    mvalid_rec_taxi_rmse = np.mean(valid_rec_rmse['taxi'])

    mvalid_rec_bus_mae = np.mean(valid_rec_mae['bus'])
    mvalid_rec_bus_mape = np.mean(valid_rec_mape['bus'])
    mvalid_rec_bus_rmse = np.mean(valid_rec_rmse['bus'])

    mvalid_rec_speed_mae = np.mean(valid_rec_mae['speed'])
    mvalid_rec_speed_mape = np.mean(valid_rec_mape['speed'])
    mvalid_rec_speed_rmse = np.mean(valid_rec_rmse['speed'])


    mvalid_pred_bike_mae = np.mean(valid_pred_mae['bike'])
    mvalid_pred_bike_mape = np.mean(valid_pred_mape['bike'])
    mvalid_pred_bike_rmse = np.mean(valid_pred_rmse['bike'])

    mvalid_pred_taxi_mae = np.mean(valid_pred_mae['taxi'])
    mvalid_pred_taxi_mape = np.mean(valid_pred_mape['taxi'])
    mvalid_pred_taxi_rmse = np.mean(valid_pred_rmse['taxi'])

    mvalid_pred_bus_mae = np.mean(valid_pred_mae['bus'])
    mvalid_pred_bus_mape = np.mean(valid_pred_mape['bus'])
    mvalid_pred_bus_rmse = np.mean(valid_pred_rmse['bus'])

    mvalid_pred_speed_mae = np.mean(valid_pred_mae['speed'])
    mvalid_pred_speed_mape = np.mean(valid_pred_mape['speed'])
    mvalid_pred_speed_rmse = np.mean(valid_pred_rmse['speed'])

    log = 'Epoch: {:03d}, Valid Total Loss: {:.4f}\n' \
          'Valid Eps KL: {:.4f}\t\t\tValid Z KL: {:.4f} \n' \
          'Valid Rec Loss: {:.4f}\t\t\tValid Pred Loss: {:.4f} \n' \
          'Valid Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Valid Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Valid Rec  Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Valid Rec  Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n\n' \
          'Valid Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Valid Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Valid Pred Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Valid Pred Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(epoch, mvalid_total_loss, mvalid_eps_kl_loss, mvalid_z_kl_loss,
                           mvalid_rec_loss, mvalid_pred_loss,
                           mvalid_rec_bike_mae, mvalid_rec_bike_rmse, mvalid_rec_bike_mape,
                           mvalid_rec_taxi_mae, mvalid_rec_taxi_rmse, mvalid_rec_taxi_mape,
                           mvalid_rec_bus_mae,  mvalid_rec_bus_rmse , mvalid_rec_bus_mape,
                           mvalid_rec_speed_mae, mvalid_rec_speed_rmse, mvalid_rec_speed_mape,

                           mvalid_pred_bike_mae, mvalid_pred_bike_rmse, mvalid_pred_bike_mape,
                           mvalid_pred_taxi_mae, mvalid_pred_taxi_rmse, mvalid_pred_taxi_mape,
                           mvalid_pred_bus_mae, mvalid_pred_bus_rmse, mvalid_pred_bus_mape,
                           mvalid_pred_speed_mae, mvalid_pred_speed_rmse, mvalid_pred_speed_mape,
                           ))

    return mvalid_pred_loss, mvalid_pred_speed_mae, mvalid_pred_speed_rmse, mvalid_pred_speed_mape,


def model_test(runid, engine, dataloader, device, logger, cfg, mode='Test'):
    logger.info('Start testing phase.....')

    test_dataloder = dataloader['test']

    test_total_loss = []
    test_eps_kl_loss = []
    test_z_kl_loss = []
    test_rec_loss = []
    test_pred_loss = []

    test_rec_mape = {}
    test_rec_rmse = {}
    test_rec_mae = {}

    test_rec_mae['bike'] = []
    test_rec_mae['taxi'] = []
    test_rec_mae['bus'] = []
    test_rec_mae['speed'] = []

    test_rec_rmse['bike'] = []
    test_rec_rmse['taxi'] = []
    test_rec_rmse['bus'] = []
    test_rec_rmse['speed'] = []

    test_rec_mape['bike'] = []
    test_rec_mape['taxi'] = []
    test_rec_mape['bus'] = []
    test_rec_mape['speed'] = []

    test_pred_mape = {}
    test_pred_rmse = {}
    test_pred_mae = {}

    test_pred_mae['bike'] = []
    test_pred_mae['taxi'] = []
    test_pred_mae['bus'] = []
    test_pred_mae['speed'] = []

    test_pred_rmse['bike'] = []
    test_pred_rmse['taxi'] = []
    test_pred_rmse['bus'] = []
    test_pred_rmse['speed'] = []

    test_pred_mape['bike'] = []
    test_pred_mape['taxi'] = []
    test_pred_mape['bus'] = []
    test_pred_mape['speed'] = []

    test_outputs_list = []

    test_tqdm_loader = tqdm(enumerate(test_dataloder))
    for iter, ((bike, bus, taxi, speed, pos)) in test_tqdm_loader:
        tpos = pos[..., :56]
        weather = pos[..., 56:]
        bike_flow = bike
        taxi_flow = taxi
        bus_flow = bus

        tpos = tpos.to(device)
        weather = weather.to(device)
        bike_flow = bike_flow.to(device)
        taxi_flow = taxi_flow.to(device)
        bus_flow = bus_flow.to(device)
        speed = speed.to(device)

        total_loss, eps_kl_loss, z_kl_loss, rec_loss, pred_loss, \
        rec_mae, rec_rmse, rec_mape, \
        gen_mae, gen_rmse, gen_mape, speed_pred = engine.eval(tpos, weather, bike_flow, taxi_flow, bus_flow, speed)

        test_total_loss.append(total_loss)
        test_eps_kl_loss.append(eps_kl_loss)
        test_z_kl_loss.append(z_kl_loss)
        test_rec_loss.append(rec_loss)
        test_pred_loss.append(pred_loss)

        test_outputs_list.append(speed_pred)

        record(test_rec_mae, test_rec_rmse, test_rec_mape, rec_mae, rec_rmse, rec_mape)
        record(test_pred_mae, test_pred_rmse, test_pred_mape, gen_mae, gen_rmse, gen_mape, only_last=True)

    mtest_total_loss = np.mean(test_total_loss)
    mtest_eps_kl_loss = np.mean(test_eps_kl_loss)
    mtest_z_kl_loss = np.mean(test_z_kl_loss)
    mtest_rec_loss = np.mean(test_rec_loss)
    mtest_pred_loss = np.mean(test_pred_loss)

    mtest_rec_bike_mae = np.mean(test_rec_mae['bike'])
    mtest_rec_bike_mape = np.mean(test_rec_mape['bike'])
    mtest_rec_bike_rmse = np.mean(test_rec_rmse['bike'])

    mtest_rec_taxi_mae = np.mean(test_rec_mae['taxi'])
    mtest_rec_taxi_mape = np.mean(test_rec_mape['taxi'])
    mtest_rec_taxi_rmse = np.mean(test_rec_rmse['taxi'])

    mtest_rec_bus_mae = np.mean(test_rec_mae['bus'])
    mtest_rec_bus_mape = np.mean(test_rec_mape['bus'])
    mtest_rec_bus_rmse = np.mean(test_rec_rmse['bus'])

    mtest_rec_speed_mae = np.mean(test_rec_mae['speed'])
    mtest_rec_speed_mape = np.mean(test_rec_mape['speed'])
    mtest_rec_speed_rmse = np.mean(test_rec_rmse['speed'])


    mtest_pred_bike_mae = np.mean(test_pred_mae['bike'])
    mtest_pred_bike_mape = np.mean(test_pred_mape['bike'])
    mtest_pred_bike_rmse = np.mean(test_pred_rmse['bike'])

    mtest_pred_taxi_mae = np.mean(test_pred_mae['taxi'])
    mtest_pred_taxi_mape = np.mean(test_pred_mape['taxi'])
    mtest_pred_taxi_rmse = np.mean(test_pred_rmse['taxi'])

    mtest_pred_bus_mae = np.mean(test_pred_mae['bus'])
    mtest_pred_bus_mape = np.mean(test_pred_mape['bus'])
    mtest_pred_bus_rmse = np.mean(test_pred_rmse['bus'])

    mtest_pred_speed_mae = np.mean(test_pred_mae['speed'])
    mtest_pred_speed_mape = np.mean(test_pred_mape['speed'])
    mtest_pred_speed_rmse = np.mean(test_pred_rmse['speed'])

    log = 'Test Total Loss: {:.4f}\n' \
          'Test Eps KL: {:.4f}\t\t\tTest Z KL: {:.4f} \n' \
          'Test Rec Loss: {:.4f}\t\t\tTest Pred Loss: {:.4f} \n' \
          'Test Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Rec  Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
          'Test Rec  Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n\n' \
          'Test Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
          'Test Pred Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
    logger.info(log.format(mtest_total_loss, mtest_eps_kl_loss, mtest_z_kl_loss,
                           mtest_rec_loss, mtest_pred_loss,
                           mtest_rec_bike_mae, mtest_rec_bike_rmse, mtest_rec_bike_mape,
                           mtest_rec_taxi_mae, mtest_rec_taxi_rmse, mtest_rec_taxi_mape,
                           mtest_rec_bus_mae,  mtest_rec_bus_rmse , mtest_rec_bus_mape,
                           mtest_rec_speed_mae, mtest_rec_speed_rmse, mtest_rec_speed_mape,

                           mtest_pred_bike_mae, mtest_pred_bike_rmse, mtest_pred_bike_mape,
                           mtest_pred_taxi_mae, mtest_pred_taxi_rmse, mtest_pred_taxi_mape,
                           mtest_pred_bus_mae, mtest_pred_bus_rmse, mtest_pred_bus_mape,
                           mtest_pred_speed_mae, mtest_pred_speed_rmse, mtest_pred_speed_mape,))

    predicts = torch.cat(test_outputs_list, dim=0)

    if mode == 'Test':
        pred_all = predicts.cpu()
        path_save_pred = os.path.join('Results', cfg['model_name'], 'exp{:d}'.format(cfg['expid']), 'result_pred')
        if not os.path.exists(path_save_pred):
            os.makedirs(path_save_pred, exist_ok=True)

        name = 'exp{:s}_Test_Loss:{:.4f}_mae:{:.4f}_rmse:{:.4f}_mape:{:.4f}'. \
            format(cfg['model_name'], mtest_pred_loss, mtest_pred_speed_mae, mtest_pred_speed_rmse, mtest_pred_speed_mape)
        path = os.path.join(path_save_pred, name)
        np.save(path, pred_all)
        logger.info('result of prediction has been saved, path: {}'.format(path))
        logger.info('shape: ' + str(pred_all.shape))

        logger.info('\n' + str(F.relu(
            torch.tanh(cfg['model']['amplify_alpha'] * engine.CausalHMM.SCM.Weight_DAG.detach())).cpu().numpy()))

    return mtest_pred_loss, mtest_pred_speed_mae, mtest_pred_speed_rmse, mtest_pred_speed_mape


def baseline_test(runid,
                  CausalHMM,
                  dataloader,
                  device,
                  logger,
                  cfg):
    bike_scalar = dataloader['scalar_bike']
    taxi_scalar = dataloader['scalar_taxi']
    bus_scalar = dataloader['scalar_bus']
    speed_scalar = dataloader['scalar_speed']

    scalar = [bike_scalar, taxi_scalar, bus_scalar, speed_scalar]

    engine = Trainer(CausalHMM=CausalHMM,
                     base_lr=cfg['train']['base_lr'],
                     weight_decay=cfg['train']['weight_decay'],
                     milestones=cfg['train']['milestones'],
                     lr_decay_ratio=cfg['train']['lr_decay_ratio'],
                     min_learning_rate=cfg['train']['min_learning_rate'],
                     max_grad_norm=cfg['train']['max_grad_norm'],
                     num_for_target=cfg['data']['num_for_target'],
                     num_for_predict=cfg['data']['num_for_predict'],
                     loss_weight=cfg['data']['num_for_predict'],
                     scaler=scalar,
                     device=device,
                     DAG_loss_weight=cfg['train']['DAG_loss_weight'],
                     )

    best_mode_path = cfg['train']['best_mode']
    logger.info("loading {}".format(best_mode_path))

    save_dict = torch.load(best_mode_path)
    engine.CausalHMM.load_state_dict(save_dict['CausalHMM_state_dict'], strict=False)
    logger.info('model load success! {}\n'.format(best_mode_path))

    total_param = 0
    logger.info('Net\'s state_dict:')
    for param_tensor in engine.CausalHMM.state_dict():
        logger.info(param_tensor + '\t' + str(engine.CausalHMM.state_dict()[param_tensor].size()))
        total_param += np.prod(engine.CausalHMM.state_dict()[param_tensor].size())
    logger.info('Net\'s total params:{:d}\n'.format(int(total_param)))

    logger.info('Optimizer\'s state_dict:')
    for var_name in engine.CausalHMM_optimizer.state_dict():
        logger.info(var_name + '\t' + str(engine.CausalHMM_optimizer.state_dict()[var_name]))

    nParams = sum([p.nelement() for p in CausalHMM.parameters()])
    logger.info('Number of model parameters is {:d}\n'.format(int(nParams)))

    mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid, engine, dataloader, device,
                                                                                    logger, cfg, mode='Test')

    return mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape, \
           mtest_total_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape
