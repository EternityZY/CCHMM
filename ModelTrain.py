# !/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from tqdm import tqdm

from ModelTest import model_val, model_test
from helper import Trainer
from tools.metrics import record

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def baseline_train(runid,
                   CausalHMM,
                   model_name,
                   dataloader,
                   device,
                   logger,
                   cfg):
    print("start training...", flush=True)
    save_path = os.path.join('Results', cfg['model_name'], 'exp{:d}'.format(cfg['expid']), 'ckpt')
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
                     loss_weight=cfg['train']['loss_weight'],
                     scaler=scalar,
                     device=device,
                     DAG_loss_weight=cfg['train']['DAG_loss_weight'],
                     )

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

    if cfg['train']['load_initial']:
        best_mode_path = cfg['train']['best_mode']
        logger.info("loading {}".format(best_mode_path))

        save_dict = torch.load(best_mode_path)
        engine.CausalHMM.load_state_dict(save_dict['CausalHMM_state_dict'], strict=False)
    else:
        logger.info('Start training from scratch!')
    save_dict = dict()


    begin_epoch = cfg['train']['epoch_start']
    epochs = cfg['train']['epochs']
    tolerance = cfg['train']['tolerance']

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = -1
    stable_count = 0

    logger.info('begin_epoch: {}, total_epochs: {}, patient: {}, best_val_loss: {:.4f}'.
                format(begin_epoch, epochs, tolerance, best_val_loss))

    for epoch in range(begin_epoch, begin_epoch + epochs + 1):

        train_total_loss = []
        train_eps_kl_loss = []
        train_z_kl_loss = []
        train_rec_loss = []
        train_pred_loss = []

        train_rec_mape = {}
        train_rec_rmse = {}
        train_rec_mae = {}

        train_rec_mae['bike']=[]
        train_rec_mae['taxi']=[]
        train_rec_mae['bus']=[]
        train_rec_mae['speed']=[]

        train_rec_rmse['bike']=[]
        train_rec_rmse['taxi']=[]
        train_rec_rmse['bus']=[]
        train_rec_rmse['speed']=[]

        train_rec_mape['bike']=[]
        train_rec_mape['taxi']=[]
        train_rec_mape['bus']=[]
        train_rec_mape['speed']=[]

        train_gen_mape = {}
        train_gen_rmse = {}
        train_gen_mae = {}

        train_gen_mae['bike']=[]
        train_gen_mae['taxi']=[]
        train_gen_mae['bus']=[]
        train_gen_mae['speed']=[]

        train_gen_rmse['bike']=[]
        train_gen_rmse['taxi']=[]
        train_gen_rmse['bus']=[]
        train_gen_rmse['speed']=[]

        train_gen_mape['bike']=[]
        train_gen_mape['taxi']=[]
        train_gen_mape['bus']=[]
        train_gen_mape['speed']=[]
        t1 = time.time()

        train_dataloder = dataloader['train']
        train_tqdm_loader = tqdm(enumerate(train_dataloder))

        for iter, (bike, bus, taxi, speed, pos) in train_tqdm_loader:

            """
            
            """
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
            gen_mae, gen_rmse, gen_mape = engine.train(tpos, weather, bike_flow, taxi_flow, bus_flow, speed)

            train_total_loss.append(total_loss)
            train_eps_kl_loss.append(eps_kl_loss)
            train_z_kl_loss.append(z_kl_loss)
            train_rec_loss.append(rec_loss)
            train_pred_loss.append(pred_loss)

            record(train_rec_mae, train_rec_rmse, train_rec_mape, rec_mae, rec_rmse, rec_mape)
            record(train_gen_mae, train_gen_rmse, train_gen_mape, gen_mae, gen_rmse, gen_mape, only_last=True)

        engine.CausalHMM_scheduler.step()

        t2 = time.time()
        train_time.append(t2 - t1)

        s1 = time.time()
        mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                             engine=engine,
                                                                                             dataloader=dataloader,
                                                                                             device=device,
                                                                                             logger=logger,
                                                                                             cfg=cfg,
                                                                                             epoch=epoch)
        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_total_loss = np.mean(train_total_loss)
        mtrain_eps_kl_loss = np.mean(train_eps_kl_loss)
        mtrain_z_kl_loss = np.mean(train_z_kl_loss)
        mtrain_rec_loss = np.mean(train_rec_loss)
        mtrain_pred_loss = np.mean(train_pred_loss)

        mtrain_rec_bike_mae = np.mean(train_rec_mae['bike'])
        mtrain_rec_bike_mape = np.mean(train_rec_mape['bike'])
        mtrain_rec_bike_rmse = np.mean(train_rec_rmse['bike'])

        mtrain_rec_taxi_mae = np.mean(train_rec_mae['taxi'])
        mtrain_rec_taxi_mape = np.mean(train_rec_mape['taxi'])
        mtrain_rec_taxi_rmse = np.mean(train_rec_rmse['taxi'])

        mtrain_rec_bus_mae = np.mean(train_rec_mae['bus'])
        mtrain_rec_bus_mape = np.mean(train_rec_mape['bus'])
        mtrain_rec_bus_rmse = np.mean(train_rec_rmse['bus'])

        mtrain_rec_speed_mae = np.mean(train_rec_mae['speed'])
        mtrain_rec_speed_mape = np.mean(train_rec_mape['speed'])
        mtrain_rec_speed_rmse = np.mean(train_rec_rmse['speed'])


        mtrain_pred_bike_mae = np.mean(train_gen_mae['bike'])
        mtrain_pred_bike_mape = np.mean(train_gen_mape['bike'])
        mtrain_pred_bike_rmse = np.mean(train_gen_rmse['bike'])

        mtrain_pred_taxi_mae = np.mean(train_gen_mae['taxi'])
        mtrain_pred_taxi_mape = np.mean(train_gen_mape['taxi'])
        mtrain_pred_taxi_rmse = np.mean(train_gen_rmse['taxi'])

        mtrain_pred_bus_mae = np.mean(train_gen_mae['bus'])
        mtrain_pred_bus_mape = np.mean(train_gen_mape['bus'])
        mtrain_pred_bus_rmse = np.mean(train_gen_rmse['bus'])

        mtrain_pred_speed_mae = np.mean(train_gen_mae['speed'])
        mtrain_pred_speed_mape = np.mean(train_gen_mape['speed'])
        mtrain_pred_speed_rmse = np.mean(train_gen_rmse['speed'])

        if (epoch - 1) % cfg['train']['print_every'] == 0:
            log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
            logger.info(log.format(epoch, (s2 - s1)))

            log = 'Epoch: {:03d}, Train Total Loss: {:.4f} Learning rate: {}\n' \
                  'Train Eps KL: {:.4f}\t\t\tTrain Z KL: {:.4f} \n' \
                  'Train Rec Loss: {:.4f}\t\t\tTrain Pred Loss: {:.4f} \n' \
                  'Train Rec  Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n' \
                  'Train Rec  Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f} \n\n' \
                  'Train Pred Bike  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Taxi  MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Bus   MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n' \
                  'Train Pred Speed MAE: {:.4f}\t\t\tRMSE: {:.4f}\t\t\tMAPE: {:.4f}\n'
            logger.info(log.format(epoch, mtrain_total_loss, str(engine.CausalHMM_scheduler.get_lr()),
                                   mtrain_eps_kl_loss, mtrain_z_kl_loss,
                                   mtrain_rec_loss, mtrain_pred_loss,
                                   mtrain_rec_bike_mae, mtrain_rec_bike_rmse, mtrain_rec_bike_mape,
                                   mtrain_rec_taxi_mae, mtrain_rec_taxi_rmse, mtrain_rec_taxi_mape,
                                   mtrain_rec_bus_mae,  mtrain_rec_bus_rmse , mtrain_rec_bus_mape,
                                   mtrain_rec_speed_mae, mtrain_rec_speed_rmse, mtrain_rec_speed_mape,

                                   mtrain_pred_bike_mae, mtrain_pred_bike_rmse, mtrain_pred_bike_mape,
                                   mtrain_pred_taxi_mae, mtrain_pred_taxi_rmse, mtrain_pred_taxi_mape,
                                   mtrain_pred_bus_mae, mtrain_pred_bus_rmse, mtrain_pred_bus_mape,
                                   mtrain_pred_speed_mae, mtrain_pred_speed_rmse, mtrain_pred_speed_mape,
                                   ))
            logger.info('\n' + str(F.relu(torch.tanh(cfg['model']['amplify_alpha']*engine.CausalHMM.SCM.Weight_DAG.detach())).cpu().numpy()))


        his_loss.append(mvalid_pred_loss)
        if mvalid_pred_loss < best_val_loss:
            best_val_loss = mvalid_pred_loss
            epoch_best = epoch
            stable_count = 0

            save_dict.update(CausalHMM_state_dict=copy.deepcopy(engine.CausalHMM.state_dict()),
                             epoch=epoch_best,
                             best_val_loss=best_val_loss)

            ckpt_name = "exp{:d}_epoch{:d}_Val_loss:{:.2f}_mae:{:.2f}_rmse:{:.2f}_mape:{:.2f}.pth". \
                format(cfg['expid'], epoch, mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape)
            best_mode_path = os.path.join(save_path, ckpt_name)
            torch.save(save_dict, best_mode_path)
            logger.info(f'Better model at epoch {epoch_best} recorded.')
            logger.info('Best model is : {}'.format(best_mode_path))
            logger.info('\n')
        else:
            stable_count += 1
            if stable_count > tolerance:
                break

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)

    logger.info("Training finished")
    logger.info("The valid loss on best model is {:.4f}, epoch:{:d}".format(round(his_loss[bestid], 4), epoch_best))

    logger.info('Start the model test phase........')
    logger.info("loading the best model for this training phase {}".format(best_mode_path))
    save_dict = torch.load(best_mode_path)
    engine.CausalHMM.load_state_dict(save_dict['CausalHMM_state_dict'])

    mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape = model_val(runid,
                                                                                       engine=engine,
                                                                                       dataloader=dataloader,
                                                                                       device=device,
                                                                                       logger=logger,
                                                                                       cfg=cfg,
                                                                                       epoch=epoch_best)

    mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape = model_test(runid,
                                                                                    engine=engine,
                                                                                    dataloader=dataloader,
                                                                                    device=device,
                                                                                    cfg=cfg,
                                                                                    logger=logger,
                                                                                    mode='Test')

    return mvalid_pred_loss, mvalid_pred_mae, mvalid_pred_rmse, mvalid_pred_mape, \
           mtest_pred_loss, mtest_pred_mae, mtest_pred_rmse, mtest_pred_mape