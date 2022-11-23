# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_mse_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mae_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape_torch(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    preds = preds[labels > 10]
    mask = mask[labels > 10]
    labels = labels[labels > 10]

    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss) * 100


def metric(pred, real):
    mae = masked_mae_torch(pred, real, np.inf).item()
    mape = masked_mape_torch(pred, real, np.inf).item()
    rmse = masked_rmse_torch(pred, real, np.inf).item()
    return mae, mape, rmse

def metric_all(preds, reals):

    time = preds[0].shape[2]

    mae = {}
    rmse = {}
    mape = {}
    mae['bike'] = np.zeros(time)
    mae['taxi'] = np.zeros(time)
    mae['bus'] = np.zeros(time)
    mae['speed'] = np.zeros(time)
    rmse['bike'] = np.zeros(time)
    rmse['taxi'] = np.zeros(time)
    rmse['bus'] = np.zeros(time)
    rmse['speed'] = np.zeros(time)
    mape['bike'] = np.zeros(time)
    mape['taxi'] = np.zeros(time)
    mape['bus'] = np.zeros(time)
    mape['speed'] = np.zeros(time)

    if len(preds) > 1:
        for t in range(time):
            mae['bike'][t], mape['bike'][t], rmse['bike'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])
            mae['taxi'][t], mape['taxi'][t], rmse['taxi'][t] = metric(preds[1][:, :, t, :], reals[1][:, :, t, :])
            mae['bus'][t], mape['bus'][t], rmse['bus'][t] = metric(preds[2][:, :, t, :], reals[2][:, :, t, :])
            mae['speed'][t], mape['speed'][t], rmse['speed'][t] = metric(preds[3][:, :, t, :], reals[3][:, :, t, :])
    else:
        for t in range(time):
            mae['speed'][t], mape['speed'][t], rmse['speed'][t] = metric(preds[0][:, :, t, :], reals[0][:, :, t, :])

    return mae, rmse, mape

def record(all_mae, all_rmse, all_mape, mae, rmse, mape, only_last=False):

    if only_last:
        all_mae['bike'].append(mae['bike'][-1])
        all_mae['taxi'].append(mae['taxi'][-1])
        all_mae['bus'].append(mae['bus'][-1])
        all_mae['speed'].append(mae['speed'][-1])

        all_rmse['bike'].append(rmse['bike'][-1])
        all_rmse['taxi'].append(rmse['taxi'][-1])
        all_rmse['bus'].append(rmse['bus'][-1])
        all_rmse['speed'].append(rmse['speed'][-1])

        all_mape['bike'].append(mape['bike'][-1])
        all_mape['taxi'].append(mape['taxi'][-1])
        all_mape['bus'].append(mape['bus'][-1])
        all_mape['speed'].append(mape['speed'][-1])

    else:

        all_mae['bike'].append(np.mean(mae['bike']))
        all_mae['taxi'].append(np.mean(mae['taxi']))
        all_mae['bus'].append(np.mean(mae['bus']))
        all_mae['speed'].append(np.mean(mae['speed']))

        all_rmse['bike'].append(np.mean(rmse['bike']))
        all_rmse['taxi'].append(np.mean(rmse['taxi']))
        all_rmse['bus'].append(np.mean(rmse['bus']))
        all_rmse['speed'].append(np.mean(rmse['speed']))

        all_mape['bike'].append(np.mean(mape['bike']))
        all_mape['taxi'].append(np.mean(mape['taxi']))
        all_mape['bus'].append(np.mean(mape['bus']))
        all_mape['speed'].append(np.mean(mape['speed']))

