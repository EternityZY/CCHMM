# !/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import torch
import copy
import sys
# sys.path.append('../..')
from torch.utils.data import Dataset, DataLoader
from tools.utils import StandardScaler


def load_dataset(dataset_dir,
                 train_batch_size,
                 valid_batch_size=None,
                 test_batch_size=None,
                 logger=None,
                 cfg=None):
    cat_data = np.load(dataset_dir, allow_pickle=True)

    if cfg['test_only']:
        all_data = {
            'test': {
                'x': np.concatenate(
                    (cat_data['test_x'].transpose((0, 2, 1, 3)), cat_data['test_target'].transpose((0, 2, 1, 3))),
                    axis=2),
                # [batch, node_num, time, dim]
                'x_time': np.concatenate((cat_data['test_x_time'], cat_data['test_target_time']), axis=1),
                'pos': cat_data['test_pos'],
            },
            'time_feature_index': cat_data['time_feature_index'].item(),
            'time_weather_data': cat_data['time_weather_data'],
        }
        scaler_bike = StandardScaler(mean=31.909935300718182,
                                     std=47.2152959892601)
        scaler_bus = StandardScaler(mean=37.93066785025187,
                                    std=64.60582188069088)
        scaler_taxi = StandardScaler(mean=17.39375995937033,
                                     std=29.45699995546719)
        scaler_speed = StandardScaler(mean=36.86575495183341,
                                      std=7.437862659846049)
    else:
        all_data = {
            'train': {
                'x': np.concatenate(
                    (cat_data['train_x'].transpose((0, 2, 1, 3)), cat_data['train_target'].transpose((0, 2, 1, 3))),
                    axis=2),  # [batch, node_num, time, dim]
                'x_time': np.concatenate((cat_data['train_x_time'], cat_data['train_target_time']), axis=1),
                'pos': cat_data['train_pos'],
            },
            'val': {
                'x': np.concatenate(
                    (cat_data['val_x'].transpose((0, 2, 1, 3)), cat_data['val_target'].transpose((0, 2, 1, 3))), axis=2),
                # [batch, node_num, time, dim]
                'x_time': np.concatenate((cat_data['val_x_time'], cat_data['val_target_time']), axis=1),
                'pos': cat_data['val_pos'],
            },
            'test': {
                'x': np.concatenate(
                    (cat_data['test_x'].transpose((0, 2, 1, 3)), cat_data['test_target'].transpose((0, 2, 1, 3))), axis=2),
                # [batch, node_num, time, dim]
                'x_time': np.concatenate((cat_data['test_x_time'], cat_data['test_target_time']), axis=1),
                'pos': cat_data['test_pos'],
            },
            'time_feature_index': cat_data['time_feature_index'].item(),
            'time_weather_data': cat_data['time_weather_data'],
        }

        train_bike = np.concatenate(all_data['train']['x'][:, :, :, 0:2])
        train_bus = np.concatenate(all_data['train']['x'][:, :, :, 2:4])
        train_taxi = np.concatenate(all_data['train']['x'][:, :, :, 4:6])
        train_speed = np.concatenate(all_data['train']['x'][:, :, :, 6:7])

        scaler_bike = StandardScaler(mean=train_bike.mean(),
                                     std=train_bike.std())
        scaler_bus = StandardScaler(mean=train_bus.mean(),
                                    std=train_bus.std())
        scaler_taxi = StandardScaler(mean=train_taxi.mean(),
                                     std=train_taxi.std())
        scaler_speed = StandardScaler(mean=train_speed.mean(),
                                      std=train_speed.std())


    if not cfg['test_only']:
        train_dataset = traffic_demand_prediction_dataset(all_data['train']['x'],
                                                          all_data['train']['x_time'],
                                                          all_data['train']['pos'],
                                                          )

        val_dataset = traffic_demand_prediction_dataset(all_data['val']['x'],
                                                        all_data['val']['x_time'],
                                                        all_data['val']['pos'],
                                                        None,
                                                        )

    test_dataset = traffic_demand_prediction_dataset(all_data['test']['x'],
                                                     all_data['test']['x_time'],
                                                     all_data['test']['pos'],
                                                     None,
                                                     )

    dataloader = {}
    if not cfg['test_only']:
        dataloader['train'] = DataLoader(dataset=train_dataset, shuffle=True, batch_size=train_batch_size)  # num_workers=16
        dataloader['val'] = DataLoader(dataset=val_dataset, shuffle=False, batch_size=valid_batch_size)
    dataloader['test'] = DataLoader(dataset=test_dataset, shuffle=False, batch_size=test_batch_size)

    dataloader['scalar_bike'] = scaler_bike
    dataloader['scalar_bus'] = scaler_bus
    dataloader['scalar_taxi'] = scaler_taxi
    dataloader['scalar_speed'] = scaler_speed
    dataloader['time_feature_index'] = all_data['time_feature_index']
    dataloader['time_weather_data'] = all_data['time_weather_data']

    if logger != None:
        if not cfg['test_only']:
            logger.info(('train x', all_data['train']['x'].shape))
            logger.info(('train x time', all_data['train']['x_time'].shape))
            logger.info(('train pos', all_data['train']['pos'].shape))

            logger.info('\n')
            logger.info(('val x', all_data['val']['x'].shape))
            logger.info(('val x time', all_data['val']['x_time'].shape))
            logger.info(('val pos', all_data['val']['pos'].shape))

        logger.info('\n')
        logger.info(('test x', all_data['test']['x'].shape))
        logger.info(('test x time', all_data['test']['x_time'].shape))
        logger.info(('test pos', all_data['test']['pos'].shape))

        logger.info('\n')
        logger.info('Bike scaler.mean : {}, scaler.std : {}'.format(scaler_bike.mean,
                                                                    scaler_bike.std))
        logger.info('\n')
        logger.info('Taxi scaler.mean : {}, scaler.std : {}'.format(scaler_taxi.mean,
                                                                    scaler_taxi.std))
        logger.info('\n')
        logger.info('Bus scaler.mean : {}, scaler.std : {}'.format(scaler_bus.mean,
                                                                   scaler_bus.std))
        logger.info('\n')
        logger.info('Speed scaler.mean : {}, scaler.std : {}'.format(scaler_speed.mean,
                                                                     scaler_speed.std))

        logger.info('\n')
        logger.info('time feature index : {}'.format(all_data['time_feature_index']))
        logger.info('time weather data : {}'.format(all_data['time_weather_data']))

    return dataloader


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, x_time, pos, target_cl=None):
        time = x_time[..., :2]
        weather = x_time[..., 2:]
        time = self.__generate_one_hot(time)
        x_time = np.concatenate([time, weather], axis=-1)

        self.x = torch.tensor(x).to(torch.float32)
        self.x_time = torch.tensor(x_time).to(torch.float32)
        # self.pos = torch.tensor(pos)
        self.x_time = torch.repeat_interleave(self.x_time.unsqueeze(dim=1), repeats=self.x.shape[1], dim=1)
        if target_cl is not None:
            self.target_cl = torch.tensor(target_cl).to(torch.float32)
        else:
            self.target_cl = None

    def __getitem__(self, item):

        if self.target_cl is not None:
            return self.x[item, :, :, 0:2], \
                   self.x[item, :, :, 2:4], \
                   self.x[item, :, :, 4:6], \
                   self.x[item, :, :, 6:7], \
                   self.x_time[item], \
                # self.pos[item], self.target_cl[item]
        else:
            return self.x[item, :, :, 0:2], \
                   self.x[item, :, :, 2:4], \
                   self.x[item, :, :, 4:6], \
                   self.x[item, :, :, 6:7], \
                   self.x_time[item], \
                # self.pos[item], self.pos[item]

    def __len__(self):
        return self.x.shape[0]

    def __generate_one_hot(self, arr):
        dayofweek_len = 7
        timeofday_len = int(arr[:, :, 1].max()) + 1

        dayofweek = np.zeros((arr.shape[0], arr.shape[1], dayofweek_len))
        timeofday = np.zeros((arr.shape[0], arr.shape[1], timeofday_len))

        for i in range(arr.shape[0]):
            dayofweek[i] = np.eye(dayofweek_len)[arr[:, :, 0][i].astype(np.int)]

        for i in range(arr.shape[0]):
            timeofday[i] = np.eye(timeofday_len)[arr[:, :, 1][i].astype(np.int)]
        arr = np.concatenate([dayofweek, timeofday, arr[..., 2:]], axis=-1)
        return arr


if __name__ == '__main__':
    pass