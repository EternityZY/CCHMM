# !/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import time
import yaml
import logging
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import numpy as np

from tools.utils import make_saved_dir

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import torch


def read_cfg_file(filename):
    '''
    :param filename:
    :return:
    '''
    with open(filename, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=Loader)
    return cfg


def _get_log_dir(kwargs):
    '''
    :param kwargs:
    :return:
    '''
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        expid = kwargs.get('expid')
        batch_size = kwargs['data'].get('train_batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        RNN_layers = kwargs['model'].get('RNN_layer')
        fusion_mode = kwargs['model'].get('fusion_mode')
        gcn_depth = kwargs['model'].get('gcn_depth')
        num_of_head = kwargs['model'].get('num_of_head')
        dropout_prob = kwargs['model'].get('dropout_prob')
        dropout_type = kwargs['model'].get('dropout_type')
        graph_type = kwargs['model'].get('graph_type')

        num_of_weeks = kwargs['data'].get('num_of_weeks')
        num_of_days = kwargs['data'].get('num_of_days')
        num_of_hours = kwargs['data'].get('num_of_hours')
        num_for_predict = kwargs['data'].get('num_for_predict')
        num_for_target = kwargs['data'].get('num_for_target')

        others = ''

        run_id = 'exp%d_%s_RNN%d_gdep%d_%s_w%dd%dh%d_his%d_pred%d_lr%g_batch%d_%s/' % (
            expid,
            graph_type,
            RNN_layers,
            gcn_depth,
            fusion_mode,
            num_of_weeks,
            num_of_days,
            num_of_hours,
            num_for_predict,
            num_for_target,
            learning_rate,
            batch_size,
            time.strftime('(%m-%d_%H:%M)'))

        log_dir = os.path.join(ckpt_save_dir, 'log', run_id)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def get_logger(log_dir,
               name,
               log_filename='info.log',
               level=logging.INFO,
               write_to_file=True):
    '''
    :param log_dir:
    :param name:
    :param log_filename:
    :param level:
    :param write_to_file:
    :return:
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Add file handler and stdout handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    if write_to_file is True:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def make_experiment_dir(base_dir,
                        cfg,
                        ckpt=False):


    expid = cfg['expid']
    num_of_weeks = cfg['data']['num_of_weeks']
    num_of_days = cfg['data']['num_of_days']
    num_of_hours = cfg['data']['num_of_hours']
    num_for_predict = cfg['data']['num_for_predict']
    num_for_target = cfg['data']['num_for_target']
    cluster_num = cfg['preprocess']['cluster_num']


    if ckpt ==True:
        experiment='expid{:d}_w{:d}_d{:d}_h{:d}_his{:d}_pred{:d}_cluster{:d}'. \
            format(expid, num_of_weeks, num_of_days, num_of_hours, num_for_predict, num_for_target, cluster_num)
    else:
        experiment='w{:d}_d{:d}_h{:d}_his{:d}_pred{:d}_cluster{:d}'. \
            format(num_of_weeks, num_of_days, num_of_hours, num_for_predict, num_for_target, cluster_num)

    experiment_dir = os.path.join(base_dir, experiment)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    return experiment_dir
