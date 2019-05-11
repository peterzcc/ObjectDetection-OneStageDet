import os
import argparse
import logging as log
import time
from statistics import mean
import numpy as np
import torch
from torchvision import transforms as tf
from pprint import pformat

import sys
sys.path.insert(0, '.')

import brambox.boxes as bbb
import vedanet as vn
from utils.envs import initEnv, randomSeeding

#==============this code added==================================================================:
# import sys
# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
# pydevd.settrace('143.89.222.167', port=12345, stdoutToServer=True, stderrToServer=True)
#================================================================================================

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    parser.add_argument('model_name', help='model name', default=None)
    args = parser.parse_args()

    train_flag = 1
    config = initEnv(train_flag=train_flag, model_name=args.model_name)
    #randomSeeding(0)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)

    # int eng
    eng = vn.engine.VOCTrainingEngine(hyper_params)

    # run eng
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch

    log.info('\nDuration of {} batches: {} seconds [{} sec/batch]'.format(b2-b1, t2-t1, round((t2-t1)/(b2-b1), 3)))
