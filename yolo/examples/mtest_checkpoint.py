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

#==============this code added==================================================================:
# import sys
# sys.path.append("pycharm-debug-py3k.egg")
# import pydevd
# pydevd.settrace('143.89.222.167', port=12345, stdoutToServer=True, stderrToServer=True)
#================================================================================================

import brambox.boxes as bbb
import vedanet as vn
from utils.envs import initEnv
from examples.eval import generate_aps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OneDet: an one stage framework based on PyTorch')
    parser.add_argument('model_name', help='model name', default=None)
    args = parser.parse_args()

    train_flag = 2
    config = initEnv(train_flag=train_flag, model_name=args.model_name,checkpoint=True)

    log.info('Config\n\n%s\n' % pformat(config))

    # init env
    hyper_params = vn.hyperparams.HyperParams(config, train_flag=train_flag)

    # init and run eng
    vn.engine.MetaTest(hyper_params)

