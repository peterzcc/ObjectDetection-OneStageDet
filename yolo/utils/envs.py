import sys
import os
import copy
from datetime import datetime
import logging
import torch
import random
import numpy as np
import subprocess
import logging

# individual packages
from .fileproc import safeMakeDirs
from .cfg_parser import getConfig

TRAIN, TEST, WEIGHTS = 1, 2, 3
def setLogging(log_dir, stdout_flag):
    safeMakeDirs(log_dir)
    dt = datetime.now()
    log_name = dt.strftime('%Y-%m-%d_time_%H_%M_%S') + '.log'

    log_fp = os.path.join(log_dir, log_name)
    # root_logger = logging.getLogger()
    # root_logger.setLevel(logging.DEBUG)
    #print os.path.abspath(log_fp)
    from importlib import reload
    reload(logging)
    if stdout_flag:
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=log_fp, format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)


def combineConfig(cur_cfg, train_flag):
    ret_cfg = {}
    for k, v in cur_cfg.items():
        if k == 'train' or k == 'test' or k == 'speed' or k == "weights":
            continue
        ret_cfg[k] = v
    if train_flag == 1:
        key = 'train'
    elif train_flag == 2:
        key = 'test'
    elif train_flag == 3:
        key = 'weights'
    else:
        key = 'speed'
    for k, v in cur_cfg[key].items():
        ret_cfg[k] = v
    return ret_cfg


def exec_cmd(cmd):
    with subprocess.Popen(cmd,
                          shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,) as sp:
        while True:
            out = sp.stdout.read(1)
            if out == b'' and sp.poll() != None:
                break
            if out != b'':
                sys.stdout.write(out.decode())
                sys.stdout.flush()


def initEnv(train_flag, model_name: str, checkpoint=False):
    cfgs_root = 'cfgs'
    if model_name.endswith(".yml"):
        assert os.path.exists(model_name)
        cfg_file = model_name
        cur_cfg = getConfig(cfgs_root, model_name, cfg_file)
        model_name = cur_cfg['model_name']
        root_dir = cur_cfg['output_root']
    else:
        cfg_file = None
        cur_cfg = getConfig(cfgs_root, model_name, cfg_file)
        root_dir = cur_cfg['output_root']
        cur_cfg['model_name'] = model_name
    version = cur_cfg.get('output_version', None)
    if cfg_file:
        file_name = os.path.split(cfg_file)[-1].split('.')[0]
        if not version:
            version = file_name

    work_dir = os.path.join(root_dir, model_name, version)

    backup_name = cur_cfg['backup_name']
    log_name = cur_cfg['log_name']

    backup_dir = os.path.join(work_dir, backup_name)
    log_dir = os.path.join(work_dir, log_name)

    if checkpoint and "checkpoint" in cur_cfg:
        checkpoint_num = cur_cfg["checkpoint"]
        weight_path = os.path.join(backup_dir, f"weights_{checkpoint_num}.pt")
        meta_weight_path = os.path.join(backup_dir, f"meta_weights_{checkpoint_num}.pt")
        reweight_path = os.path.join(backup_dir, f"reweights_{checkpoint_num}.pkl")
        results_path = os.path.join(work_dir, f"results_{checkpoint_num}")
        # if not (os.path.isfile(weight_path)
        #         and os.path.isfile(meta_weight_path)) \
        #         and "server" in cur_cfg:
        #     exec_cmd(f"rsync --checksum -RP {cur_cfg['server']}/./{weight_path} .")
        #     exec_cmd(f"rsync --checksum -RP {cur_cfg['server']}/./{meta_weight_path} .")
    else:
        weight_path = None
        meta_weight_path = None
        reweight_path = None
        results_path = None

    if train_flag == TRAIN:
        safeMakeDirs(backup_dir)
        stdout_flag = cur_cfg['train']['stdout']
        setLogging(log_dir, stdout_flag)

        gpus = cur_cfg['train']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

        cur_cfg['train']['backup_dir'] = backup_dir
    elif train_flag == TEST:
        stdout_flag = cur_cfg['test']['stdout']
        setLogging(log_dir, stdout_flag)
        gpus = cur_cfg['test']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        if weight_path is not None and reweight_path is not None:
            cur_cfg['test']['weights'] = weight_path
            cur_cfg['test']['reweights'] = reweight_path
            cur_cfg['test']['results'] = results_path
    elif train_flag == WEIGHTS:
        stdout_flag = cur_cfg['weights']['stdout']
        setLogging(log_dir, stdout_flag)

        gpus = cur_cfg['weights']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        if weight_path is not None and meta_weight_path is not None:
            cur_cfg['weights']['weights'] = weight_path
            cur_cfg['weights']['meta_weights'] = meta_weight_path
            cur_cfg['weights']['results'] = reweight_path
    else:
        gpus = cur_cfg['speed']['gpus']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

    ret_cfg = combineConfig(cur_cfg, train_flag)
    return ret_cfg


def randomSeeding(seed):
     np.random.seed(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)


if __name__ == '__main__':
    pass
