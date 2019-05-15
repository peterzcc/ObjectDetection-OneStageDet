import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os

from .. import data as vn_data
from .. import models
from . import engine
import time
from utils.test import voc_wrapper
from examples.eval import generate_aps
from ._voc_test import CustomDataset

__all__ = ['MetaTest']


def MetaTest(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    conf_thresh = hyper_params.conf_thresh
    network_size = hyper_params.network_size
    labels = hyper_params.labels
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    nms_thresh = hyper_params.nms_thresh
    #prefix = hyper_params.prefix
    results = hyper_params.results
    model_name = hyper_params.model_name

    model_cls = models.Yolov2Wrn
    if model_name:
        model_cls = models.__dict__[model_name]

    # net = model_cls(hyper_params.classes, hyper_params.weights, train_flag=1,
    #                 clear=hyper_params.clear,
    #                 loss_allobj=hyper_params.loss_allobj,
    #                 use_yolo_loss=hyper_params.use_yolo_loss)
    # meta_param_size = net.meta_param_size
    print(model_name)
    test_args = {'conf_thresh': conf_thresh, 'network_size': network_size, 'labels': labels}
    net = model_cls(hyper_params.classes, weights, train_flag=2, test_args=test_args, reweights_file=hyper_params.reweights)
    net.eval()
    log.info('Net structure\n%s' % net)
    #import pdb
    #pdb.set_trace()
    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(hyper_params),
        batch_size = batch,
        shuffle = False,
        drop_last = False,
        num_workers = nworkers if use_cuda else 0,
        pin_memory = pin_mem if use_cuda else False,
        collate_fn = vn_data.list_collate,
    )

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0
    t0 = time.time()
    for idx, (data, box) in enumerate(loader):
        if (idx + 1) % 20 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        t1 = time.time()
        # print('prepare data took {:.4f}s'.format(t1 - t0))
        with torch.no_grad():
            output, loss = net(data, box)
            t2 = time.time()
            # print('forward took {:.4f}s'.format(t2 - t1))
        key_val = len(anno)
        anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
        det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})
        t0 = time.time()
        # print('update took {:.4f}s'.format(t0 - t2))

    netw, neth = network_size
    reorg_dets = voc_wrapper.reorgDetection(det, netw, neth) #, prefix)
    if not os.path.isdir(results):
        os.mkdir(results)
    voc_wrapper.genResults(reorg_dets, results, nms_thresh)
    generate_aps(results_root=results)


