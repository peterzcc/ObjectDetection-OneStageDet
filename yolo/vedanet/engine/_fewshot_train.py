import logging as log
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tf
from statistics import mean
import os
import random
from PIL import Image
import copy
import numpy as np
from collections import deque
from .. import data
from .. import models
from . import SyncDualEngine
from .. import network

__all__ = ['FewshotTrainingEngine']


class VOCDataset(data.MetaboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        root = hyper_params.data_root
        flip = hyper_params.flip
        jitter = hyper_params.jitter
        hue, sat, val = hyper_params.hue, hyper_params.sat, hyper_params.val
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        rf = data.transform.RandomFlip(flip)
        rc = data.transform.RandomCropLetterbox(self, jitter)
        hsv = data.transform.HSVShift(hue, sat, val)
        it = tf.ToTensor()

        img_tf = data.transform.Compose([rc, rf, hsv, it])
        anno_tf = data.transform.Compose([rc, rf])

        def identify(img_id):
            # return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)


class VOCMetaDataset(data.OneboxDataset):
    def __init__(self, hyper_params, annos):
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        it = tf.ToTensor()

        lb = data.transform.Letterbox(hyper_params.meta_input_shape)

        self.meta_tf = data.transform.Compose([lb, it])
        self.meta_anno_tf = data.transform.Compose([lb])

        def identify(img_id):
            # return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCMetaDataset, self).__init__(annos, network_size, labels, identify, self.meta_tf, self.meta_anno_tf)


class FewshotSampleManager(object):
    def __init__(self, box_dataset: data.OneboxDataset, input_batchsize, k_shot,):
        self.box_dataset = box_dataset
        self.batchsize = input_batchsize
        self.k_shot = k_shot
        self.query_batches = None
        self.support_batches = None
        self.seed = 1
        self.rng = np.random.RandomState(self.seed)

    def prepare_batches(self):
        query_img_ids = deque(self.rng.permutation(np.arange(len(self.box_dataset.fileid_2_boxid), dtype=int)))
        support_box_ids = [deque(self.rng.permutation(self.box_dataset.cls_2_boxid[ci]))
                           for ci, boxids in enumerate(self.box_dataset.cls_2_boxid)]

        self.query_batches, self.support_batches = [], []
        finished_sampling = False
        is_file_seen = np.zeros(len(self.box_dataset.fileid_2_boxid), dtype=bool)
        while not finished_sampling:
            is_box_seen = np.zeros(len(self.box_dataset.file_box), dtype=bool)
            this_query_batch = []
            while len(this_query_batch) < self.batchsize:
                if len(query_img_ids) == 0:
                    return
                this_img_id = query_img_ids.popleft()
                if not is_file_seen[this_img_id]:
                    this_query_batch.append(this_img_id)
                    is_file_seen[this_img_id] = True
                    is_box_seen[self.box_dataset.fileid_2_boxid[this_img_id]] = True
            this_support_batch = []
            for shot_i in range(self.k_shot):
                for ci, boxids in enumerate(self.box_dataset.cls_2_boxid):
                    finished_ci = False
                    this_box_id = -1
                    while not finished_ci:
                        if len(support_box_ids[ci]) == 0:
                            unseen_boxes = np.logical_not(is_box_seen[boxids])
                            if np.any(unseen_boxes):
                                boxids_ci = (np.array(boxids))
                                support_box_ids[ci] = \
                                    deque(self.rng.permutation(
                                        boxids_ci[unseen_boxes]))
                            else:
                                return
                        this_box_id = support_box_ids[ci].popleft()
                        if not is_box_seen[this_box_id]:
                            finished_ci = True
                    this_support_batch.append(this_box_id)
                    # is_box_seen[this_box_id] = True
                    # is_file_seen[self.box_dataset.boxid_2_fileid[this_box_id]] = True
            self.query_batches.append(this_query_batch)
            self.support_batches.append(this_support_batch)
        return

    def get_query_batches(self):
        if self.query_batches is None and self.support_batches is not None:
            raise ValueError("support not used")
        else:
            if self.query_batches is None:
                self.prepare_batches()
            query_batches = self.query_batches.copy()
            self.query_batches = None
            return query_batches

    def get_support_batches(self):
        if self.support_batches is None and self.query_batches is not None:
            raise ValueError("query not used")
        else:
            if self.support_batches is None:
                self.prepare_batches()
            support_batches = self.support_batches.copy()
            self.support_batches = None
            return support_batches


class FewshotTrainingEngine(SyncDualEngine):
    """ This is a custom engine for this training cycle """

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        # all in args
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches

        self.classes = hyper_params.classes

        self.cuda = hyper_params.cuda
        self.backup_dir = hyper_params.backup_dir

        log.debug('Creating network')
        model_name = hyper_params.model_name
        assert model_name == "Yolov2_Meta"
        net = models.Yolov2_Meta(hyper_params.classes, hyper_params.weights, train_flag=1,
                                 clear=hyper_params.clear,
                                 loss_allobj=hyper_params.loss_allobj,
                                 use_yolo_loss=hyper_params.use_yolo_loss)
        metanet = network.metanet.Metanet(hyper_params.classes, weights_file=hyper_params.meta_weights,
                                          use_dummy_reweight=hyper_params.use_dummy_reweight)

        log.info('Net structure\n\n%s\n' % net)
        self.multi_gpu = False
        if self.cuda:
            net.cuda()
            if torch.cuda.device_count() > 1:
                metanet_device = 1
                network.metanet.Metanet.device = metanet_device
            else:
                metanet_device = 0
                network.metanet.Metanet.device = metanet_device
            metanet.cuda(device=network.metanet.Metanet.device)

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        self.max_input_shape = hyper_params.network_size
        log.info(f'Adjusting learning rate to [{learning_rate}]')


        meta_net_parameters = [
            {'params': metanet.parameters(), },
            {'params': net.backbone.parameters()},
            {'params': net.head.parameters()}
        ]
        optim = torch.optim.SGD(meta_net_parameters, lr=learning_rate / batch, momentum=momentum, dampening=0,
                                weight_decay=decay * batch)

        log.debug('Creating dataloader')
        dataset = VOCDataset(hyper_params)
        meta_dataset = VOCMetaDataset(hyper_params, annos=dataset.annos)
        k_shot = 1
        sample_manager = FewshotSampleManager(meta_dataset,
                                              input_batchsize=self.mini_batch_size, k_shot=k_shot)
        dataloader = data.DataLoader(
            dataset,
            # batch_size=self.mini_batch_size,
            # shuffle=False,
            # drop_last=True,
            num_workers=hyper_params.nworkers if self.cuda else 0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=data.list_collate,
            resize_range=(10, self.max_input_shape[0]//32),
            batch_sampler= data.ListBatchSampler(f_get_batches=sample_manager.get_query_batches,
                                                 input_dimension=dataset.input_dim)
        )
        meta_dataloader = data.DataLoader(
            meta_dataset,
            # batch_size=k_shot,
            # shuffle=False,
            # drop_last=True,
            num_workers=0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=default_collate,
            batch_sampler=data.ListBatchSampler(f_get_batches=sample_manager.get_support_batches,
                                                input_dimension=dataset.input_dim)
        )

        super(FewshotTrainingEngine, self).__init__(net, metanet, optim, dataloader, meta_dataloader)


        self.nloss = self.network.nloss

        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]

    def start(self):
        log.debug('Creating additional logging objects')
        hyper_params = self.hyper_params

        lr_steps = hyper_params.lr_steps
        lr_rates = hyper_params.lr_rates

        bp_steps = hyper_params.bp_steps
        bp_rates = hyper_params.bp_rates
        backup = hyper_params.backup

        rs_steps = hyper_params.rs_steps
        rs_rates = hyper_params.rs_rates
        resize = hyper_params.resize

        self.add_rate('learning_rate', lr_steps, [lr / self.batch_size for lr in lr_rates])
        self.add_rate('backup_rate', bp_steps, bp_rates, backup)
        self.add_rate('resize_rate', rs_steps, rs_rates, resize)

        self.dataloader.change_input_dim()

    def process_meta_img(self, meta_imgs):
        if self.cuda:
            meta_imgs = meta_imgs.cuda()
        if meta_imgs.shape[0] == 1:
            meta_imgs = meta_imgs[0]
        reweights = self.dist_meta_network(meta_imgs)
        # log.info(f"reweights l2 norm:\n {torch.norm(reweights,p=2,dim=2).view(-1)}")
        return reweights

    def process_batch(self, data):
        data, target, reweights = data
        # to(device)
        # if self.cuda:
        #     data = data.cuda()
        # meta_imgs = torch.autograd.Variable(meta_imgs, requires_grad=True)

        loss = self.network((data, reweights), target)
        loss.backward(retain_graph=True)

        for ii in range(self.nloss):
            self.train_loss[ii]['tot'].append(self.network.loss[ii].loss_tot.item() / self.mini_batch_size)
            self.train_loss[ii]['coord'].append(self.network.loss[ii].loss_coord.item() / self.mini_batch_size)
            self.train_loss[ii]['conf'].append(self.network.loss[ii].loss_conf.item() / self.mini_batch_size)
            if self.network.loss[ii].loss_cls is not None:
                self.train_loss[ii]['cls'].append(self.network.loss[ii].loss_cls.item() / self.mini_batch_size)

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        # print(f"batch#: {self.batch}")
        all_tot = 0.0
        all_coord = 0.0
        all_conf = 0.0
        all_cls = 0.0
        for ii in range(self.nloss):
            tot = mean(self.train_loss[ii]['tot'])
            coord = mean(self.train_loss[ii]['coord'])
            conf = mean(self.train_loss[ii]['conf'])
            all_tot += tot
            all_coord += coord
            all_conf += conf
            if self.classes > 1:
                cls = mean(self.train_loss[ii]['cls'])
                all_cls += cls

            if self.classes > 1:
                log.info(
                    f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)} Cls:{round(cls, 2)})')
            else:
                log.info(f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)})')

        if self.classes > 1:
            log.info(
                f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)} Cls:{round(all_cls, 2)})')
        else:
            log.info(
                f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)})')
        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'weights_{self.batch}.pt'))
            self.meta_network.save_weights(os.path.join(self.backup_dir, f'meta_weights_{self.batch}.pt'))

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            self.meta_network.save_weights(os.path.join(self.backup_dir, f'meta_backup.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            self.meta_network.save_weights(os.path.join(self.backup_dir, f'meta_backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            self.meta_network.save_weights(os.path.join(self.backup_dir, f'final.pt'))
            return True
        else:
            return False
