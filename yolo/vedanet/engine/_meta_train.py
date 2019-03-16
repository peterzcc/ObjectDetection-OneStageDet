import logging as log
import torch
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as tf
from statistics import mean
import os
import random
from PIL import Image
import copy

from .. import data
from .. import models
from . import dual_engine
from .. import network

__all__ = ['MetaTrainingEngine']


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


class VOCMetaDataset(data.MetaboxDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        network_size = hyper_params.network_size
        labels = hyper_params.labels

        it = tf.ToTensor()

        lb = data.transform.Letterbox(hyper_params.meta_input_shape)

        self.meta_tf = data.transform.Compose([lb, it])
        self.meta_anno_tf = data.transform.Compose([lb])

        def identify(img_id):
            # return f'{root}/VOCdevkit/{img_id}.jpg'
            return f'{img_id}'

        super(VOCMetaDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, self.meta_tf, self.meta_anno_tf)

    def __getitem__(self, index):
        meta_imgs = []
        for i in range(len(self.class_label_map)):
            if len(self.classid_anno[i]) != 0:
                # get image
                randidx = random.randint(0, len(self.classid_anno[i]) - 1)
                class_img_name = self.classid_anno[i][randidx][0]
                class_img = Image.open(class_img_name)
                class_img_tf = self.meta_tf(class_img)          # [3, w, h]

                # get annotation
                class_anno = [copy.deepcopy(self.classid_anno[i][randidx][1])]
                class_anno = self.meta_anno_tf(class_anno)
                class_anno = class_anno[0]

                # add a mask
                x0 = int(class_anno.x_top_left)
                y0 = int(class_anno.y_top_left)
                x1 = int(class_anno.width + x0)
                y1 = int(class_anno.height + y0)
                mask = torch.zeros((1, class_img_tf.shape[1], class_img_tf.shape[2]))
                mask[0, y0:y1, x0:x1] = 1
                class_img = torch.cat((class_img_tf, mask), dim=0)
                meta_imgs.append(class_img.unsqueeze(0))
        meta_imgs = torch.cat(meta_imgs, dim=0)

        return meta_imgs


class MetaTrainingEngine(dual_engine.DualEngine):
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
        net = models.__dict__[model_name](hyper_params.classes, hyper_params.weights, train_flag=1,
                                          clear=hyper_params.clear)
        metanet = network.metanet.Metanet(hyper_params.classes)
        log.info('Net structure\n\n%s\n' % net)
        if self.cuda:

            net.cuda()
            metanet.cuda()
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = torch.nn.DataParallel(net)
                metanet = torch.nn.DataParallel(metanet)

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        log.info(f'Adjusting learning rate to [{learning_rate}]')
        if torch.cuda.device_count() > 1:
            meta_net_parameters = [
                {'params': metanet.module.parameters(), 'lr': learning_rate * 100.0 / batch},
                {'params': net.module.backbone.parameters()},
                {'params': net.module.head.parameters()}
            ]
        else:
            meta_net_parameters = [
                {'params': metanet.parameters(), 'lr': learning_rate * 100.0 / batch},
                {'params': net.backbone.parameters()},
                {'params': net.head.parameters()}
            ]
        optim = torch.optim.SGD(meta_net_parameters, lr=learning_rate / batch, momentum=momentum, dampening=0,
                                weight_decay=decay * batch)

        log.debug('Creating dataloader')
        dataset = VOCDataset(hyper_params)
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.mini_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=hyper_params.nworkers if self.cuda else 0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=data.list_collate,
        )

        meta_dataset = VOCMetaDataset(hyper_params)
        meta_dataloader = data.DataLoader(
            meta_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=hyper_params.pin_mem if self.cuda else False,
            collate_fn=default_collate,
        )

        super(MetaTrainingEngine, self).__init__(net, metanet, optim, dataloader, meta_dataloader)

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

        reweights = self.meta_network(meta_imgs)
        return reweights

    def process_batch(self, data):
        data, target, reweights = data
        # to(device)
        if self.cuda:
            data = data.cuda()
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

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)

    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            return True
        else:
            return False
