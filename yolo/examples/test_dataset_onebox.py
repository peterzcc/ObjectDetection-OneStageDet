#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#

import os
import copy
import logging as log
from PIL import Image
import random

import brambox.boxes as bbb
from vedanet.data._dataloading import Dataset
from vedanet import data
from torchvision import transforms as tf
from vedanet.data import OneboxDataset
from unittest import TestCase
import numpy as np


def get_dataset():
    anno = 'VOCdevkit/onedet_cache/train.pkl'
    network_size = (608, 608)
    labels = ['aeroplane',
              'bicycle',
              'bird',
              'boat',
              'bottle',
              'bus',
              'car',
              'cat',
              'chair',
              'cow',
              'diningtable',
              'dog',
              'horse',
              'motorbike',
              'person',
              'pottedplant',
              'sheep',
              'sofa',
              'train',
              'tvmonitor']

    def identify(img_id):
        # return f'{root}/VOCdevkit/{img_id}.jpg'
        return f'{img_id}'

    flip = 0.5
    jitter = 0.3
    hue, sat, val = 0.1, 1.5, 1.5

    rf = data.transform.RandomFlip(flip)
    rc = data.transform.RandomCropLetterbox(input_dim=network_size[0:2], jitter=jitter)
    hsv = data.transform.HSVShift(hue, sat, val)
    it = tf.ToTensor()

    img_tf = data.transform.Compose([rc, rf, hsv, it])
    anno_tf = data.transform.Compose([rc, rf])
    return OneboxDataset('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)


class OneBoxTests(TestCase):
    def setUp(self):
        self.dataset = get_dataset()

    def test_total_count(self):
        self.assertEqual(len(self.dataset), 40058)

    def test_sel_class(self):
        target_classes = {"person", "cat"}
        subset = self.dataset.sel_classes(target_classes)
        for i in range(10, 20):
            this_class = subset.class_label_map[subset[0][1][0].class_id]
            self.assertTrue(this_class in target_classes)

    def test_rgbd(self):
        for i in range(40,50):
            x, y = self.dataset[i]
            display = False
            if display:
                import torch
                import matplotlib.pyplot as plt
                plt.imshow(torch.mean(x[0:3], dim=0)*(x[3,...]+0.5)/1.5,cmap="gray")
                plt.show()
            self.assertEqual(x.shape[0], 4)


