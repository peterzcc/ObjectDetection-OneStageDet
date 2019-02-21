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


class TestOneboxDataset(OneboxDataset):
    def __init__(self):
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
        rc = data.transform.RandomCropLetterbox(self, jitter)
        hsv = data.transform.HSVShift(hue, sat, val)
        it = tf.ToTensor()

        img_tf = data.transform.Compose([rc, rf, hsv, it])
        anno_tf = data.transform.Compose([rc, rf])
        super(TestOneboxDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)


def test_dataset():
    dataset = TestOneboxDataset()
    pass



if __name__ == '__main__':
    test_dataset()
