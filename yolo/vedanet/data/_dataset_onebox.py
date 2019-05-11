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
from ._dataloading import Dataset
from .. import data
from torchvision import transforms as tf
import numpy as np
import torch
from collections import OrderedDict, defaultdict
__all__ = ['OneboxDataset']


class OneboxDataset(Dataset):
    """ Dataset for any brambox parsable annotation format.

    Args:
        anno_format (brambox.boxes.formats): Annotation format
        anno_filename (list or str): Annotation filename, list of filenames or expandable sequence
        input_dimension (tuple): (width,height) tuple with default dimensions of the network
        class_label_map (list): List of class_labels
        identify (function, optional): Lambda/function to get image based of annotation filename or image id; Default **replace/add .png extension to filename/id**
        img_transform (torchvision.transforms.Compose): Transforms to perform on the images
        anno_transform (torchvision.transforms.Compose): Transforms to perform on the annotations
        kwargs (dict): Keyword arguments that are passed to the brambox parser
    """
    def __init__(self, annos,
                 input_dimension, class_list=None, identify=None, img_transform=None, anno_transform=None,
                 **kwargs):
        super().__init__(input_dimension)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name: os.path.splitext(name)[0] + '.png'
        self.label_class_map = class_list
        self.class_label_map = {v: i for i,v in enumerate(self.label_class_map)}
        # Get annotations
        self.annos = annos
        # annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_list, **kwargs)
        # self.keys = list(self.annos)
        self.file_box = []
        self.cls_2_boxid = [list() for _ in range(len(self.label_class_map))]
        self.fileid_2_boxid = [list() for _ in range(len(self.annos))]
        self.boxid_2_fileid = []
        # Add class_ids
        if class_list is None:
            log.warn('No class_label_map given, annotations wont have a class_id values for eg. loss function')
        for file_id, (k, anno) in enumerate(annos.items()):
            for a in anno:
                if class_list is not None:
                    try:
                        a.class_id = self.class_label_map[a.class_label] #class_list.index(a.class_label)
                    except KeyError as err:
                        raise ValueError('{} is not found in the class_label_map'.format(a.class_label)) from err
                else:
                    a.class_id = 0
                self.file_box.append((k, [a]))

                box_id = len(self.file_box) -1
                self.fileid_2_boxid[file_id].append(box_id)
                self.cls_2_boxid[a.class_id].append(box_id)
                self.boxid_2_fileid.append(file_id)

        log.info('Dataset loaded: {} boxes'.format(len(self.file_box)))

    def __len__(self):
        return len(self.file_box) #len(self.keys)

    def get_anno(self, index):
        anno = copy.deepcopy(self.boxes[index])
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)
        return anno

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the ``self.keys`` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError('list index out of range [{}/{}]'.format(index, len(self)-1))

        # Load
        file, box = self.file_box[index]
        img = Image.open(self.id(file))
        anno = copy.deepcopy(box)
        random.shuffle(anno)

        #Huang Daoji 11/05
        # convert gray image to RGB
        img = img.convert("RGB")

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)
        mask = np.zeros((1, *img.shape[1:3]), dtype=np.float32)
        for a in anno:
            x,y,w,h = a.x_top_left, a.y_top_left, a.width, a.height
            x0 = np.clip(round(x),0,self.input_dim[1])
            x1 = np.clip(round(x+w),0, self.input_dim[1])
            y0 = np.clip(round(y),0,self.input_dim[0])
            y1 = np.clip(round(y+h),0,self.input_dim[0])
            try:
                mask[0, y0:y1, x0:x1] = 1.
            except TypeError:
                print("x, y, w, h not int error")
        torch_mask = torch.Tensor(mask)
        img_rgbm = torch.cat([img, torch_mask], dim=0)
        return img_rgbm, #anno
