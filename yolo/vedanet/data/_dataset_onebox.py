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
    def __init__(self, anno_format, anno_filename,
                 input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None,
                 file_boxes=None, **kwargs):
        super().__init__(input_dimension)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name: os.path.splitext(name)[0] + '.png'
        self.class_label_map = class_label_map
        if file_boxes is None:
            # Get annotations
            annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
            # self.keys = list(self.annos)
            self.boxes = []
            self.files = []

            # Add class_ids
            if class_label_map is None:
                log.warn(f'No class_label_map given, annotations wont have a class_id values for eg. loss function')
            for k, anno in annos.items():
                for a in anno:
                    if class_label_map is not None:
                        try:
                            a.class_id = class_label_map.index(a.class_label)
                        except ValueError as err:
                            raise ValueError(f'{a.class_label} is not found in the class_label_map') from err
                    else:
                        a.class_id = 0
                    self.boxes.append([a])
                    self.files.append(k)

            log.info(f'Dataset loaded: {len(self.boxes)} boxes')
        else:
            self.files, self.boxes = file_boxes

    def sel_classes(self, classes):
        new_files, new_boxes = [], []
        for file, this_anno in zip(self.files, self.boxes):
            if self.class_label_map[this_anno[0].class_id] in classes:
                new_files.append(file)
                new_boxes.append(this_anno)
        return OneboxDataset(None, None, self.input_dim,
                             self.class_label_map, self.id,
                             self.img_tf, self.anno_tf, (new_files, new_boxes))

    def __len__(self):
        return len(self.boxes) #len(self.keys)

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """ Get transformed image and annotations based of the index of ``self.keys``

        Args:
            index (int): index of the ``self.keys`` list containing all the image identifiers of the dataset.

        Returns:
            tuple: (transformed image, list of transformed brambox boxes)
        """
        if index >= len(self):
            raise IndexError(f'list index out of range [{index}/{len(self)-1}]')

        # Load
        img = Image.open(self.id(self.files[index]))
        anno = copy.deepcopy(self.boxes[index])
        random.shuffle(anno)

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
        return img_rgbm, anno


