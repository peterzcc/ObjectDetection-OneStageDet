#
#   Lightnet dataset that works with brambox annotations
#   Copyright EAVISE
#

import os
import copy
import logging as log
from PIL import Image
import random
import torch

import brambox.boxes as bbb
from ._dataloading import Dataset

__all__ = ['MetaboxDataset']


class MetaboxDataset(Dataset):
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
    def __init__(self, anno_format, anno_filename, input_dimension, class_label_map=None, identify=None, img_transform=None, anno_transform=None, **kwargs):
        super().__init__(input_dimension)
        self.img_tf = img_transform
        self.anno_tf = anno_transform
        if callable(identify):
            self.id = identify
        else:
            self.id = lambda name: os.path.splitext(name)[0] + '.png'

        # Get annotations
        self.annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
        self.keys = list(self.annos)

        self.classid_anno = {i: [] for i in range(len(class_label_map))}

        # Add class_ids
        if class_label_map is None:
            log.warn(f'No class_label_map given, annotations wont have a class_id values for eg. loss function')
        for k, annos in self.annos.items():
            for a in annos:
                if class_label_map is not None:
                    try:
                        a.class_id = class_label_map.index(a.class_label)
                    except ValueError as err:
                        raise ValueError(f'{a.class_label} is not found in the class_label_map') from err
                else:
                    a.class_id = 0
                self.classid_anno[a.class_id].append((k, a))

        log.info(f'Dataset loaded: {len(self.keys)} images')

    def __len__(self):
        return len(self.keys)

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
        img = Image.open(self.id(self.keys[index]))
        anno = copy.deepcopy(self.annos[self.keys[index]])
        random.shuffle(anno)

        # Transform
        if self.img_tf is not None:
            img = self.img_tf(img)
        if self.anno_tf is not None:
            anno = self.anno_tf(anno)

        load_img = self.id(self.keys[index])
        meta_imgs = []
        for i in self.classid_anno:
            if len(self.classid_anno[i]) != 0:
                # get image
                randidx = random.randint(0, len(self.classid_anno[i]) - 1)
                class_img_name = self.classid_anno[i][randidx][0]
                while class_img_name == load_img:
                    randidx = random.randint(0, len(self.classid_anno[i]) - 1)
                    class_img_name = self.classid_anno[i][randidx][0]
                class_img = Image.open(class_img_name)
                class_img_tf = self.img_tf(class_img)          # [3, w, h]

                # get annotation
                class_anno = [copy.deepcopy(self.classid_anno[i][randidx][1])]
                class_anno = self.anno_tf(class_anno)
                while(len(class_anno) == 0):
                    class_img_tf = self.img_tf(class_img)
                    class_anno = [copy.deepcopy(self.classid_anno[i][randidx][1])]
                    class_anno = self.anno_tf(class_anno)
                class_anno = class_anno[0]

                # add a mask
                x0 = int(class_anno.x_top_left)
                y0 = int(class_anno.y_top_left)
                x1 = int(class_anno.width + x0)
                y1 = int(class_anno.height + y0)
                mask = torch.zeros((1, img.shape[1], img.shape[2]))
                mask[0, y0:y1, x0:x1] = 1
                class_img = torch.cat((class_img_tf, mask), dim=0)
                meta_imgs.append(class_img.unsqueeze(0))
        meta_imgs = torch.cat(meta_imgs, dim=0)
        return img, anno, meta_imgs
