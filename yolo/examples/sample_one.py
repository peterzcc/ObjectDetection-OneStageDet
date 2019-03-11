import sys
sys.path.append('/home/data/urop2018/sfuab/ObjectDetection-OneStageDet/yolo')

import brambox.boxes as bbb
from collections import OrderedDict
import numpy as np


def main():
    random_seed = 123
    rng = np.random.RandomState(random_seed)
    anno_format = "anno_pickle"
    class_label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    identify = lambda f: f
    anno_filename = '../VOCdevkit/onedet_cache/train.pkl'
    kwargs = {}
    annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
    keys = list(annos)
    k_shot_classes = ['bird', 'bus', 'cow', 'motorbike', 'sofa']

    class2id = {k_shot_classes[i]: i for i in range(len(k_shot_classes))}
    class2file2numdet = {cls: OrderedDict() for cls in k_shot_classes}
    base_anno = {}
    index = 0
    for file, list_annos in annos.items():
        base_list = []
        for anno in list_annos:
            if anno.class_label in class2id:
                try:
                    class2file2numdet[anno.class_label][file] += 1
                except KeyError:
                    class2file2numdet[anno.class_label][file] = 1
            else:
                base_list.append(anno)
        if len(base_list) > 0:
            for i in range(100):
                base_list.append(anno)
            base_anno[file] = base_list
        if index > 10:
            break
        index += 1

    print(f"got {len(base_anno.keys())} base files in total")
    bbb.generate('anno_pickle', base_anno, "../VOCdevkit/onedet_cache/one_sample.pkl")
    pass


if __name__ == '__main__':
    main()