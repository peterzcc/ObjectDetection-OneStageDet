import numpy as np
from utils.test.datasets.voc_eval import voc_eval
import os


def main():
    results_dir = "results/comp4_det_test_{}.txt"
    anno_path = "VOCdevkit/VOC2007/Annotations/{}.xml"

    # imageset_path = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"

    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    # i = 0
    # class_name = class_names[i]
    recs = []
    precs = []
    aps = []
    for class_name in class_names:
        imageset_path = f"VOCdevkit/VOC2007/ImageSets/Main/{class_name}_test.txt"
        cachedir = "VOCdevkit/onedet_cache"
        # cache_file = os.path.join(cachedir, "annots.pkl")
        # if os.path.exists(cache_file):
        #     os.remove(cache_file)
        rec, prec, ap = voc_eval(results_dir, anno_path, imageset_path, class_name, cachedir, use_07_metric=True)
        recs.append(rec)
        precs.append(prec)
        aps.append(ap)
    mAP = np.array(aps).mean()
    print(f"mAP:{mAP}")




if __name__ == '__main__':
    main()