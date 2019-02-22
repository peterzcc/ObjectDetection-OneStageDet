import numpy as np
from utils.test.datasets.voc_eval import voc_eval
import os
import argparse


def generate_aps(results_root="results"):
    results_dir = results_root+"/comp4_det_test_{}.txt"
    output_path = results_root+"/perf.csv"
    anno_path = "VOCdevkit/VOC2007/Annotations/{}.xml"
    # imageset_path = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
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
    header = ",".join([*class_names, "mean"])+'\n'
    data = ",".join([f"{ap}"for ap in aps]+ [f"{mAP}"])+'\n'
    with open(output_path, "w") as f:
        f.write(header)
        f.write(data)
    print(header+data)

def generate_aps_onebox(results_root="results",file_names=None):
    results_dir = results_root+"/comp4_det_test_{}.txt"
    output_path = results_root+"/perf.csv"
    anno_path = "VOCdevkit/VOC2007/Annotations/{}.xml"
    # imageset_path = "VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
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
    header = ",".join([*class_names, "mean"])+'\n'
    data = ",".join([f"{ap}"for ap in aps]+ [f"{mAP}"])+'\n'
    with open(output_path, "w") as f:
        f.write(header)
        f.write(data)
    print(header+data)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mcnn worldexp.')
    parser.add_argument('--root', type=str, default="results")
    args = parser.parse_args()
    generate_aps(args.root)