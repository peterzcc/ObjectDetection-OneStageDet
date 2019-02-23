# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import _pickle as cPickle
import numpy as np
from vedanet.data import OneboxDataset


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_overlap(BBGT,bb_det):
    ixmin = np.maximum(BBGT[:, 0], bb_det[0])
    iymin = np.maximum(BBGT[:, 1], bb_det[1])
    ixmax = np.minimum(BBGT[:, 2], bb_det[2])
    iymax = np.minimum(BBGT[:, 3], bb_det[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb_det[2] - bb_det[0] + 1.) * (bb_det[3] - bb_det[1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax,jmax


def load_gt(annopath, imagesetfile, cachedir, classname):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.split(" ")[0].strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        anno_data = {}
        for i, imagename in enumerate(imagenames):
            anno_data[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            cPickle.dump(anno_data, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            anno_data = cPickle.load(f)

    # extract gt objects for this class
    class_anno_gt = {}
    npos = 0
    for imagename in imagenames:
        R_gt = [obj for obj in anno_data[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R_gt])
        difficult = np.array([x['difficult'] for x in R_gt]).astype(np.bool)
        det = [False] * len(R_gt)
        npos = npos + sum(~difficult)
        class_anno_gt[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}
    return class_anno_gt, npos


def anno_to_bbox(iv):
    xmin = iv.x_top_left
    ymin = iv.y_top_left
    xmax = xmin + iv.width
    ymax = ymin + iv.height

    return (xmin, ymin, xmax, ymax)


def load_gt_from_dataset(dataset: OneboxDataset,classname):
    target_class_id = dataset.class_label_map.index(classname)
    class_anno_gt = {}
    npos = 0
    for imgid in range(len(dataset)):
        this_anno = dataset.get_anno(imgid)
        as_gt = [obj for obj in this_anno if obj.class_id == target_class_id]
        bbox = np.array([anno_to_bbox(x) for x in as_gt])
        difficult = np.array([x.difficult for x in as_gt]).astype(np.bool)
        npos = npos + sum(~difficult)
        det = [False] * len(as_gt)
        class_anno_gt[f"{imgid}"] = {'bbox': bbox,
                                'difficult': difficult,
                                'det': det}
    return class_anno_gt, npos


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    class_recs_gt, npos = load_gt(annopath, imagesetfile, cachedir, classname)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines_det = [x.strip().split(' ') for x in lines]
    image_ids_det = [x[0] for x in splitlines_det]
    confidence = np.array([float(x[1]) for x in splitlines_det])
    BB_det = np.array([[float(z) for z in x[2:]] for x in splitlines_det])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB_det = BB_det[sorted_ind, :]
    image_ids_det = [image_ids_det[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs

    nd = len(image_ids_det)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        gt_class_this_img = class_recs_gt[image_ids_det[d]]
        det_boxes = BB_det[d, :].astype(float)
        ovmax = -np.inf
        gt_boxes = gt_class_this_img['bbox'].astype(float)

        if gt_boxes.size > 0:
            # compute overlaps
            # intersection
            ovmax, jmax = get_overlap(gt_boxes, det_boxes)

        if ovmax > ovthresh:
            if not gt_class_this_img['difficult'][jmax]:
                if not gt_class_this_img['det'][jmax]:
                    tp[d] = 1.
                    gt_class_this_img['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval_onebox(detpath,
                    test_dataset,
                    classname,
                    ovthresh=0.5,
                    use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    class_recs_gt, npos = load_gt_from_dataset(test_dataset, classname)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines_det = [x.strip().split(' ') for x in lines]
    image_ids_det = [x[0] for x in splitlines_det]
    confidence = np.array([float(x[1]) for x in splitlines_det])
    BB_det = np.array([[float(z) for z in x[2:]] for x in splitlines_det])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB_det = BB_det[sorted_ind, :]
    image_ids_det = [image_ids_det[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs

    nd = len(image_ids_det)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        gt_class_this_img = class_recs_gt[image_ids_det[d]]
        det_boxes = BB_det[d, :].astype(float)
        ovmax = -np.inf
        gt_boxes = gt_class_this_img['bbox'].astype(float)

        if gt_boxes.size > 0:
            # compute overlaps
            # intersection
            ovmax, jmax = get_overlap(gt_boxes, det_boxes)

        if ovmax > ovthresh:
            if not gt_class_this_img['difficult'][jmax]:
                if not gt_class_this_img['det'][jmax]:
                    tp[d] = 1.
                    gt_class_this_img['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap