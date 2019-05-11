#Huang Daoji 11/05
#    [x] tested on jupyter nootbook
#    [x] tested on python
#    [x] run
#    [x] see result
#
# TODO: change eval.py, s.t. we do not need to generate .xml files?
#
# when testing on VOC classes
# - change dataloader, for some images in coco are gray image
# - delete images with no annotation
# - move .xml files to the correct place
#
# when testing on 80 classes, also
# - set use_coco_classes = 1
# - change cfgs/*.yml -> labels
# - delete VOCdevkit/annos.pkl if necessary



import json
import os
import sys
sys.path.insert(0, '.')
import brambox.boxes as bbb

import itertools
from xmltodict import unparse

# change to where COCO is
ROOT = "C:/Users/AndrewHuang/Documents/GitHub/ObjectDetection-OneStageDet/yolo/COCO"
# where to save the .pkl files
DST = "C:/Users/AndrewHuang/Documents/GitHub/ObjectDetection-OneStageDet/yolo/VOCdata"
# may change to 1, for there (seems to) be a slightly difference
# on bbox indexing between VOC and MSCOCO
BBOX_OFFSET = 0
# testing = 1 if testing
testing = 1

# tested on train/val/test 2017, the smallest file of MSCOCO
years = [
    "2017",
]

datasets = [
    "val",
]

use_coco_classes = 0

labels =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

labels_coco = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# map some classes in COCO to VOC
change_list = {'motorcycle': 'motorbike', 'airplane': 'aeroplane', 'tv': 'tvmonitor', 'couch': 'sofa', 'dining table': 'diningtable', 'potted plant': 'pottedplant'}

# helper function to construct VOC-like xml files for coco images
def base_dict(filename, width, height, depth=3):
    return {
        "annotation": {
            "filename": os.path.split(filename)[-1],
            "folder": "VOCtmp", "segmented": "0", "owner": {"name": "unknown"},
            "source": {'database': "The COCO 2017 database", 'annotation': "COCO 2017", "image": "unknown"},
            "size": {'width': width, 'height': height, "depth": depth},
            "object": []
        }
    }

def base_object(size_info, name, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h

    width = size_info['width']
    height = size_info['height']

    x1 = int(max(x1, 0) + BBOX_OFFSET)
    y1 = int(max(y1, 0) + BBOX_OFFSET)
    x2 = int(min(x2, width - 1) + BBOX_OFFSET)
    y2 = int(min(y2, height - 1) + BBOX_OFFSET)

    return {
        'name': name, 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0',
        'bndbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
    }
# end of helper functions

# get xml files for calculating mAP
def get_xml_for_test_set(year, dataset):
    json_path = os.path.join(ROOT, f"annotations/instances_{dataset}{year}.json")
    data = json.load(open(json_path))
    cate = {x['id']: x['name'] for x in data['categories']}

    images = {}
    for im in data["images"]:
        img = base_dict(im['coco_url'], im['width'], im['height'], 3)
        images[im["id"]] = img

    for an in data["annotations"]:
        ann = base_object(images[an['image_id']]['annotation']["size"], cate[an['category_id']], an['bbox'])
        images[an['image_id']]['annotation']['object'].append(ann)

    # change labels form coco to voc
    for img in images:
        new_objs = []
        for obj in images[img]['annotation']['object']:
            if use_coco_classes:
                if obj['name'] in labels_coco:
                    new_objs.append(obj)
            else:
                if obj['name'] in change_list.keys():
                    obj['name'] = change_list[obj['name']]
                if obj['name'] in labels:
                    new_objs.append(obj)
        images[img]['annotation']['object'] = new_objs
    # delete images with no labels, otherwise will raise error
    del_keys = []
    for image in images:
        if images[image]['annotation']['object'] == []:
            #print(image)
            del_keys.append(image)
    for key in del_keys:
        del images[key]

    print(len(images))

    # write xml files
    dst_dirs = {x: os.path.join(f'{ROOT}/VOCtmp', x) for x in ["Annotations", "ImageSets", "JPEGImages"]}
    dst_dirs['ImageSets'] = os.path.join(dst_dirs['ImageSets'], "Main")
    for k, d in dst_dirs.items():
        os.makedirs(d, exist_ok=True)

    for k, im in images.items():
        im['annotation']['object'] = im['annotation']['object'] or [None]
        unparse(im,
                open(os.path.join(dst_dirs["Annotations"], "{}.xml".format(str(k).zfill(12))), "w"),
                full_document=False, pretty=True)

    #print("Write image sets")
    with open(os.path.join(dst_dirs["ImageSets"], "{}.txt".format(dataset)), "w") as f:
        f.writelines(list(map(lambda x: str(x).zfill(12) + "\n", images.keys())))

def get_label_for_single_file(year, dataset):

    data = json.load(open(f"{ROOT}/annotations/instances_{dataset}{year}.json"))
    cate = {x['id']: x['name'] for x in data['categories']}

    # hold all images
    images = {}
    for image in data["images"]:
        images[image["id"]] = {
            # add more attributes if needed
            "file_name" : image["file_name"],
            "image_size" : {
                "height": image["height"],
                "width": image["width"],
            },
            "img_id" : image["id"],
            "obj" : []
        }

    for anno in data["annotations"]:
        # attach annotations to corresponding images
        images[anno["image_id"]]["obj"].append({
            "class_label": cate[anno["category_id"]],
            "bbox": anno["bbox"]
        })

    # delete images with no annotations
    del_keys = []
    for image in images:
        if images[image]["obj"] == []:
            del_keys.append(image)
    for key in del_keys:
        del images[key]

    val_annos = {}
    for image in images:
        val_annos[f'{ROOT}/{dataset}{year}/{images[image]["file_name"]}'] = []
        for obj in images[image]["obj"]:
            tmp_obj = bbb.annotations.PascalVocAnnotation()
            tmp_obj.class_label = obj["class_label"]
            tmp_obj.x_top_left = int(max(obj["bbox"][0], 0))
            tmp_obj.y_top_left = int(max(obj["bbox"][1], 0))
            # maybe out of boundry! need to check?
            tmp_obj.width = int(obj["bbox"][2])
            tmp_obj.height = int(obj["bbox"][3])
            val_annos[f'{ROOT}/{dataset}{year}/{images[image]["file_name"]}'].append(tmp_obj)
    # this one contains categories not in VOC
    del_keys = []
    for image in val_annos:
        if val_annos[image] == []:
            del_keys.append(image)
    for key in del_keys:
        del val_annos[key]
    if use_coco_classes:
        print(len(val_annos))
        bbb.generate('anno_pickle', val_annos, f'{DST}/onedet_cache/MSCOCO{dataset}{year}.pkl')

    val_annos_fix_label = {}
    for image in val_annos:
        val_annos_fix_label[image] = []
        for anno in val_annos[image]:
            # map some coco label to voc labels
            if anno.class_label in change_list.keys():
                anno.class_label = change_list[anno.class_label]
            if anno.class_label in labels:
                val_annos_fix_label[image].append(anno)
    del_keys = []
    for image in val_annos_fix_label:
        if val_annos_fix_label[image] == []:
            del_keys.append(image)
    for key in del_keys:
        del val_annos_fix_label[key]
    if not use_coco_classes:
        print(len(val_annos_fix_label))
        bbb.generate('anno_pickle', val_annos_fix_label, f'{DST}/onedet_cache/MSCOCO{dataset}{year}_fix_label.pkl')

    # generate test file for each label
    if use_coco_classes:
        val_annos_fix_label = val_annos
        labels_here = labels_coco
    else:
        labels_here = labels
    if testing:
        main = {}
        for label in labels_here:
            main[label] = []
        for img in val_annos_fix_label:
            has_obj = {}
            #print(img)
            for obj in val_annos_fix_label[img]:
                #print(obj)
                has_obj[obj.class_label] = 1
            #print(has_obj)
            for label in labels_here:
                if label in has_obj:
                    main[label].append([img.split('/')[-1].split('.')[0], 1])
                else:
                    main[label].append([img.split('/')[-1].split('.')[0], -1])
        for label in labels_here:
            with open(f'{ROOT}/VOCtmp/ImageSets/Main/{label}_test.txt', 'w') as f:
                for case in main[label]:
                    f.write(case[0] + ' ' + str(case[1]) + '\n')
                    #break;

def get_label_for_all_files():
    for year in years:
        for dataset in datasets:
            get_label_for_single_file(year, dataset)
            if testing:
                get_xml_for_test_set(year, dataset)

if __name__ == '__main__':
    get_label_for_all_files()
