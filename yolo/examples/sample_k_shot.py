import brambox.boxes as bbb
from collections import OrderedDict
import numpy as np


def convert_to_ordered(annos):
    return OrderedDict([(k,v) for k,v in sorted(list(annos.items()),key=lambda x:x[0])])


def main():
    random_seed = 123
    rng = np.random.RandomState(random_seed)
    anno_format = "anno_pickle"
    class_label_map = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    identify = lambda f: f
    anno_filename = 'VOCdevkit/onedet_cache/train.pkl'
    kwargs = {}
    annos = bbb.parse(anno_format, anno_filename, identify=lambda f: f, class_label_map=class_label_map, **kwargs)
    keys = list(annos)

    k_shot_classes =['bird', 'bus', 'cow', 'motorbike', 'sofa'] # ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa"] #

    class2id = {k_shot_classes[i]: i for i in range(len(k_shot_classes))}
    class2file2numdet = {cls: OrderedDict() for cls in k_shot_classes}
    base_anno = {}
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
            base_anno[file] = base_list

    k_shot = 3
    existing_sample_nums = np.zeros(len(k_shot_classes))

    current_class_id = 0
    k_shot_samples = set()
    finished_sampling = False

    while existing_sample_nums.min() < k_shot:
        assert existing_sample_nums[current_class_id] <= k_shot
        if existing_sample_nums[current_class_id] == k_shot:
            current_class_id = min(current_class_id+1, len(k_shot_classes))
            continue
        class_label = k_shot_classes[current_class_id]
        file2numdet = class2file2numdet[class_label]
        sample_file_id = rng.randint(len(file2numdet))
        sample_file_name = list(file2numdet.items())[sample_file_id][0]
        if sample_file_name in k_shot_samples:
            continue
        pred_sample_nums = existing_sample_nums.copy()
        for anno in annos[sample_file_name]:
            if anno.class_label in k_shot_classes:
                pred_sample_nums[class2id[anno.class_label]] += 1
        if pred_sample_nums.max() <= k_shot:
            k_shot_samples.add(sample_file_name)
            existing_sample_nums = pred_sample_nums
            print('class_label is {}'.format(class_label))
    assert not np.any(existing_sample_nums - k_shot)
    k_shot_annos = {fn: annos[fn] for fn in k_shot_samples}
    full_k_shot_class_files = {file for cls in k_shot_classes for file in class2file2numdet[cls].keys()}
    files2remove = full_k_shot_class_files - k_shot_samples
    joint_anno_files = set(annos.keys()) - files2remove

    joint_annos = {k:annos[k] for k in joint_anno_files}
    print(f"random seed: {random_seed}")
    print(f"total file number: {len(annos.keys())}")
    print(f"sampled {len(k_shot_annos)} files in total")
    print(f"removed {len(files2remove)} files in total")
    print(f"got {len(base_anno.keys())} base files in total")
    bbb.generate('anno_pickle', joint_annos, "../VOCdevkit/onedet_cache/k_shot_{}_joint.pkl".format(k_shot))
    bbb.generate('anno_pickle', k_shot_annos, "../VOCdevkit/onedet_cache/k_shot_{}_finetune.pkl".format(k_shot))

    # bbb.generate('anno_pickle', base_anno, "../VOCdevkit/onedet_cache/k_shot_base.pkl")
    pass


if __name__ == '__main__':
    main()