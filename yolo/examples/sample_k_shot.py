import brambox.boxes as bbb
from collections import OrderedDict, defaultdict
import numpy as np


def convert_to_ordered(annos):
    return OrderedDict([(k,v) for k,v in sorted(list(annos.items()),key=lambda x:x[0])])


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

    novel_classes =['bird', 'bus', 'cow', 'motorbike', 'sofa'] # ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa"] #
    novel_classes_set = set(novel_classes)
    k_shot_classes = class_label_map.copy()
    class2id = {k_shot_classes[i]: i for i in range(len(k_shot_classes))}
    class2file2numdet = {cls: defaultdict(lambda: 0) for cls in k_shot_classes}
    for file, list_annos in annos.items():
        for anno in list_annos:
            if anno.class_label in class2id:
                class2file2numdet[anno.class_label][file] += 1

    k_shot = 3
    existing_sample_nums = np.zeros(len(k_shot_classes))

    current_class_id = 0
    k_shot_samples = set()
    novel_class_samples = set()
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
        contain_novel_class = False
        for anno in annos[sample_file_name]:
            if anno.class_label in k_shot_classes:
                pred_sample_nums[class2id[anno.class_label]] += 1
            if anno.class_label in novel_classes_set:
                contain_novel_class = True
        if pred_sample_nums.max() <= k_shot:
            k_shot_samples.add(sample_file_name)
            existing_sample_nums = pred_sample_nums
            if contain_novel_class:
                novel_class_samples.add(sample_file_name)
            print(f"file: {sample_file_name}")
            for anno in annos[sample_file_name]:
                print('   {}'.format(anno.class_label))
    assert not np.any(existing_sample_nums - k_shot)
    k_shot_annos = {fn: annos[fn] for fn in k_shot_samples}
    novel_class_annos = {f: annos[f] for f in novel_class_samples}
    novel_class_files = {file for cls in novel_classes for file in class2file2numdet[cls].keys()}
    files2remove = novel_class_files - novel_class_samples
    joint_anno_files = set(annos.keys()) - files2remove

    joint_annos = {k: annos[k] for k in joint_anno_files}
    base_annos = {k: annos[k] for k in set(annos.keys())-novel_class_files}
    print(f"random seed: {random_seed}")
    print(f"total file number: {len(annos.keys())}")
    print(f"sampled {len(k_shot_annos)} files in total")
    print(f"removed {len(files2remove)} files in total")
    print(f"got {len(base_annos.keys())} base files in total")
    bbb.generate('anno_pickle', novel_class_annos, "../VOCdevkit/onedet_cache/k_shot_{}_finetune.pkl".format(k_shot))
    bbb.generate('anno_pickle', joint_annos, "../VOCdevkit/onedet_cache/k_shot_{}_joint.pkl".format(k_shot))
    bbb.generate('anno_pickle', k_shot_annos, "../VOCdevkit/onedet_cache/k_shot_{}_allclass.pkl".format(k_shot))
    bbb.generate('anno_pickle', base_annos, "../VOCdevkit/onedet_cache/base.pkl")

    pass


if __name__ == '__main__':
    main()
