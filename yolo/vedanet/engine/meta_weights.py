import logging as log
import torch
import os
import pickle

from torchvision import transforms as tf
from .. import data as vn_data
import numpy as np
from ..network import metanet


__all__ = ['MetaWeights']
class CustomDataset(vn_data.WeightDataset):
    def __init__(self, hyper_params):
        anno = hyper_params.trainfile
        root = hyper_params.data_root
        network_size = hyper_params.network_size
        labels = hyper_params.labels


        lb  = vn_data.transform.Letterbox(network_size)
        it  = tf.ToTensor()
        img_tf = vn_data.transform.Compose([lb, it])
        anno_tf = vn_data.transform.Compose([lb])

        def identify(img_id):
            return f'{img_id}'

        super(CustomDataset, self).__init__('anno_pickle', anno, network_size, labels, identify, img_tf, anno_tf)

    def __getitem__(self, index):
        img, anno = super(CustomDataset, self).__getitem__(index)
        for a in anno:
            a.ignore = a.difficult  # Mark difficult annotations as ignore for pr metric
        return img, anno


def MetaWeights(hyper_params):
    log.debug('Creating network')

    model_name = hyper_params.model_name
    batch = hyper_params.batch
    use_cuda = hyper_params.cuda
    weights = hyper_params.weights
    nworkers = hyper_params.nworkers
    pin_mem = hyper_params.pin_mem
    classes = hyper_params.classes
    labels = hyper_params.labels
    results = hyper_params.results
    sample = hyper_params.sample
    sample_seed = hyper_params.sampleseed
    rng = np.random.RandomState(sample_seed)

    print(model_name)
    net = metanet.Metanet(num_classes=classes, weights_file=weights,
                          use_dummy_reweight=hyper_params.use_dummy_reweight)
    net.eval()
    log.info('Net structure\n%s' % net)

    if use_cuda:
        net.cuda()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        CustomDataset(hyper_params),
        batch_size = batch,
        shuffle = False,
        drop_last = False,
        num_workers = nworkers if use_cuda else 0,
        pin_memory = pin_mem if use_cuda else False,
        collate_fn = vn_data.list_collate,
    )

    log.debug('Running meta network')

    class_weight = {i: [] for i in range(classes)}

    for idx, (data, annos) in enumerate(loader):
        if (idx + 1) % 100 == 0:
            log.info('%d/%d' % (idx + 1, len(loader)))
        if use_cuda:
            data = data.cuda()
        assert data.shape[0] == 1
        data = data[0]

        with torch.no_grad():
            reweights = net(data)
            cur_idx = 0
            for anno in annos:      # batch
                for a in anno:
                    class_id = labels.index(a.class_label)
                    if not hyper_params.use_dummy_reweight:
                        class_weight[class_id].append(reweights[cur_idx])
                    else:
                        class_weight[class_id].append(reweights[0, class_id])
                    cur_idx += 1

    for i in class_weight:
        if sample is None:
            class_weight[i] = sum(class_weight[i]) / len(class_weight[i])
        else:
            n_boxes = len(class_weight[i])
            assert sample <= n_boxes
            sample_ids = rng.choice(n_boxes,sample)
            class_weight[i] = sum([class_weight[i][s] for s in sample_ids]) / sample

        print('weight for class {} is {}'.format(labels[i], torch.norm(class_weight[i].view(-1), p=2)))

    if not os.path.isdir(results) and not results.endswith('.pkl'):
        os.mkdir(results)
        results = os.path.join(results, 'weights.pkl')

    with open(results, 'wb') as handle:
        pickle.dump(class_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ''' to load weights
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)
    '''




