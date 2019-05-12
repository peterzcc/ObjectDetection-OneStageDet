# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
try:
    from ..nms.gpu_nms import gpu_nms
except ImportError:
    pass

#Huang Daoji 11/05
# just in case cpu_nms is not found
try:
    from ..nms.cpu_nms import cpu_nms, cpu_soft_nms
except ImportError:
    from ..nms.py_cpu_nms import py_cpu_nms
import numpy as np

def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep


# Original NMS implementation
def nms(dets, thresh, force_cpu=False, gpu_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if gpu_id is not None and not force_cpu:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        #Huang Daoji 05/12
        # umm..I did not compile this...
        # use python instead
        return py_cpu_nms(dets, thresh)
