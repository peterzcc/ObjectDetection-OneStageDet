from vedanet.models.network import metanet





if __name__ == '__main__':
    classes = ['aeroplane', 'bicycle', 'boat', 'bottle', 'car', 'cat', 'chair', 'diningtable', 'dog', 'horse', 'person', 'pottedplant', 'sheep', 'train', 'tvmonitor']
    net_meta = metanet.Metanet(num_classes=len(classes))

