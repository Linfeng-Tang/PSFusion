import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    CLASSES = ('unlabelled', 'car', 'person', 'bike', 'curve', 'car_stop', 'guardrail', 'color_cone', 'bump')

    PALETTE = [[0, 0, 0], [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192],
               [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0]]
    # CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    #            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    #            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor')

    # PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    #            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    #            [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    #            [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
    #            [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        print(self.img_dir, self.split)
        assert osp.exists(self.img_dir) and self.split is not None
