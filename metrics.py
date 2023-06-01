# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None or ignore_index < 0 or ignore_index > n_classes:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()
        cls_acc = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(cls_acc)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))
        cls_acc = dict(zip(range(self.n_classes), cls_acc))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },
            cls_iu,
            cls_acc,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count