import torch
import numpy as np

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""
class SegmentationMetric(object):
    def __init__(self, numClass, device):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2).to(device)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        # acc = acc.item()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        # classAcc  = classAcc.item()
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean() # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        # meanAcc = meanAcc.item()
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # IoU = [a.item() for a in IoU] 
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()# 求各类别IoU的平均
        # mIoU = mIoU.item()
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU, 频权交并比:为MIoU的一种提升, 这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                torch.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        # FWIoU = FWIoU.item()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        with torch.no_grad():
            self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))
        
def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class