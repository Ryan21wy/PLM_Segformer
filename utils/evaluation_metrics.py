"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / (self.confusionMatrix.sum() + 1e-6)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-6)
        classAcc[np.isnan(classAcc)] = 0
        return np.round(classAcc, 3)  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def classRecall(self):
        # recall = (TP) / TP + FN
        classRecall = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-6)
        classRecall[np.isnan(classRecall)] = 0
        return np.round(classRecall, 3)

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return np.round(meanAcc, 3)  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanRecall(self):
        classRecall = self.classRecall()
        meanRecall = np.nanmean(classRecall)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return np.round(meanRecall, 3)

    def class_F1_score(self):
        classRecall = self.classRecall()
        classRecall[classRecall == 0] = 0.00001
        classAcc = self.classPixelAccuracy()
        class_F1_score = classRecall * classAcc * 2 / ((classRecall + classAcc) + 1e-6)
        return np.round(class_F1_score, 3)

    def F1_score(self):
        classRecall = self.classRecall()
        meanRecall = np.nanmean(classRecall)
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        F1_score = meanRecall * meanAcc * 2 / ((meanRecall + meanAcc) + 1e-6)
        return np.round(F1_score, 3)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / (union + 1e-6)  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return [np.round(mIoU, 3), np.round(IoU, 3)]

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusion_Matrix = count.reshape(self.numClass, self.numClass)
        return confusion_Matrix

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / (np.sum(self.confusionMatrix) + 1e-6)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def confusion_Matrix(self):
        confusionMatrix = self.confusionMatrix / (np.sum(self.confusionMatrix) + 1e-6)
        return confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))