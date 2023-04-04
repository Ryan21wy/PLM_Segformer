import cv2
import numpy as np


class SegmentationMetric2(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.hit = np.zeros(numClass)
        self.count = np.zeros(numClass)

        self.overlap_ = np.zeros(numClass)
        self.all_ = np.zeros(numClass)

    def addBatch(self, pred, Label):
        for c in range(self.numClass):
            class_gt = np.zeros_like(Label)
            class_gt[Label == c] = 1

            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(class_gt, connectivity=8, ltype=None)
            for j in range(num_labels - 1):
                region_gt = np.zeros_like(labels)
                region_gt[labels == j + 1] = 1

                overlap = pred * region_gt
                recall = np.sum(overlap) / np.sum(region_gt)
                if recall >= 0.5:
                    self.hit[c] += np.sum(region_gt)
                self.count[c] += np.sum(region_gt)

                self.overlap_[c] += np.sum(overlap)
                self.all_[c] += np.sum(region_gt)
            self.all_[c] += np.sum(pred)

    def getMetrics(self):
        hit = self.hit / self.count
        IoU = self.overlap_ / (self.all_ - self.overlap_)
        return np.round(hit, 3), np.round(IoU, 3)