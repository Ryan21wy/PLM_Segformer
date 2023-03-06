import cv2
import numpy as np
from skimage import morphology


def remove_small_noise(mask_array, min_size=20):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_array, connectivity=8, ltype=None)
    remains = morphology.remove_small_objects(ar=labels, min_size=min_size, connectivity=1)
    mask_remain = np.zeros_like(mask_array)
    mask_remain[remains > 0] = 1
    return mask_remain


def connect_region(mask_array, kernel_size=(25, 25), iterations=1, kernel_type=cv2.MORPH_RECT):
    kernel_close = cv2.getStructuringElement(kernel_type, kernel_size)
    closing = cv2.morphologyEx(mask_array, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)
    return closing


def remove_small_holes(mask_array, min_size=100):
    mask_trans = np.ones_like(mask_array)
    mask_trans[mask_array == 1] = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_trans, connectivity=8, ltype=None)
    imgs = morphology.remove_small_objects(ar=labels, min_size=min_size, connectivity=1)
    mask_new = np.ones_like(mask_array)
    mask_new[imgs > 0] = 0
    return mask_new


