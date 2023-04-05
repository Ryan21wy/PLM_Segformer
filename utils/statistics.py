import numpy as np


def getStatistics(preds, num_class=6):
    counts = np.zeros(num_class).astype(np.uint16)
    pixel_counts = np.zeros(num_class)
    image_path_list = list(preds.keys())
    for path in image_path_list:
        masks = preds[path]
        damage_pixels = 0
        h, w = masks[0].shape
        for i, m in enumerate(masks):
            num = np.sum(m)
            pixel_counts[i + 1] += num
            if num != 0:
                counts[i + 1] += 1
            damage_pixels += num
        pixel_counts[0] += h * w - damage_pixels
        if damage_pixels == 0:
            counts[0] += 1
    area_ratios = np.round((pixel_counts / np.sum(pixel_counts)) * 100, 3)
    print('frequency of each damages:')
    print('non-damage, incompleteness, rupture, fiber delamination and warping, contamination, improper restoration')
    print(counts)
    print('area proportion of each damages:')
    print('non-damage, incompleteness, rupture, fiber delamination and warping, contamination, improper restoration')
    print(area_ratios)
    return counts, area_ratios