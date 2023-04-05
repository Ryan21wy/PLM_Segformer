import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import cv2
import os


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(image)
    plt.title(title, fontsize=10)
    plt.axis('off')
    return


def visualize_mask(preds, colors=([0, 0, 255], [255, 0, 0], [0, 255, 0], [0, 255, 255], [255, 0, 255]), fig_size=(20, 50)):
    plt.figure(figsize=(fig_size[0], fig_size[1]))
    image_path_list = list(preds.keys())
    num = len(image_path_list)
    count = 0
    for path in image_path_list:
        image = np.array(Image.open(path))
        image_name = os.path.basename(path)
        masks = preds[path]
        vis_masks = []
        for m, c in zip(masks, colors):
            mask = np.ones((image.shape[0], image.shape[1], 3)) * 200
            mask[m == 1] = c

            mask = cv2.addWeighted(image, 0.4, mask, 0.6, 0, dtype=cv2.CV_32F)
            vis_masks.append(mask.astype(np.uint8))

        plt.subplot(6 * num, 1, 1 + 6 * count)
        show_image(image, image_name)

        plt.subplot(6 * num, 1, 2 + 6 * count)
        show_image(vis_masks[0], "Incompleteness")

        plt.subplot(6 * num, 1, 3 + 6 * count)
        show_image(vis_masks[1], "rupture")

        plt.subplot(6 * num, 1, 4 + 6 * count)
        show_image(vis_masks[2], "fiber delamination and warping")

        plt.subplot(6 * num, 1, 5 + 6 * count)
        show_image(vis_masks[3], "contamination")

        plt.subplot(6 * num, 1, 6 + 6 * count)
        show_image(vis_masks[4], "improper restoration")

        count += 1

    plt.subplots_adjust(hspace=0.4)