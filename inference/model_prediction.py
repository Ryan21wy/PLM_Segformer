import os
from tqdm import tqdm
import numpy as np
import PIL.Image as Image
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from functools import partial
from training.SegFormer import MixVisionTransformer as Segformer
from training.dataset import PLMDataset
from utils.pre_processing import image_cropping, patch_concat, image_resizing
from utils.post_processing import remove_small_noise, connect_region, remove_small_holes
from utils import evaluation_metrics as eval


def prediction(img_path, model_dir, label_path=None, save_path=None,
               n_class=2, crop_size=None, TTA=False, TLC=False, post=False):
    """Predict the segmentation mask of each damage.

    The segmentation mask of each damage with type of numpy array is predicted
    using optional inference phase augmentation methods.

    Args:
      img_path: file path of PLM images.
      model_dir: file path of saved model parameters.
      label_path:
        file path of damage annotation of PLM images, which uesd to calculate the evaluation metrics.
        If none, no evaluation metrics are provided.
      save_path: file path for saving the segmentation mask of each damage.
      n_class: num of classes, background counts as well.
      crop_size:
        the size of image patches when using the resizing and cropping method.
        If none, using resized images for prediction.
      TTA: If true, using Test Time Augmentation.
      TLC: If true, using Test-time Local Converter method.
      post: If true, using image post-processing method

    Returns:
      preds:
        A dictionary of the prediction results with the file path of the image as the key and
        the predicted segmentation masks list as the value. Segmentation mask is numpy array.
        For example:
        {'C:\data\BG001_013_1.png': [mask1, mask2, mask3],}
    """

    device = torch.device("cuda")

    net_list = []
    for m in os.listdir(model_dir):
        net = Segformer(img_size=512, in_chans=3, num_classes=n_class, patch_size=4, embed_dims=[64, 128, 320, 512],
                        num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                        drop_rate=0.0, drop_path_rate=0.1, TLC=TLC).to(device)

        state_dict = {}
        model_dict = net.state_dict()
        pretrain_model_para = torch.load(os.path.join(model_dir, m), map_location=device)
        for k, v in pretrain_model_para.items():
            if k in model_dict.keys():
                state_dict[k] = v
        model_dict.update(state_dict)
        net.load_state_dict(model_dict)
        net.eval()
        net_list.append(net)

    preds = {}
    file_list = os.listdir(img_path)
    metrics = [eval.SegmentationMetric(n_class) for _ in range(5)]
    for image_name in tqdm(file_list):
        img_file_path = os.path.join(img_path, image_name)
        img_data = Image.open(img_file_path)
        w_r, h_r = img_data.size
        img_unify = image_resizing(img_data, new_h=512, unify_type='all', resize_type=Image.Resampling.LANCZOS)
        img_arr = np.array(img_unify)
        if img_arr.shape[0] >= img_arr.shape[1]:
            img_arr = img_arr.transpose((1, 0, 2))
        h, w, c = img_arr.shape

        if label_path:
            label_file_path = os.path.join(label_path, image_name)
            label = np.array(Image.open(label_file_path))

        with torch.no_grad():
            if crop_size:
                test_imgs = image_cropping(img_arr, crop_size, stride=(1, 1))
            else:
                if w % 16 != 0:
                    a = w % 16
                    b1 = a // 2
                    b2 = a - b1
                    img_arr_cut = img_arr[:, b1: -b2, :]
                else:
                    a = 0
                    img_arr_cut = img_arr
                test_imgs = [img_arr_cut]
            test_dataset = PLMDataset(test_imgs, type='test')
            test_dataload = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0, shuffle=False)

            masks = []
            for i, net in enumerate(net_list):
                pred_images = []
                for x, xh, xv, xhv in test_dataload:
                    if TTA:
                        outputs = net(x.float().to(device))
                        outputs_h = net(xh.float().to(device))
                        outputs_v = net(xv.float().to(device))
                        outputs_hv = net(xhv.float().to(device))
                        img = outputs.cpu().detach().numpy()

                        img_h = outputs_h.cpu().detach().numpy()
                        img_v = outputs_v.cpu().detach().numpy()
                        img_hv = outputs_hv.cpu().detach().numpy()
                        img = img + np.flip(img_h, 2) + np.flip(img_v, 3) + np.flip(img_hv, (2, 3))
                        img = img.transpose((0, 2, 3, 1))
                    else:
                        outputs = net(x.float().to(device))
                        img = outputs.cpu().detach().numpy()
                        img = img.transpose((0, 2, 3, 1))
                    pred_images = img
                if crop_size:
                    mask = patch_concat(pred_images, (h, w), stride=(1, 1))
                else:
                    if a == 0:
                        mask = pred_images[0]
                    else:
                        mask = np.zeros((h, w, 2))
                        mask[:, b1: -b2, :] = pred_images[0]
                mask = np.argmax(mask, axis=2).astype('uint8')
                mask = cv2.resize(mask, dsize=(w_r, h_r), interpolation=cv2.INTER_NEAREST)

                # post-processing
                if post:
                    mask = remove_small_noise(mask)
                    mask = connect_region(mask)
                    mask = remove_small_holes(mask)

                if label_path:
                    class_label = np.zeros_like(label)
                    class_label[label == i + 1] = 1
                    metrics[i].addBatch(mask, class_label)
                masks.append(mask)

            preds[img_file_path] = masks
            if save_path:
                save_dir_path = os.path.join(save_path, os.path.basename(image_name).split('.')[0])
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                damage_names = ['Incompleteness', 'rupture', 'fiber delamination and warping',
                                'contamination', 'improper restoration']
                for n, vm in zip(damage_names, masks):
                    Image.fromarray(mask).save(os.path.join(save_dir_path, n + '.png'), 'PNG')
    if label_path:
        IoUs = []
        for metric in metrics:
            IoU = metric.meanIntersectionOverUnion()[1][1]
            IoUs.append(IoU)
        print('class IoU:')
        print(IoUs)
    return preds