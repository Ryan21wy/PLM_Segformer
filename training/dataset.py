import os
import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data


class PLMDataset(data.Dataset):
    def __init__(self, root, img_aug_form=None, transform=None, target_transform=None, type=None):
        if type != 'test':
            imgs = []
            file_list = os.listdir(root)
            for img in file_list:
                if 'mask' not in img:
                    mask_name = str(img.split('.')[0]) + '_mask.png'
                    img = os.path.join(root, img)
                    mask = os.path.join(root, mask_name)
                    imgs.append([img, mask])
        else:
            imgs = root

        self.type = type
        self.imgs = imgs
        self.img_aug_form = img_aug_form
        self.transform = transform
        self.target_transform = target_transform

    def size_norm(self, img, norm_size=16):
        w, h = img.size
        if w % norm_size != 0:
            a = w % 16
            b1 = a // 2
            b2 = a - b1
            img_norm = img.crop((b1, 0, w - b2, h))
        else:
            img_norm = img
        return img_norm

    def __getitem__(self, index):
        if self.type != 'test':
            x_path, y_path = self.imgs[index]
            img_x = Image.open(x_path)
            img_y = Image.open(y_path)
            img_x = self.size_norm(img_x)
            img_y = self.size_norm(img_y)
            if self.img_aug_form is not None and self.type == 'train':
                sample = {'image': img_x, 'label': img_y}
                for aug in self.img_aug_form:
                    sample = aug(sample)
                img_x = sample['image']
                img_y = sample['label']

            if self.transform is not None:
                img_x = self.transform(img_x)

            if self.target_transform is not None:
                img_y = self.target_transform(img_y)
            return img_x, img_y

        else:
            img_x = self.imgs[index]
            img_x = img_x / 255.
            img_x_h = np.flip(img_x, axis=0).copy()
            img_x_v = np.flip(img_x, axis=1).copy()
            img_x_hv = np.flip(img_x, axis=(0, 1)).copy()
            imgs = [img_x, img_x_h, img_x_v, img_x_hv]
            for i in range(len(imgs)):
                imgs[i] = (imgs[i] - np.array([[[0.487, 0.428, 0.352]]])) / np.array([[[0.254, 0.219, 0.174]]])
                imgs[i] = imgs[i].transpose((2, 0, 1))
                imgs[i] = torch.from_numpy(imgs[i].copy())
            return imgs

    def __len__(self):
        return len(self.imgs)