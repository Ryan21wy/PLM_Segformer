import os
from tqdm import tqdm
import random
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader

from training.dataset import PLMDataset
from training.augmentaion import Normalize, RandomScaleCrop, RandomVerticalFlip, RandomHorizontalFlip
from utils import evaluation_metrics as eval
from training.Adan import Adan
from training.SegFormer import MixVisionTransformer as Segformer
from training.losses import Focal, Dice

from timm.scheduler import CosineLRScheduler
from functools import partial
from torch.cuda.amp import autocast as autocast, GradScaler


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


def evaluation_index(metric):
    cm = metric.confusion_Matrix()
    c_precision = metric.classPixelAccuracy()
    c_recall = metric.classRecall()
    c_F1_score = metric.class_F1_score()
    m_precision = metric.meanPixelAccuracy()
    m_recall = metric.meanRecall()
    F1_score = metric.F1_score()
    miou = metric.meanIntersectionOverUnion()[0]
    c_iou = metric.meanIntersectionOverUnion()[1]
    m1 = [c_precision, c_recall, c_F1_score, c_iou]
    m2 = [m_precision, m_recall, F1_score, miou]
    return m2, m1, cm


def calculate_weights_labels(data_path, num_classes, lamda=1):
    z = np.zeros((num_classes,))
    file_list = os.listdir(data_path)
    for file in file_list:
        if '_mask' in file:
            mask = np.array(Image.open(os.path.join(data_path, file)))
            y = (mask >= 0) & (mask < num_classes)
            labels = mask[y].astype(np.uint8)
            count_l = np.bincount(labels, minlength=num_classes)
            z += count_l
    ratios = z / np.sum(z)
    class_weights = 1 / (ratios ** lamda)
    class_weights = class_weights / np.sum(class_weights) * num_classes
    ret = np.array([class_weights]).astype(np.float32)
    return ret


def train_model(train_path, val_path, model_name, save_path,
                lr=1e-2, wd=1e-2, bs=8, acc_step=1, epochs=100, n_class=2, lamda=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    a_transform = [RandomScaleCrop(scale_rate=[0.8, 1.5]),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip()]
    x_transform = T.Compose([T.ToTensor(),
                             T.Normalize([0.487, 0.428, 0.352], [0.254, 0.219, 0.174])],
                            )
    y_transform = T.ToTensor()

    model = Segformer(img_size=512, in_chans=3, num_classes=n_class, patch_size=4, embed_dims=[64, 128, 320, 512],
                      num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                      drop_rate=0.0, drop_path_rate=0.1).to(device)

    epoch_loss_list = []
    epoch_miou_list = []
    epoch_ap_list = []

    val_loss_list = []
    val_miou_list = []
    val_ap_list = []

    print('prepare finished')
    save_path = save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset = PLMDataset(train_path, img_aug_form=a_transform, transform=x_transform, target_transform=y_transform)
    val_dataset = PLMDataset(val_path, transform=x_transform, target_transform=y_transform)

    print('data set load finished')
    print('size of train set: ' + str(len(train_dataset)))
    print('size of val set: ' + str(len(val_dataset)))

    train_dataload = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    val_dataload = DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    print('batch load finished')
    print('Training rounds：' + str(len(train_dataload)))

    opt_parameters = []
    no_decay = ["bias", "norm", "bn", "pos_embed", "relative_position_bias_table"]
    for name, param in model.named_parameters():
        no_wd = False
        if 'decode_head' in name:
            for p in no_decay:
                if p in name:
                    params = {"params": param, "weight_decay": 0., 'lr': 1 * lr}
                    opt_parameters.append(params)
                    no_wd = True
                    break
            if not no_wd:
                params = {"params": param, "weight_decay": wd, 'lr': 1 * lr}
                opt_parameters.append(params)
        else:
            for p in no_decay:
                if p in name:
                    params = {"params": param, "weight_decay": 0.}
                    opt_parameters.append(params)
                    no_wd = True
                    break
            if not no_wd:
                params = {"params": param, "weight_decay": wd}
                opt_parameters.append(params)
    optimizer = Adan(opt_parameters, lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=0.01 * lr, warmup_t=10, warmup_lr_init=0.01 * lr)

    weights = calculate_weights_labels(train_path, num_classes=n_class, lamda=lamda)
    weights = y_transform(weights).to(device).float()
    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')

    metric = eval.SegmentationMetric(n_class)
    metric2 = eval.SegmentationMetric(n_class)

    scaler = GradScaler()

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        batch_loss_list = []
        batch_vloss_list = []
        for k, (x, y) in enumerate(stable(train_dataload, 42 + epoch)):
            inputs = x.float().to(device)
            labels = y.to(device) * 255.
            labels = labels.long().squeeze(1)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / acc_step
            pred = outputs.max(1, keepdim=True)[1]
            pred = pred.squeeze().cpu().detach().numpy()
            truth = labels.squeeze().cpu().detach().numpy()
            metric.addBatch(pred.astype(np.int64), truth.astype(np.int64))
            scaler.scale(loss).backward()
            if ((k + 1) % acc_step == 0) or ((k + 1) == len(train_dataload)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            batch_loss_list.append(loss.item() * acc_step)
            train_loss += loss.item() * acc_step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x, y in stable(val_dataload, 42 + epoch):
                inputs = x.float().to(device)
                labels = y.to(device) * 255.
                outputs = model(inputs)
                labels = labels.long().squeeze(1)

                pred = outputs.max(1, keepdim=True)[1]
                pred = pred.squeeze().cpu().detach().numpy()

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                batch_vloss_list.append(loss.item())
                truth = labels.squeeze().cpu().detach().numpy()
                metric2.addBatch(pred.astype(np.int64), truth.astype(np.int64))
        tloss = train_loss / len(train_dataload)
        epoch_loss_list.append(tloss)
        miou = metric.meanIntersectionOverUnion()
        epoch_miou_list.append(miou[0])
        pa = metric.pixelAccuracy()
        epoch_ap_list.append(pa)
        metric.reset()

        vloss = val_loss / len(val_dataload)
        val_loss_list.append(vloss)
        vacc = metric2.pixelAccuracy()
        val_ap_list.append(vacc)
        vmiou = metric2.meanIntersectionOverUnion()
        val_miou_list.append(vmiou[0])
        metric2.reset()

        if (epoch + 1) % 20 == 0:
            print('train_acc is :' + str(pa))
            print('val_acc is :' + str(vacc))
            print('train_miou is :' + str(miou))
            print('val_miou is :' + str(vmiou))
        scheduler.step(epoch)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    history = np.array([np.array(epoch_loss_list),
                         np.array(val_loss_list),
                         np.array(epoch_miou_list),
                         np.array(val_miou_list),])
    np.save(save_path + '/history.npy', history)


if __name__ == "__main__":
    lr = 3e-4
    wd = 1e-2
    batch_size = 8
    accumulation_steps = 1
    epochs = 10
    num_class = 7
    lamda = 0.5

    seed_everything(42)
    train_model('train set path',
                'validation set path',
                'save path',
                'model name',
                lr=lr, wd=wd, bs=batch_size, acc_step=accumulation_steps, epochs=epochs, n_class=num_class, lamda=lamda)