import numpy as np
from PIL import Image


def get_inital_seq(length, size, stride):
    n1 = length // size
    l_r = length - n1 * size
    size_2 = size // stride
    n2 = l_r // size_2
    l_rr = l_r - n2 * size_2
    if l_rr == 0:
        num = (n1 - 1) * stride + n2 + 1
    else:
        num = (n1 - 1) * stride + n2 + 2
    seq = np.arange(0, num * size_2, size_2)
    seq[-1] = length - size
    return seq


def image_cropping(data, target_size, stride):
    w, h, c = data.shape
    ws, hs = stride[0], stride[1]
    rowsize, colsize= target_size

    row_seq = get_inital_seq(w, rowsize, ws)
    col_seq = get_inital_seq(h, colsize, hs)

    pieces = []
    for r in row_seq:
        for c in col_seq:
            piece = data[r: r + rowsize, c: c + colsize, :]
            pieces.append(piece)
    return pieces


def patch_concat(data_list, target_shape, stride):
    w, h = target_shape
    rowsize, colsize, ch = data_list[0].shape
    ws, hs = stride[0], stride[1]
    target = np.zeros((w, h, ch))

    row_seq = get_inital_seq(w, rowsize, ws)
    col_seq = get_inital_seq(h, colsize, hs)

    num = 0
    for r in row_seq:
        for c in col_seq:
            target[r: r + rowsize, c: c + colsize, :] = data_list[num]
            num += 1
    return target


def image_resizing(img, new_h=None, new_w=None, unify_type='all', resize_type=Image.Resampling.NEAREST):
    w, h = img.size
    resize_ = False
    if unify_type == 'all':
        resize_ = True
    elif unify_type == 'low':
        if new_h is not None and h < new_h:
            resize_ = True
        elif new_w is not None and w < new_w:
            resize_ = True
        else:
            resize_ = False
    else:
        resize_ = False

    if resize_:
        if new_h is not None and new_w is None:
            scale_factor = new_h / h
            w_n = int(scale_factor * w)
            h_n = new_h
        elif new_w is not None and new_h is None:
            scale_factor = new_w / w
            h_n = int(scale_factor * h)
            w_n = new_w
        elif new_h is not None and new_w is not None:
            h_n = new_h
            w_n = new_w
        pic_scale = img.resize((w_n, h_n), resize_type)
    else:
        pic_scale = img
    return pic_scale