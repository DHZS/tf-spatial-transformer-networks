import math
import numpy as np


def image_gallery(images, columns, span=5, padding=2, norm_image=False, expand_dim=False):
    if norm_image:
        images = image_norm(images)
    num_image, h, w, channel = images.shape
    dtype = images.dtype
    rows = math.ceil(num_image / columns)

    # blank gallery
    g_h, g_w = padding * 2 + (h + span) * rows - span, padding * 2 + (w + span) * columns - span
    if dtype == np.uint8:
        gallery = np.ones([g_h, g_w, channel], dtype=dtype) * 255
    elif dtype == np.float32:
        gallery = np.ones([g_h, g_w, channel], dtype=dtype)
    else:
        return None

    # place image block
    for i in range(rows):
        y = padding + (h + span) * i
        for j in range(columns):
            image_index = i * columns + j
            if image_index + 1 > num_image:
                break
            x = padding + (w + span) * j
            gallery[y: y+h, x: x+w] = images[image_index]
    if expand_dim:
        gallery = np.expand_dims(gallery, axis=0)
    return gallery


def image_norm(images):
    n, h, w, c = images.shape
    # reshape
    images = images.transpose([0, 3, 1, 2])
    images = images.reshape([n * c, h * w])
    # normalize
    c_min = np.expand_dims(images.min(axis=1), axis=-1)
    c_max = np.expand_dims(images.max(axis=1), axis=-1)
    images = (images - c_min) / (c_max - c_min)
    # reshape
    images = images.reshape([n, c, h, w])
    images = images.transpose([0, 2, 3, 1])
    return images

