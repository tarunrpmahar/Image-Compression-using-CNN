import skimage.io
import skimage.transform
import numpy as np


def merge(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def load_img(path):
    img = skimage.io.imread(path).astype(np.float)
    img /= 255.0
    X = img.shape[0]
    Y = img.shape[1]
    S = min(X, Y)
    XX = int((X - S) / 2)
    YY = int((Y - S) / 2)

    if len(img.shape) == 2: img = np.tile(img[:, :, None], 3)
    return skimage.transform.resize(img[XX:XX + S, YY:YY + S], [224, 224])


def ld_one_image(image):
    return np.expand_dims(load_img(image), 0)


def normalization(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)
