# coding: utf-8
# @Time    : 2021/1/8 14:10
# @Author  : chenhe<hee0624@163.com>
# @FileName: noisy_image.py
# @Software: PyCharm


from functools import wraps


import numpy as np
import cv2
from vcam import vcam
from vcam import meshGen


def noisy_image(image: np.ndarray, mode: str, *args, **kwargs):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        'sp'       Replaces random pixels with 0 or 1.
        'strip'   .

    """
    if mode == 'gauss':
        return gauss_noise(image, kwargs.get('mean', 0), kwargs.get('var', 1))
    elif mode == 'poisson':
        return poisson_noise(image)
    elif mode == 'sp':
        return sp_noise(image, kwargs.get('amount'), kwargs.get('prob'))
    elif mode == 'strip':
        return strip_noise(image, kwargs.get('gap'), kwargs.get('degree'), kwargs.get('prob'))
    elif mode == 'wrap':
        return wrap_noise(image)
    elif mode == 'glass':
        return glass_noise(image)
    else:
        raise ValueError('')


def std_img(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        image = func(*args, **kwargs)
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype(np.uint8)
        return image
    return wrapper


@std_img
def gauss_noise(image: np.ndarray, mean: float = 0, var: float = 1):
    """高斯噪声
    image: 图像的ndarray
    mean: 概率分布的均值
    var: 概率分布的方差
    """
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    dst = image + noise
    return dst


@std_img
def poisson_noise(image: np.ndarray):
    """泊松噪声
    """
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    dst = np.random.poisson(image * vals) / float(vals)
    return dst


def sp_noise(image: np.ndarray, amount: float, prob: float):
    """椒盐噪声
    image:
    amount: 噪声比例
    prob: 盐噪声比例
    """
    out_img = image.copy()
    p, q = amount, prob
    flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
    salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
    peppered = ~salted
    out_img[flipped & salted] = 255
    out_img[flipped & peppered] = 1
    return out_img


def strip_noise(image: np.ndarray, gap: float, degree: float, prob: float):
    """"""
    h, w, c = image.shape
    if degree == 0:
        line = np.linspace(0, h, int(h/gap))
        switch = np.random.choice([True, False], size=(len(line)), p=[prob, 1-prob])
        line = line[switch]
        for p in line[:-1]:
            p = int(np.floor(p))
            for i in range(w):
                image[p][i] = image[p][i] + np.random.normal(0, 90)
        return image
    elif degree == 90:
        line = np.linspace(0, w, int(w/gap))
        switch = np.random.choice([True, False], size=(len(line)), p=[prob, 1-prob])
        line = line[switch]
        for p in line[:-1]:
            p = int(np.floor(p))
            for i in range(h):
                image[i][p] = image[i][p] + np.random.normal(0, 90)
        return image


def wrap_noise(image: np.ndarray):
    img = image
    h, w = img.shape[:2]
    camera = vcam(H=h, W=w)
    plane = meshGen(h, w)
    plane.Z = np.sin(8 * np.pi * (plane.X / plane.W))
    # plane.Z -= 100*np.sqrt((plane.X*1.0/plane.W)**2+(plane.Y*1.0/plane.H)**2)
    plane.Z += 20*np.exp(-0.5*((plane.X*1.0/plane.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))
    pts3d = plane.getPlane()
    pts2d = camera.project(pts3d)
    map_x, map_y = camera.getMaps(pts2d)
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    out = cv2.flip(out, 1)
    return out
    pass


def glass_noise(image: np.ndarray):
    dst = np.zeros_like(image)
    h, w, _ = dst.shape
    offsets = 10
    random_num = 0

    for y in range(h - offsets):
        for x in range(w - offsets):
            random_num = np.random.randint(0, offsets)
            dst[y, x] = image[y + random_num, x + random_num]
    return dst


if __name__ == '__main__':
    INPUT_PATH = '../data/input/img/00385801.png'
    OUTPUT_PATH = '../data/output/img/00385801.png'
    img = cv2.imread(INPUT_PATH)
    # out = sp_noise(img, 0.1, 0.2)
    # out = strip_noise(img, gap=10, degree=0, prob=0.8)
    # out = wrap_noise(img)
    out = gauss_noise(img)
    # print(out.shape, img.shape)
    cv2.imwrite(OUTPUT_PATH, out)
