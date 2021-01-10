# coding: utf-8
# @Time    : 2021/1/8 14:10
# @Author  : chenhe<hee0624@163.com>
# @FileName: noisy_image.py
# @Software: PyCharm

"""
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n,is uniform noise with specified mean & variance.

"""

import os
from functools import wraps


import numpy as np
import cv2


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
    out = image + noise
    return out


@std_img
def poisson_noise(image: np.ndarray):
    """泊松噪声
    """
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noise = np.random.poisson(image * vals) / float(vals)
    return noise


def sp_noise(image: np.ndarray, amount: float, prob: float):
    """椒盐噪声
    image:
    amount: 噪声比例
    prob: 盐噪声比例
    """
    out = image.copy()
    p, q = amount, prob
    flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
    salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
    peppered = ~salted
    out[flipped & salted] = 255
    out[flipped & peppered] = 1
    return out


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



if __name__ == '__main__':
    INPUT_PATH = '../data/input/img/00385801.png'
    OUTPUT_PATH = '../data/output/img/00385801.png'
    img = cv2.imread(INPUT_PATH)
    out = strip_noise(img, gap=10, degree=0, prob=0.8)
    # print(out.shape, img.shape)
    cv2.imwrite(OUTPUT_PATH, out)
