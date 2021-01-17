# coding: utf-8
# @Time    : 2021/1/8 14:10
# @Author  : chenhe<hee0624@163.com>
# @FileName: noisy_image.py
# @Software: PyCharm

import os
from functools import wraps
from multiprocessing import Process
from multiprocessing import Pool


import fire
import numpy as np
import cv2
from vcam import vcam
from vcam import meshGen
from tqdm import tqdm
from PIL import ImageEnhance
from PIL import Image
from math import *


def noisy_image(input_dir: str, output_dir: str, mode: str, **kwargs):
    """
     Function to add random noise of various types to a floating-point image.

    Parameters
    ----------
    input_dir : str
        Input image dirs.
    output_dir: str
        Output image dirs.
    mode : str
        One of the following strings, selecting the type of noise to add:
        - 'gauss'   Gaussian-distributed additive noise\n.
        - 'poisson'   Poisson-distributed noise generated from the data\n.
    mean: float, optional
        Mean of random distribution. Used in 'gauss'.
        Defualt: 0.
    var: float, optional
        Variance of random distribution. Used in 'gauss'
        Default: 1
    amount: float, optional
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt & pepper'. Default : 0.05
    gap: int, optional
        pass
    degree: int, optional
        pass
    prob: float, optional
        Proportion of image pixels to replace with noise on range [0, 1].
        Used in 'salt & pepper'.Default :
    """
    if os.path.exists(output_dir):
        pass
    else:
        os.mkdir(output_dir)
    if mode == 'gauss':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            src = cv2.imread(path)
            dst = gauss_noise(src, kwargs.get('mean', 0), kwargs.get('var', 100))
            cv2.imwrite(out_path, dst)
    elif mode == 'poisson':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            src = cv2.imread(path)
            dst = poisson_noise(src)
            cv2.imwrite(out_path, dst)
    elif mode == 'sp':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            src = cv2.imread(path)
            dst = sp_noise(src, kwargs.get('amount', 0.05), kwargs.get('prob', 0.2))
            cv2.imwrite(out_path, dst)
    elif mode == 'snow':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            src = cv2.imread(path)
            dst = snow_noise(src, kwargs.get('amount', 0.05))
            cv2.imwrite(out_path, dst)
    elif mode == 'stripe':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            image = cv2.imread(path)
            out = stripe_noise(image, kwargs.get('gap', 10), kwargs.get('width', 5), kwargs.get('degree', 0), kwargs.get('color'), kwargs.get('value', 2))
            cv2.imwrite(out_path, out)
    elif mode == 'net':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            image = cv2.imread(path)
            out = net_noise(image, kwargs.get('gap', 10), kwargs.get('width', 5), kwargs.get('degree', 0), kwargs.get('color'), kwargs.get('value', 2))
            cv2.imwrite(out_path, out)
    elif mode == 'wrap':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            image = cv2.imread(path)
            out = wrap_noise(image)
            cv2.imwrite(out_path, out)
    elif mode == 'glass':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            name, ext = os.path.splitext(out_path)
            out_path = name + '.jpg'
            if ext != '.png':
                continue
            image = cv2.imread(path)
            out = glass_noise(image)
            cv2.imwrite(out_path, out)
    else:
        raise ValueError('不支持该类型')


def enhance_image(input_dir: str, output_dir: str, mode: str, value: str):
    """
     Function to enhance image.

    Parameters
    ----------
    input_dir : str
        Input image dirs.
    output_dir: str
        Output image dirs.
    mode : str
        One of the following strings, selecting the type of noise to add:
        - 'brightness'   .
        - 'color' .
        - 'contrast' .
        - 'sharpness' .
    value: float
    """
    if mode == 'brightness':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            out_path = os.path.splitext(out_path)[0] + '.jpg'
            image = cv2.imread(path)
            out = enhance_brightness(image, value)
            cv2.imwrite(out_path, out)
    elif mode == 'color':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            out_path = os.path.splitext(out_path)[0] + '.jpg'
            image = cv2.imread(path)
            out = enhance_color(image, value)
            cv2.imwrite(out_path, out)
    elif mode == 'contrast':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            out_path = os.path.splitext(out_path)[0] + '.jpg'
            image = cv2.imread(path)
            out = enhance_contrast(image, value)
            cv2.imwrite(out_path, out)
    elif mode == 'sharpness':
        for path in tqdm(in_path(input_dir), total=10, desc='处理', mininterval=0.3, maxinterval=10.0, ncols=100, unit='个'):
            out_path = os.path.join(output_dir, os.path.basename(path))
            out_path = os.path.splitext(out_path)[0] + '.jpg'
            image = cv2.imread(path)
            out = enhance_sharpness(image, value)
            cv2.imwrite(out_path, out)
    else:
        raise ValueError('不支持该类型')


def std_img(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        image = func(*args, **kwargs)
        image = np.clip(image, 0, 255)
        image[image < 0] = 0
        image[image > 255] = 255
        # image = image.astype(np.uint8)
        return image
    return wrapper


def in_path(dir):
    for rt, ds, fs in os.walk(dir):
        for f in fs:
            path = os.path.join(rt, f)
            yield path


def gauss_noise(image: np.ndarray, mean: float = 0, var: float = 1):
    """高斯噪声
    image: 图像的ndarray
    mean: 概率分布的均值
    var: 概率分布的方差
    """
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    dst = image + noise
    return dst


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
    if amount < 0 or amount > 1:
        raise ValueError(f'amount 参数必须在0-1之间')
    if prob < 0 or prob > 1:
        raise ValueError(f'prob 参数必须在0-1之间')
    dst = image.copy()
    p, q = amount, prob
    flipped = np.random.choice([True, False], size=image.shape, p=[p, 1 - p])
    salted = np.random.choice([True, False], size=image.shape, p=[q, 1 - q])
    peppered = ~salted
    dst[flipped & salted] = 0
    dst[flipped & peppered] = 255
    return dst


def snow_noise(image: np.ndarray, amount: float):
    """雪花噪声"""
    return sp_noise(image=image, amount=amount, prob=0)


def stripe_noise(image: np.ndarray, gap: float, width: int, degree: float, color, value):
    h, w, c = image.shape
    if degree == 0:
        line = np.linspace(0, h, int(h/gap))
        for p in line[:-1]:
            p = int(np.floor(p))
            up, down = p - int(width/2), p + int(width/2)
            up = up if up > 0 else 0
            down = down if down < h else h
            if color == 'black':
                image[up: down, :] = 0
            elif color == 'white':
                image[up: down, :] = 255
            elif color == 'bright':
                im = Image.fromarray(image[up: down, :])
                enh_bri = ImageEnhance.Brightness(im)
                brightness = value
                dst = enh_bri.enhance(brightness)
                image[up: down, :] = np.array(dst)
            else:
                image[up: down, :] = image[up: down, :] + np.random.normal(0, 100, image[up: down, :].shape)
        return image
    elif degree == 90:
        line = np.linspace(0, w, int(w/gap))
        for p in line[:-1]:
            p = int(np.floor(p))
            left, right = p - int(width/2), p + int(width/2)
            left = left if left > 0 else 0
            right = right if right < w else w
            if color == 'black':
                image[:, left:right] = 0
            elif color == 'white':
                image[:, left:right] = 255
            elif color == 'bright':
                im = Image.fromarray(image[:, left:right])
                enh_bri = ImageEnhance.Brightness(im)
                brightness = value
                dst = enh_bri.enhance(brightness)
                image[:, left:right] = np.array(dst)
            else:
                image[:, left:right] = image[:, left:right] + np.random.normal(0, 10, image[:, left:right].shape)
        return image
    else:
        dst = remote(image, degree)
        dst = stripe_noise(dst, gap=gap, width=width, degree=0, color=color, value=value)
        dst = remote(dst, -1*degree)
        dst_h, dst_w = dst.shape[:2]
        up = int((dst_h - h) / 2)
        left = int((dst_w - w) / 2)
        dst = dst[up:up + h, left: left + w]
        dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_AREA)
        return dst


def net_noise(image: np.ndarray, gap: float, width: int, degree: float, color, value):
    image = stripe_noise(image, gap, width, degree, color, value)
    image = stripe_noise(image, gap, width, degree+90, color, value)
    return image


def wrap_noise(image: np.ndarray):
    """扭曲噪声"""
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
    """毛玻璃噪声"""
    dst = np.zeros_like(image)
    h, w, _ = dst.shape
    offsets = 10
    random_num = 0
    for y in range(h - offsets):
        for x in range(w - offsets):
            random_num = np.random.randint(0, offsets)
            dst[y, x] = image[y + random_num, x + random_num]
    return dst


def enhance_brightness(image: np.ndarray, vaule: float):
    """亮度增强"""
    image = Image.fromarray(image)
    enh_bri = ImageEnhance.Brightness(image)
    brightness = vaule
    dst = enh_bri.enhance(brightness)
    return np.array(dst)


def enhance_color(image: np.ndarray, vaule: float):
    """色度，饱和度增强"""
    image = Image.fromarray(image)
    enh_col = ImageEnhance.Color(image)
    color = vaule
    dst = enh_col.enhance(color)
    return np.array(dst)


def enhance_contrast(image: np.ndarray, vaule: float):
    """对比度增强"""
    image = Image.fromarray(image)
    enh_con = ImageEnhance.Contrast(image)
    contrast = vaule
    dst = enh_con.enhance(contrast)
    return np.array(dst)


def enhance_sharpness(image: np.ndarray, vaule: float):
    """锐度增强"""
    image = Image.fromarray(image)
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = vaule
    dst = enh_sha.enhance(sharpness)
    return np.array(dst)


def remote(img, degree):
    #degree左转
    src_h, src_w = img.shape[:2]
    dst_h = int(src_w * fabs(sin(radians(degree))) + src_h * fabs(cos(radians(degree))))
    dst_w = int(src_h * fabs(sin(radians(degree))) + src_w * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((src_w / 2, src_h / 2), degree, 1)
    matRotation[0, 2] += (dst_w - src_w) / 2
    matRotation[1, 2] += (dst_h - src_h) / 2
    dst = cv2.warpAffine(img, matRotation, (dst_w, dst_h), borderValue=(255, 255, 255))
    return dst


def main():
    fire.Fire({
        'noisy': noisy_image,
        'enhance': enhance_image
    })

import  time
def f(input_dir, output_dir, **kwargs):
    print('hello', input_dir)
    print('hello', output_dir)
    print(kwargs)
    time.sleep(1000)


def run():
    top_dir = '../data'
    fun = noisy_image
    mode = 'gauss'
    kwds = {}
    pool = Pool()
    for rt, ds, fs in os.walk(top_dir):
        for d in ds:
            if os.path.basename(d) in {'rgb'}:
                path = os.path.join(rt, d)
                input_dir = path
                output_dir = os.path.join(rt, mode + 'rgb')
                pool.apply_async(fun, args=(input_dir, output_dir), kwds=kwds)
    pool.close()
    pool.join()


if __name__ == '__main__':
    # main()
    run()