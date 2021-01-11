# noisy_image
generate noisy image

噪点噪声
雪花噪声
条纹噪声

1. [done]高斯噪声
2. [done]泊松噪声
3. [done]椒盐噪声
4. [done]滚动横条纹
5. [done]滚动竖条纹
6. 网状条纹
7. [done]轻微扭曲
8. 雪花状斑点

色调
色相
亮度:亮度指照射在景物或图像上光线的明暗程度。图像亮度增加时，就会显得耀眼或刺眼，亮度越小时，图像就会显得灰暗。
饱和对:
对比度: 对比度指不同颜色之间的差别。对比度越大，不同颜色之间的反差越大，即所谓黑白分明，对比度过大，图像就会显得很刺眼。对比度越小，不同颜色之间的反差就越小。

##### 执行代码

`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/img/ --mode=sp --prob=0`
```shell script
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
```