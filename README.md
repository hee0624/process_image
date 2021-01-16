# noisy_image
generate noisy image

噪点噪声
雪花噪声
条纹噪声

**成像**
1. [done]高斯噪声
2. [done]泊松噪声
3. [done]椒盐噪声


**对抗**
1. [done]雪花噪声-随机信噪比干扰 表现为雪花干扰，图像上会出现雪花状的斑点。
2. [done]条纹噪声 水平+垂直+ 黑色 白色 亮度 
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

s'd
##### 执行代码

高斯噪声 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/gauss/ --mode=gauss --var=250`
泊松噪声 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/poisson/ --mode=poisson`
椒盐噪声 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/sp/ --mode=sp --amount=0.1 --prob=0.2`
雪花噪声 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/snow/ --mode=snow --amount=0.2`

条纹噪声12种类

条纹噪声 水平+白`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/h_b_strip/ --mode=strip --gap=10 --width=3 --degree=0 --color=white`
条纹噪声 水平+黑 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/h_w_strip/ --mode=strip --gap=10 --width=3 --degree=0 --color=black`
条纹噪声 水平+随机 `python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/h_b_strip/ --mode=strip --gap=10 --width=3 --degree=0`
条纹噪声 垂直+黑`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/v_b_strip/ --mode=strip --gap=10 --width=3 --degree=90 --color=black`
条纹噪声 垂直+白`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/v_w_strip/ --mode=strip --gap=10 --width=3 --degree=90 --color=white`
条纹噪声 垂直+随机`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/v_b_strip/ --mode=strip --gap=10 --width=3 --degree=90`
条纹噪声 斜+白`process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/l_w_strip/ --mode=strip --gap=10 --width=3 --degree=45 --color=white`
条纹噪声 斜+黑`python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/l_b_strip/ --mode=strip --gap=10 --width=3 --degree=45 --color=black`
条纹噪声 斜+随机`>python process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/l_strip/ --mode=strip --gap=10 --width=3 --degree=45`

网状噪声8
网状噪声 正+黑
网状噪声 正+白
网状噪声 正+随机
网状噪声 斜+黑 `process_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/n_b_strip/ --mode=net --gap=10 --width=3 --degree=45  --color=black`
网状噪声 斜+白
网状噪声 斜+随机


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