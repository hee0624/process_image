# 数字图像中的图像噪声


#### 1. 基本介绍
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;噪声是干扰和妨碍人类认知和理解信息的重要因素，而图像噪声则是图像中干扰和妨碍人类认识和理解图像信息的重要因素。其中，在视频采集、视频压缩、信道编码、传输误差和视频解码等常规的视频信息处理过程中，都可能会产生一些失真，引起图像质量的损伤，视频图像会出现信号缺失、模糊、噪点、雪花、条纹、视频偏色、视频抖动等常见异常。这种噪声会对图像质量及后续图像分析都产生一定的影响。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;图像噪声使得图像模糊，甚至有可能淹没图像特征，为了验证图像处理算法的鲁棒性，本项目模拟往数字图像中添加噪声，最大效果地生成现实当中的异常图片，从而为后期图像算法鲁棒性验证提供数据支撑。

对于不同干扰机制生成的噪声图像，表现形式为噪点噪声、条纹噪声、网状噪声。具体下表
<table>
   <tr>
      <td>噪声来源</td>
      <td>干扰形式</td>
      <td>表现形式</td>
   </tr>
   <tr>
      <td rowspan="5">对抗</td>
      <td>随机干扰</td>
      <td>白雪花斑点</td>
   </tr>
   <tr>
      <td>单频干扰</td>
      <td>网状条纹</td>
   </tr>
   <tr>
      <td>电磁干扰</td>
      <td>斜形网纹</td>
   </tr>
   <tr>
      <td>接地干扰</td>
      <td>黑色条纹</td>
   </tr>
   <tr>
      <td>工频干扰</td>
      <td>宽暗条纹、雪花噪点</td>
   </tr>
   <tr>
      <td>机动</td>
      <td>振动</td>
      <td>图像模糊</td>
   </tr>
   <tr>
      <td  rowspan="3">噪声</td>
      <td>高斯噪声</td>
      <td>高斯噪点</td>
   </tr>
   <tr>
      <td>泊松噪声</td>
      <td>泊松噪点</td>
   </tr>
   <tr>
      <td>椒盐噪声</td>
      <td>椒盐噪点</td>
   </tr>
   <tr>
      <td rowspan="3">成像</td>
      <td>色相</td>
      <td>色彩</td>
   </tr>
   <tr>
      <td>饱和度</td>
      <td>色彩纯度</td>
   </tr>
   <tr>
      <td>亮度</td>
      <td>亮度</td>
   </tr>

</table>

#### 2. 代码结构
```shell
.
├── __init__.py
└── noisy_image.py  # 加图像噪声
```

#### 3. 实践
step1. `pip install -i https://pypi.doubanio.com/simple -r requirements.txt`

step2. `python noisy_image.py noisy --input_dir=../data/input/img/ --output_dir=../data/output/img/ --mode=sp --prob=0` 



```shell script
    """
NAME
process_image.py
USE
python process_image.py [TYPE] [OPTION] 
DESCRIPTION
Add noise of various types to a floating-point image
TYPE
	noisy   noise type include gausss, poisson,sp,snow,trip
	enhance  enhance type include brightness,color,contrast,sharpness
OPTION
	--input_dir: str
		Input image dirs
	--output: str
		Output image dirs
--mode : str
    	One of the following strings, selecting the type of noise to add:
    	- 'gauss'   Gaussian-distributed additive noise
        - 'poisson'  Poisson-distributed noise generated from the data
		- 'sp'  Randam switch point add salt and pepper noise
		- 'snow` 	Randam switch point change white 
		- 'stripe` 	Randam switch point generate stripe noise
		- 'net`	 Generate net noise
    --mean: float, optional
        Mean of random distribution.
        Defualt: 0. Used in 'gauss'.
    --var: float, optional
        Variance of random distribution. 
        Default: 100 Used in 'gauss'
    --amount: float, optional
        Proportion of image pixels to replace with noise on range [0, 1]. 
Default:0.05 Used in 'sp'.
Default:0.05 Used in 'snow'.
    --prob: float, optional
        Proportion of image pixels to replace with noise on range [0, 1]. 
		Default:0.2 Used in 'sp'.
    --gap: int, optional
       	Image segment width
		Default:10 Used in 'strip' and 'net'.
    --width: int, optional
Stripe width
Default:5 Used in 'width' and 'net'.
    --degree: int, optional
        Stripe degree
Used in 'stripe' and 'net'.
    --color: str, optional
		Used in 'stripe and 'net'.
		Color parameters in ['black'，'white'， 'bright']
If this parameter is None, image point normal distrubite.
		--value: float, option
			Set color bright, this parameter mush set.
```