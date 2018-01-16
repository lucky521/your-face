# Your-Face

这是一个有关图像处理的综合项目。



## Basic Graphic works 

对图像的基本操作，空间变换、几何变换、颜色变换、图像叠加。

### Color Space Transform

RGB to Gray

    $ python rgb2gray.py Lena.bmp

Alpha channel

### DWT

    $ python dwt.py lena.png dwt2

   ![](doc/dwt.figure.png)

### FFT

    $ python fft.py lena.png

### Geometric Transform

partitioning

scaling

shifting

skewing

### File Format Transform

png, bmp, jpg

### Filter

noise

mean filter

median filter


### Image blend

blending

overlap


### Brightness

brightness

gama-correction






## Image Accessment

对图像的评估。

### Image Quality Accessment

Mean Square Error (MSE) 均方差

Structure Similaruty (SSIM) 结构相似度

Peak Signal To Noise Ratio (PSNR) 峰值信噪比


	python image-iqa.py original_image new_image





## Image Recognition

图像识别、分类。

tensorflow-model

caffe-model

self-train-model


## Human Face

Face Recognition Using OpenCV Haar Cascades

    $ python static.py IMAGE_FILE
    
    $ python live.py
    




## Third-party package

numpy

matplotlib

pywt

python-opencv
 
    $ brew install opencv
    add "export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH" to ~/.bashrc or ~/.zshrc

PIL

TensorFlow

Scikit-learn

scikit-image
