# Your-Face

这是一个有关图像处理的综合项目。

## Basic Graphic works 

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




## Image Recognition

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
