import sys, os
import numpy
import matplotlib.pyplot
import matplotlib.image
from PIL import Image

imagePath = sys.argv[1]

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# read from file
imgfile = Image.open(imagePath)
img = numpy.array(imgfile)

# read from file option2, use matplotlib, but it only support png.
#img = matplotlib.image.imread(imagePath)


# rgb to gray
print 'rgb matrix'
print img # three channel for each pixel
gray = rgb2gray(img)
print 'gray matrix'
print gray # one channel for each pixel

# show to figure
matplotlib.pyplot.imshow(gray, cmap=matplotlib.pyplot.get_cmap('gray'))
matplotlib.pyplot.show()

# save to file
matplotlib.image.imsave('gray_'+imagePath+'.png', gray, cmap=matplotlib.pyplot.get_cmap('gray'))



