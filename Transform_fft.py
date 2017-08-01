import sys,os
import numpy
import matplotlib.pyplot
from PIL import Image

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

image_path = sys.argv[1]
show_fq_figure = 1

#######################################################################3
# read img from file
img_file = Image.open(image_path)
img = numpy.array(img_file)
print 'rgb matrix'
print img

# turn rbg to gray 
gray = rgb2gray(img)
print 'gray matrix', gray.shape
print gray

#######################################################################3
# fft process
fft_img = numpy.fft.fft2(gray)

# do some change for frequency display
show_fft_img = fft_img
show_fft_img = numpy.fft.fftshift(show_fft_img)
show_fft_img = numpy.abs(show_fft_img)
show_fft_img = numpy.log(show_fft_img)

# show frequency domain figure
if show_fq_figure:
    print 'fft show matrix', show_fft_img.shape
    print show_fft_img
    matplotlib.pyplot.imshow(show_fft_img, cmap=matplotlib.pyplot.get_cmap('gray'))
    matplotlib.pyplot.show()


'''
# do some modify in frequency domain
print 'do something on frequency domain...'
print 'min', numpy.amin(fft_img)
print 'max', numpy.amax(fft_img)
#fft_img = fft_img.astype(int)
#show_fft_img = show_fft_img.astype(int)
ma = numpy.amax(fft_img)
mi = numpy.amin(fft_img)
for (x,y),v in numpy.ndenumerate(show_fft_img):
    if not( abs(x-512/2) > 100 or abs(y-512/2) > 100):
        show_fft_img[x][y] = mi
    else:
        show_fft_img[x][y] = numpy.float64(show_fft_img[x][y])
    #print 'x,y,v',x,y,show_fft_img[x][y]


#matplotlib.pyplot.imshow(show_fft_img, cmap=matplotlib.pyplot.get_cmap('gray'))
#matplotlib.pyplot.show()


show_fft_img = numpy.fft.ifftshift(show_fft_img)


fft_img = show_fft_img
'''

#######################################################################3
# inverse fft process
ifft_img = numpy.fft.ifft2(fft_img)
ifft_img = numpy.abs(ifft_img)

# show new gray figure
print 'new gray matrix'
print ifft_img
matplotlib.pyplot.imshow(ifft_img, cmap=matplotlib.pyplot.get_cmap('gray'))
matplotlib.pyplot.show()




