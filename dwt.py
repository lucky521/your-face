import sys,os
import numpy
import matplotlib.pyplot
from PIL import Image
import pywt

def rgb2gray(rgb):
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])

image_path = sys.argv[1]
dwt_method = sys.argv[2] # dwt/dwt2

print pywt.families(short=False)

#######################################################################3
# read img from file
print '#######################################################################'
img_file = Image.open(image_path)
img = numpy.array(img_file)
print 'rgb matrix'
print img

# turn rbg to gray 
gray = rgb2gray(img)
print 'gray matrix', gray.shape
print gray

# create a figure
figure, ax = matplotlib.pyplot.subplots(5,2)

ax[0][0].imshow(gray, cmap=matplotlib.pyplot.get_cmap('gray'))
ax[0][0].set_title("gray")

#######################################################################3
# dwt process
if dwt_method == "dwt":
    dwt_data_cA, dwt_data_cD = pywt.dwt(gray, 'haar')
    dwt_data_cA, dwt_data_cD = pywt.dwt(gray, 'haar')
    print '#######################################################################'
    print dwt_data_cA.shape
    print dwt_data_cA
    print dwt_data_cD.shape
    print dwt_data_cD
elif dwt_method == "dwt2":
    dwt_data_cA, (dwt_data_cH, dwt_data_cV, dwt_data_cD) = pywt.dwt2(gray, 'haar')
    print dwt_data_cA.shape
    print dwt_data_cA
    print dwt_data_cH.shape
    print dwt_data_cH
    print dwt_data_cV.shape
    print dwt_data_cV
    print dwt_data_cD.shape
    print dwt_data_cD


# do some change for frequency display
'''
print numpy.amax(dwt_data_cD)
print numpy.amin(dwt_data_cD)
for (x,y),v in numpy.ndenumerate(dwt_data_cD):
    if v>0:
        print x,y,v
        dwt_data_cD[x][y]=0
'''

# show frequency domain figure
if dwt_method == "dwt":
    ax[1][0].imshow(dwt_data_cA, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[1][0].set_title("dwt_data_cA")

    ax[2][0].imshow(dwt_data_cD, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[2][0].set_title("dwt_data_cD")

elif dwt_method == "dwt2":
    ax[1][0].imshow(dwt_data_cA, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[1][0].set_title("dwt_data_cA")
    ax[1][1].plot(dwt_data_cA)
    ax[1][1].set_title(str(dwt_data_cA.shape)+str(numpy.amin(dwt_data_cA))+" "+str(numpy.amax(dwt_data_cA)))

    ax[2][0].imshow(dwt_data_cH, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[2][0].set_title("dwt_data_cV")
    ax[2][1].plot(dwt_data_cH)
    ax[2][1].set_title(str(dwt_data_cH.shape)+str(numpy.amin(dwt_data_cH))+" "+str(numpy.amax(dwt_data_cH)))

    ax[3][0].imshow(dwt_data_cD, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[3][0].set_title("dwt_data_cV")
    ax[3][1].plot(dwt_data_cD)
    ax[3][1].set_title(str(dwt_data_cD.shape)+str(numpy.amin(dwt_data_cD))+" "+str(numpy.amax(dwt_data_cD)))

    ax[4][0].imshow(dwt_data_cD, cmap=matplotlib.pyplot.get_cmap('gray'))
    ax[4][0].set_title("dwt_data_cD")
    ax[4][1].plot(dwt_data_cD)
    ax[4][1].set_title(str(dwt_data_cD.shape)+str(numpy.amin(dwt_data_cD))+" "+str(numpy.amax(dwt_data_cD)))

#######################################################################3
# idwt process
print '#######################################################################'
if dwt_method == "dwt":
    idwt_data = pywt.idwt(dwt_data_cA, dwt_data_cD, 'haar')
    print idwt_data
elif dwt_method == "dwt2":
    idwt_data = pywt.idwt2((dwt_data_cA, (dwt_data_cH, dwt_data_cV, dwt_data_cD)), 'haar')
    print idwt_data

ax[0][1].imshow(idwt_data, cmap=matplotlib.pyplot.get_cmap('gray'))
ax[0][1].set_title("idwt_data")
matplotlib.pyplot.show()



