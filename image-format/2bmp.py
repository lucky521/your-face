import sys,os
import cv2
import numpy

# python 2bmp.py lena.png

input_file = sys.argv[1] #"lena.png"
out_file = input_file[:-3] + "bmp"

read_flag = cv2.IMREAD_UNCHANGED
#read_flag = cv2.IMREAD_COLOR #cv2.IMREAD_UNCHANGED
img = cv2.imread(input_file, flags=read_flag)
print img.dtype
print img.shape

new_img = numpy.zeros((img.shape[0], img.shape[1], 4))

# conversion from 24bit to 32bit
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
        for k in range(0, 3):
            new_img[i][j][k] = img[i][j][k]
        #new_img[i][j][3] = 0xFF

# test
print img[133][122]
print new_img[133][122]
#print new_img

cv2.imwrite(out_file, new_img)



