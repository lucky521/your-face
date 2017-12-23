import cv2
import numpy
import matplotlib.pyplot as plt

# skew, affine transform 
# http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

img = cv2.imread('lena.png')
rows,cols,ch = img.shape

'''
pts1 = numpy.float32([[rows/2,0],[0,cols/2],[rows/2,cols/2]])
pts2 = numpy.float32([
    [rows/2-rows/2/1.414, rows/2-rows/2/1.414],
    [rows/2-rows/2/1.414,rows/2+rows/2/1.141],
    [rows/2,rows/2]
    ])
'''

#pts1 = numpy.float32([[rows/2,0],[0,cols/2],[rows/2,cols/2]])
#pts2 = numpy.float32([[0,0],[0,rows/2*1.414],[rows/4*1.414,rows/4*1.414]])

#pts1 = numpy.float32([[50,50],[200,50],[50,200]])
#pts2 = numpy.float32([[10,100],[200,50],[100,250]])

pts1 = numpy.float32([[0,0],[0,100],[100,0]])
pts2 = numpy.float32([[0,0],[0,50],[50,0]])


M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols,rows))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()

#cv2.imshow('', dst)
#cv2.waitKey(0)
