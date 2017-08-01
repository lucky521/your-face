import cv2, numpy

# http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

img = cv2.imread('lena.png')
height, weight, channel = img.shape

#M = numpy.float32([[1,0,100],[0,1,50]])
M = numpy.float32([[1,0,50],[0,1,50]])

dst = cv2.warpAffine(img, M, (height, weight))
cv2.imshow('shifted_img.png', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
