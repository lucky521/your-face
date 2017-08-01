import cv2, numpy

# http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html

img = cv2.imread('lena.png')
height, width, channel = img.shape
print height, width, channel

#res = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
res = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('scaled_img.png', res)
