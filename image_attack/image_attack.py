import cv2
import numpy
import matplotlib.pyplot as plt

input_file = 'lena32bit.bmp'

########################################################

# shifting without resize
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img
pts1 = numpy.float32([[0,0],[0,100],[100,0]])
pts2 = numpy.float32([[50,50],[50,150],[150,50]])

M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols,rows))

cv2.imwrite("shifting_without_resize"+".png", dst)

########################################################

# shifting with resize
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
new_row = rows + 123
new_col = cols + 123
dst = numpy.zeros((new_row, new_col, 3))

for i in range(0, new_row):
	for j in range(0, new_col):
		if i < cols and j < rows:
			dst[i][j] = img[i][j]
		else:
			pass

pts1 = numpy.float32([[0,0],[0,100],[100,0]])
pts2 = numpy.float32([[50,50],[50,150],[150,50]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(dst, M, (new_row,new_col))

cv2.imwrite("shifting_with_resize"+".png", dst)

########################################################

# cropping without resize
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img
for i in range(0, cols):
	for j in range(0, rows):
		if i > 450 or j > 450:
			dst[i][j] = 0
cv2.imwrite("cropping_without_resize"+".png", dst)

########################################################

# cropping with resize
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
new_row = rows - 123
new_col = cols - 123
dst = numpy.zeros((new_row, new_col, 3))

for i in range(0, new_row):
	for j in range(0, new_col):
		if i < cols and j < rows:
			dst[i][j] = img[i][j]
		else:
			pass

cv2.imwrite("cropping_with_resize"+".png", dst)

########################################################

# add noise
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img
noise = numpy.zeros((rows, cols, 3))
cv2.randu(noise, (-10,-10,-10), (10,10,10))
#print noise
dst = dst + noise

cv2.imwrite("add_noise"+".png", dst)

########################################################

# median filter
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img
dst = cv2.medianBlur(dst, 3)

cv2.imwrite("median_filter"+".png", dst)

########################################################

# median filter
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img
dst = cv2.blur(dst,(5,5))

cv2.imwrite("mean_filter"+".png", dst)

########################################################

# doodle
img = cv2.imread(input_file, flags=cv2.IMREAD_UNCHANGED)
rows,cols,channels = img.shape
dst = img

for i in range(0, rows):
	for j in range(0, cols):
		if i + j > rows - 20 and i+j < rows + 20:
			dst[i][j] = 0
		if abs(i-j) < 20:
			dst[i][j] = 255

cv2.imwrite("doodle"+".png", dst)

########################################################



