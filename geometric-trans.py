import cv2
import numpy


origin_file = "lena.png"
unknow_file = "1000.png"

block_w = 256
block_h = 256


# draw a black line on the border of image buffer
def draw_border(image_buffer, width, height):
	if len(image_buffer) < height:
		return
	if len(image_buffer[0]) < width:
		return

	for i in range(0, width):
		image_buffer[i][0] = [0,0,0]
		image_buffer[i][height-1] = [0,0,0]

	for j in range(0, height):
		image_buffer[0][j] = [0,0,0]
		image_buffer[width-1][j] = [0,0,0]



# Give a image file, return image buffer with aligned size
def image_file_align(origin_file, block_w, block_h):

	img = cv2.imread(origin_file)
	height, width, channels = img.shape

	# Compute aligned_width and aligned_height
	aligned_width = ((width + block_w-1) & (~(block_w-1)))
	aligned_height = ((height + block_h-1) & (~(block_h-1)))

	new_img = numpy.zeros(shape=(aligned_width, aligned_height, channels))

	for i in range(0, width):
		for j in range(0, height):
			new_img[i][j] = img[i][j]

	for i in range(width, aligned_width):
		for j in range(height, aligned_height):
			new_img[i][j] = [0,0,0]

	print new_img
	return new_img, aligned_width, aligned_height





# Given a image file, cropping it 
def image_cropping(origin_file):

	img = cv2.imread(origin_file)

	height, width, channels = img.shape
	print height, width, channels

	# Crop image to unit blocks
	for i in range(0, width, block_w):
		for j in range(0, height, block_h):
			print i, j
			crop_part = img[j:j+block_h, i:i+block_w]
			cv2.imshow("crop", crop_part)
			cv2.waitKey(0)


# Given a image file, try to crop but not seperate them
def image_draw_cropping(origin_file):

	img, width, height = image_file_align(origin_file, block_w, block_h)

	# Crop image to unit blocks
	for i in range(0, width, block_w):
		for j in range(0, height, block_h):
			print i, j
			draw_border(img[j:j+block_h, i:i+block_w], block_w, block_h)

	cv2.imshow("crop", img)
	cv2.waitKey(0)





image_draw_cropping(origin_file)




# Given a unknow file, explore all the possibilities
def cropping_explore(unknow_file):

	img = cv2.imread(unknow_file)
	height, width, channels = img.shape

	# remove one row
	for i in range(1, block_w):
		crop_img = img[0:height, i:width]
		cv2.imshow("", crop_img)
		cv2.waitKey(0)
		#cv2.imwrite("cut.png", crop_img)


	# remove one column
	for j in range(1, block_h):
		crop_img = img[j:height, 0:width]
		cv2.imshow("", crop_img)
		cv2.waitKey(0)
		#cv2.imwrite("cut.png", crop_img)














'''
crop_img = img[200:400, 100:300] 
# Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
'''


#cv2.imwrite('messigray.png',img)