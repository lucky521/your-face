import sys,os
import cv2
import numpy


input_file = sys.argv[1]
todo = sys.argv[2] # "add" or "del" alpha channel

if sys.argv != 3:
	print "python alpa-channel.py input.png add"

read_flag = cv2.IMREAD_UNCHANGED
img = cv2.imread(input_file, flags=read_flag)


print img.shape


if img.shape[2] != 4:

	print "No alpha Channel!"
	if todo == "add":

		print "ADD alpha channel..."

		new_img = numpy.zeros((img.shape[0], img.shape[1], 4))

		for i in range(0, img.shape[0]):
			for j in range(0, img.shape[1]):
				for k in range(0,3):
					new_img[i][j][k] = img[i][j][k]
				new_img[i][j][3] = 100


		out_file = input_file[:-4] + "-alpha" + ".png"
		cv2.imwrite(out_file, new_img)

else:
	print "Have alpha Channel!"
	print img
	if todo == "del":
		print "Del alpha channel..."
		new_img = numpy.zeros((img.shape[0], img.shape[1], 3))
		for i in range(0, img.shape[0]):
			for j in range(0, img.shape[1]):
				for k in range(0,3):
					new_img[i][j][k] = img[i][j][k]

		out_file = input_file[:-4] + "-no-alpha" + ".png"
		cv2.imwrite(out_file, new_img)