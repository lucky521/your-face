import sys,os
import cv2
import numpy

# python overlap.py bottom.png top.png

bottom_file = sys.argv[1]
top_file = sys.argv[2]

read_flag = cv2.IMREAD_UNCHANGED
bottom_img = cv2.imread(bottom_file, flags=read_flag)
top_img = cv2.imread(top_file, flags=read_flag)

print bottom_img.shape
print top_img.shape

print top_img


# do transparency overlap

# dst = src1*alpha + src2*beta + gamma;
alpha = 1
beta = 0.5
gamma = 0
dst = cv2.addWeighted(bottom_img, alpha, top_img, beta, gamma)

cv2.imshow("overlaped", dst)
cv2.waitKey(0)
cv2.imwrite("1.png", dst)



# do opaque overlap

img2gray = cv2.cvtColor(top_img,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(bottom_img,bottom_img,mask = mask_inv)  # change top area to 0

img2_fg = cv2.bitwise_and(top_img,top_img,mask = mask) # change bottom area to 0

dst = cv2.add(img1_bg,img2_fg)

cv2.imshow("overlaped", dst)
cv2.waitKey(0)