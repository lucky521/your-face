'''

Compute the image quality of an edited or processed image 

python image-iqa.py original_image new_image


Make sure the two images have the same size

http://scikit-image.org/docs/dev/api/skimage.measure.html

'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


origin_name = sys.argv[1]
target_name = sys.argv[2]

origin_img = cv2.imread(origin_name)
target_img = cv2.imread(target_name)
###############################################################################

# Image Quality Assessments

# less is better
# scope 0~inf
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

# more is better
# scope 0~1
def ssim(imageA, imageB):

	return compare_ssim(imageA, imageB, multichannel=True)

# more is better
# scope 0~inf
def psnr(imageA, imageB):

	return compare_psnr(imageA, imageB)



###############################################################################


mse_result = mse(origin_img, target_img)
print("mse of {} and {} is {}".format(origin_name, target_name, mse_result))


ssim_result = ssim(origin_img, target_img)
print("ssim of {} and {} is {}".format(origin_name, target_name, ssim_result))


psnr_result = psnr(origin_img, target_img)
print("psnr of {} and {} is {}".format(origin_name, target_name, psnr_result))


'''


plt.subplot(121), plt.imshow(img), plt.title('Origin')
plt.subplot(122), plt.imshow(dst), plt.title('Target')
plt.show()

'''