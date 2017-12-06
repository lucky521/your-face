from __future__ import print_function
import sys
import time
import numpy as np
import cv2

def is_grayscale(my_image):
    return len(my_image.shape) < 3

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value

def sharpen(my_image):
    if is_grayscale(my_image):
        height, width = my_image.shape
    else:
        my_image = cv2.cvtColor(my_image, cv2.CV_8U)
        height, width, n_channels = my_image.shape
    result = np.zeros(my_image.shape, my_image.dtype)
    
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if is_grayscale(my_image):
                sum_value = 5 * my_image[j, i] - my_image[j + 1, i] - my_image[j - 1, i] \
                            - my_image[j, i + 1] - my_image[j, i - 1]
                result[j, i] = saturated(sum_value)
            else:
                for k in range(0, n_channels):
                    sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k]  \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k]\
                                - my_image[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)
    
    return result

def main(argv):
    filename = "lena.png"
    img_codec = cv2.IMREAD_COLOR
    if argv:
        filename = sys.argv[1]
        if len(argv) >= 2 and sys.argv[2] == "G":
            img_codec = cv2.IMREAD_GRAYSCALE
    src = cv2.imread(filename, img_codec)
    if src is None:
        print("Can't open image [" + filename + "]")
        print("Usage:")
        print("mat_mask_operations.py [image_path -- default ../../../../data/lena.jpg] [G -- grayscale]")
        return -1




    cv2.imshow("Input", src)
    t = round(time.time())
    dst0 = sharpen(src)
    t = (time.time() - t) / 1000
    print("Hand written function time passed in seconds: %s" % t)
    cv2.imshow("Output1", dst0)
    cv2.waitKey()



    t = time.time()
    
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)  # kernel should be floating point type
    
    dst1 = cv2.filter2D(src, -1, kernel)
    # ddepth = -1, means destination image has depth same as input image
    
    t = (time.time() - t) / 1000
    print("Built-in filter2D time passed in seconds:     %s" % t)
    cv2.imshow("Output2", dst1)
    cv2.waitKey(0)


    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])