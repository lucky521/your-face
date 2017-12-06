
import sys
import time
import numpy as np
import cv2

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value


def main(argv):
    filename = "../lena.png"
    img_codec = cv2.IMREAD_COLOR
    src = cv2.imread(filename, img_codec)
    cv2.imshow("input", src)

    src_shape = src.shape
    print src_shape
    dst = np.zeros(src.shape, src.dtype)

    alpha = 2.0
    beta = 0.0

    for i in range(0, src_shape[0]):
        for j in range(0, src_shape[1]):
            for k in range(0, 3):
                dst[i][j][k] = saturated(src[i][j][k] * alpha + beta)

    print dst
    cv2.imshow("output", dst)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return 0

'''

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



'''


if __name__ == "__main__":
    main(sys.argv[1:])