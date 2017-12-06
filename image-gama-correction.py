
import sys
import time
import numpy as np
import cv2
import math

# https://en.wikipedia.org/wiki/Gamma_correction

def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0
    return sum_value


def main(argv):
    filename = "lena.png"
    img_codec = cv2.IMREAD_COLOR
    src = cv2.imread(filename, img_codec)
    cv2.imshow("input", src)

    src_shape = src.shape
    print src_shape
    dst = np.zeros(src.shape, src.dtype)

    gama = 0.5

    for i in range(0, src_shape[0]):
        for j in range(0, src_shape[1]):
            for k in range(0, 3):
                dst[i][j][k] = saturated( math.pow(src[i][j][k]/255.0, gama) * 255.0 )

    print dst
    cv2.imshow("output", dst)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    return 0



if __name__ == "__main__":
    main(sys.argv[1:])