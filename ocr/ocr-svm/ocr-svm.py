'''

OCR using svm classifier

python3.

'''

import cv2 as cv
import numpy as np
import sys
#####################################################

# train the model with digits.png 5000 digit pictures

SZ=20
bin_n = 16 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR

# orignal pixel image 50x50
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

img = cv.imread('digits.png',0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

# train data
deskewed = [list(map(deskew,row)) for row in train_cells]
print(deskewed[49][49])
hogdata = [list(map(hog,row)) for row in deskewed]
print(type(hogdata))
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]
print(len(trainData[0]))
print(trainData[0])
print(len(responses))
print(responses)

# train
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)
svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

# predict data
deskewed = [list(map(deskew,row)) for row in test_cells]
cv.imwrite("resized-input-11.png", deskewed[25][25])
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
# predict
result = svm.predict(testData)[1]
print(len(result))
print(result)

mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)

####################################################

# read a new image and resize to 50x50
new_img = cv.imread(sys.argv[1],0)
new_img = cv.resize(new_img, (20,20))
cv.imwrite("resized-input.png", new_img)

testData = []
testData.append(hog(deskew(new_img)))
testData = np.float32(testData).reshape(-1,bin_n*4)
# predict new image
result = svm.predict(testData)[1]
print('Predict value is {}'.format(int(result[0][0])))








