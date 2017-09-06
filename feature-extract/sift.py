import cv2  
import numpy as np  
import sys

if len(sys.argv) != 2:
    print 'python sift.py src.png'
    sys.exit()

src_file = sys.argv[1]

#read image  
img = cv2.imread(src_file, cv2.IMREAD_COLOR)  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
cv2.imshow('origin',img);  

#SIFT  
detector = cv2.SIFT()  
keypoints = detector.detect(gray,None)  

#img = cv2.drawKeypoints(gray,keypoints)  
img = cv2.drawKeypoints(gray,keypoints,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
cv2.imshow('sift feature',img);  
cv2.waitKey(0)  
cv2.destroyAllWindows()  
