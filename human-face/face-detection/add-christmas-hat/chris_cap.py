import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml" 

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE  # CV_HAAR_SCALE_IMAGE
)

print "Found {0} faces!".format( len(faces) )

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    print "position of face is {} {}".format(x,y)
    print "size of face is {} {}".format(w,h)
    # read christmas hat 
    christ_hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
    hat_h, hat_w, hat_c = christ_hat.shape
    print christ_hat.shape
    # rescale hat according size of face
    scale_change = float(w) / hat_w
    scale_change = 0.9 * scale_change
    #print scale_change
    christ_hat = cv2.resize(christ_hat, None, fx=scale_change, fy=scale_change, interpolation=cv2.INTER_CUBIC)
    hat_h, hat_w, hat_c = christ_hat.shape
    print christ_hat.shape
    print "size of hat is {} {}".format(hat_w,hat_h) 
    # paint with alphal channel
    for x_i in range(0, hat_h):
        for y_i in range(0, hat_w):
            if christ_hat[x_i][y_i][3] > 0:
                if x + x_i - h/2 < 0:
                    continue
                image[x + x_i - h/2][y + y_i] = christ_hat[x_i][y_i][0:3]



cv2.imshow("Faces found" ,image)
cv2.waitKey(0)

