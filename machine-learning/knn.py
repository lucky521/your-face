import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors

# the trainData is 2D data (x,y)->label

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0,2,(25,1)).astype(np.float32)

# print the train data and labels
for i,a in enumerate(trainData):
    print i, a, '->',responses[i]

# Take Red families and plot them 0s
red = trainData[responses.ravel()==0]
plt.scatter(red[:,0],red[:,1],50,'r','^')
# Take Blue families and plot them 1s
blue = trainData[responses.ravel()==1]
plt.scatter(blue[:,0],blue[:,1],50,'b','s')

#plt.show()



newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
print 'New data->', newcomer
plt.scatter(newcomer[:,0],newcomer[:,1],80,'g','o')


# OpenCV KNN
knn = cv2.ml.KNearest_create()
knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)

print( "result:  {}\n".format(results) )
print( "neighbours:  {}\n".format(neighbours) )
print( "distance:  {}\n".format(dist) )




'''
# sciki-learn KNN
# Finds the K-neighbors of a point.
neigh = neighbors.NearestNeighbors(5)
neigh.fit(trainData, responses)
print 'Nearest 5 data->', neigh.kneighbors(newcomer, return_distance=False)
# knn classifier
knn = neighbors.KNeighborsClassifier(5)
knn.fit(trainData, responses)
print 'New Comer should be labeled as ->', knn.predict(newcomer)
'''

plt.show()
