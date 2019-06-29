#########################################################################
  #Running HOG SVM on all videos of all classes
##########################################################################
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from pathlib import Path
import tensorflow as tf
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier



listOfVideoPaths = [] #carrying paths of all videos
videoCount = 0  # Count of all videos
videoFrames = [] #carrying all frames of the all videos
listOfFrameLabels = []  #carrying label of each frame
folderIndex = 0 #carrying the index of each folder to use it as a label for each frame

#Creates an instance of a HOGDescriptor class
#Has a built in function "compute"
#Takes one frame as parameter
#Returns the feature vector of that frame
hog = cv2.HOGDescriptor()

#Fills the "listOfVideoPaths" with the paths of all videos - No reading done yet
dataSetPath = 'VisionDataSet/'
#creates a list of folders inside the "VisionDataSet" folder
#folders that are containing the training video data
folders = [f for f in listdir(dataSetPath) if not isfile(join(dataSetPath, f))]
#Fills the "listOfVideoPaths" with paths of all the videos in the dataset
for folder in folders:
    listOfVideoPaths.append([dataSetPath+folder+'/'+ f for f in listdir(dataSetPath+folder) if isfile(join(dataSetPath+folder, f))])
for folder in listOfVideoPaths:
  videoCount+= len(folder)
print("Video Count = ", videoCount)


#Loops on folders inside "ListOfVideoPaths" and reading the videos using their paths
for videoDirectory in listOfVideoPaths:
  folderName = str(videoDirectory[0]).split('/')[1]
  for videoPath in videoDirectory:    
    vidcapture = cv2.VideoCapture(videoPath)
    # Captures video frame by frame
    #success is boolean declares if read successfully or not
    #image is the current frame
    success,image = vidcapture.read()
    #count of frames captured
    framesCount = 0
    #success will be false when there's no frames left to read
    success = True
    while success:
      success,image2 = vidcapture.read()
      #Skipping 10 frames
      if framesCount%10 == 0 and success == True:
        image3 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        image3 = cv2.resize(image3,(64,128))
        #image3 = cv2.resize(image3,(320,240))
        #cv2.imshow('frame%d in list of frames' %len(videoFrames),image3)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #Adding the HOG feature vector of current frame into the list of frames
        videoFrames.append(hog.compute(image3))
        #Adding the label of the current frame into the list of frame labels
        listOfFrameLabels.append(folderIndex)
        print("Frame %d of video %s from class %s added." %(framesCount,videoPath,folderName))
        framesCount += 1
      else:
        framesCount += 1
  #Increasing the index of the folder at the end of each folder
  #To use the index of the next folder as a new label for a new class
  folderIndex +=1


# splits the (videoFrames) and (framesLabels) into training data and testing data  
# framesTrain is the frames training data
# framesTest is the frames testing data 
# labelsTrain is the labels training data
# labelsTest is the labels testing data 
framesTrain, framesTest, labelsTrain, labelsTest = train_test_split(videoFrames, listOfFrameLabels, test_size= 0.30)
svcClassifier = LinearSVC(random_state=0)
framesTrain = np.array(framesTrain)
framesTrain = framesTrain.reshape((framesTrain.shape[0],framesTrain.shape[1]))
framesTest = np.array(framesTest)
framesTest = framesTest.reshape((framesTest.shape[0],framesTest.shape[1]))
labelsTrain = np.array(labelsTrain)
labelsTest = np.array(labelsTest)
labelsTrain=labelsTrain.reshape((labelsTrain.shape[0],1))
svmClassifier=OneVsRestClassifier(svcClassifier)
svcClassifier.fit(framesTrain,labelsTrain)
labelPredict = svcClassifier.predict(framesTest)
print("---------------SVM---------------")
print(confusion_matrix(labelsTest, labelPredict))  
print(classification_report(labelsTest, labelPredict)) 
print('SVM Total Accuracy: ',accuracy_score(labelsTest,labelPredict))

print("\n---------------KNN---------------")
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(framesTrain, labelsTrain)
labelPredict = neigh.predict(framesTest)
print(confusion_matrix(labelsTest, labelPredict))
print(classification_report(labelsTest, labelPredict))
print('KNN Total Accuracy: ',accuracy_score(labelsTest,labelPredict))