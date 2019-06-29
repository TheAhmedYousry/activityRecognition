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
import math
import itertools
from sklearn.neighbors import KNeighborsClassifier


listOfVideoPaths = [] #carrying paths of all videos
videoCount = 0  # Count of all videos
listOfHistograms = [] #carrying list of histograms for each cell
featureVector = [] #carrying feature vectors of each frame
listOfFeatureVectors = [] #carrying feature vectors of all frames - divided
listOfFrameLabels = []  #carrying label of each frame
folderIndex = 0 #carrying the index of each folder to use it as a label for each frame

#Window size 8*8
windowsize_height = 8
windowsize_width = 8

#Histogram Bins
histogramBins = [0,20,40,60,80,100,120,140,160]

#Getting index of an angle inside the bin
def get_indicies(number):
    min = 0
    max = 0

    if number > 360:
      number = number - 360

    if number > 180 and number < 360:
      number = number-180

    if number > 160 and number < 180:
      min = 160
      max = 0
      return min, max

    # check max
    for i in range(0,160,20):
      if number >  i:
        min = i
        max = i + 20
      
    return min, max

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
    imagePrev = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imagePrev = cv2.resize(imagePrev,(64,128))
    #count of frames captured
    framesCount = 0
    
    #success will be false when there's no frames left to read
    success = True
    while success:
      success,image2 = vidcapture.read()
      #Skipping 3 frames
      if framesCount%3 == 0 and success == True:
        imageNext = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        imageNext = cv2.resize(imageNext,(64,128))
        #creating a 9-valued bin for current cell
        cellBin = [0,0,0,0,0,0,0,0,0]

        for windowHeight in range (0,imageNext.shape[0],windowsize_width):
          for windowWidth in range (0,imageNext.shape[1],windowsize_height):
            windowPrev = imagePrev[windowHeight:windowHeight+windowsize_height, windowWidth:windowWidth+windowsize_width]
            windowNext = imageNext[windowHeight:windowHeight+windowsize_height, windowWidth:windowWidth+windowsize_width]
            #print(len(windowPrev))
            #Calculating the flow between the two windows
            flow = cv2.calcOpticalFlowFarneback(windowPrev,windowNext, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #Calculating the magnitude and the angle
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1],angleInDegrees=True)
            
            #print('\nAngles of cell:\n',ang)
            #print('\nmagnitudes of cell :\n',mag)
            
            #calculating the histogram bin of each window
            for heightOfCell in range(8):
              for widthOfCell in range(8):
                #if angle == 0 or 20 or 40 or 60 or 80 or 100 or .... 160
                #will add it's value directly without applying any equations
                if ang[heightOfCell][widthOfCell] in histogramBins:
                  cellBin[histogramBins.index(ang[heightOfCell][widthOfCell])] += mag[heightOfCell][widthOfCell]
                #but if the angle equals another different number :
                else:
                  if ang[heightOfCell][widthOfCell] > 180 and ang[heightOfCell][widthOfCell] < 360:
                    ang[heightOfCell][widthOfCell] = ang[heightOfCell][widthOfCell] - 180
                  lowerBound, upperBound = get_indicies(ang[heightOfCell][widthOfCell])
                  minIndex = histogramBins.index(lowerBound)
                  maxIndex = histogramBins.index(upperBound)
                  percentageMax = (ang[heightOfCell][widthOfCell]-lowerBound)/(upperBound-lowerBound)
                  percentageMin = 1 - percentageMax
                  cellBin[maxIndex] += percentageMax * mag[heightOfCell][widthOfCell]
                  cellBin[minIndex] += percentageMin * mag[heightOfCell][widthOfCell]
            listOfHistograms.append(cellBin)
            cellBin = [0,0,0,0,0,0,0,0,0]

        for histogramBin in range (len(listOfHistograms)):
          if (histogramBin - 7)%8 ==0:
            continue
          else:
            if histogramBin <= len(listOfHistograms)-9:
              topLeft = listOfHistograms[histogramBin]
              topRight = listOfHistograms[histogramBin+1]
              bottomLeft = listOfHistograms[histogramBin+8]
              bottomRight = listOfHistograms[histogramBin+9]

              wholeList = topLeft + topRight + bottomLeft + bottomRight
              #topLeft = topRight = bottomLeft = bottomRight = topList = bottomList = None
              l2Norm = math.sqrt(sum(i*i for i in wholeList))
              
              for number in topLeft:
                if l2Norm != 0:
                  number = number/l2Norm
              featureVector += topLeft
              topLeft = None

              for number in topRight:
                if l2Norm != 0:
                  number = number/l2Norm
              featureVector += topRight
              topRight = None

              for number in bottomLeft:
                if l2Norm != 0:
                  number = number/l2Norm
              featureVector += bottomLeft
              bottomLeft = None

              for number in bottomRight:
                if l2Norm != 0:
                  number = number/l2Norm
              featureVector += bottomRight
              bottomRight = None
            else:
              break

        listOfFeatureVectors.append(featureVector)
        featureVector = []
        listOfFrameLabels.append(folderIndex)
        listOfHistograms = []
        print("Frame %d of video %s from class %s added." %(framesCount,videoPath,folderName))
        framesCount += 1
        imagePrev = imageNext
      else:
        framesCount += 1
        imagePrev = imageNext
  #Increasing the index of the folder at the end of each folder
  #To use the index of the next folder as a new label for a new class
  folderIndex +=1


# splits the (videoFrames) and (framesLabels) into training data and testing data  
# framesTrain is the frames training data
# framesTest is the frames testing data 
# labelsTrain is the labels training data
# labelsTest is the labels testing data 
framesTrain, framesTest, labelsTrain, labelsTest = train_test_split(listOfFeatureVectors, listOfFrameLabels, test_size= 0.30)
svcClassifier = LinearSVC(random_state=0)
framesTrain = np.array(framesTrain)
framesTest = np.array(framesTest)
labelsTrain = np.array(labelsTrain)
labelsTest = np.array(labelsTest)
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