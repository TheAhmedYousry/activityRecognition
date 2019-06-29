***********************************Activity Recognition***********************************

Activity Recognition is a python project that learns from a video-based dataset to categorize a set of videos, each in their category.

- Algorithms Applied:
	* Histogram of Oriented Gradients (HOG) - for feature extraction.
	* Histogram of Optical Flow (HOF) - for feature extraction.

- Classifiers used:
	* KNN on features extracted by HOG.
	* KNN on features extracted by HOF.
	* KNN on features extracted by HOG + features extracted by HOF.
	* SVM on features extracted by HOG.
	* SVM on features extracted by HOF.
	* SVM on features extracted by HOG + features extracted by HOF.

- Results and accuracies:
	* KNN on features extracted by HOG : 97.6%
	* KNN on features extracted by HOF : 61.2%
	* KNN on features extracted by HOG + features extracted by HOF : 75%
	* SVM on features extracted by HOG : 95.15%
	* SVM on features extracted by HOF : 58.6%
	* SVM on features extracted by HOG + features extracted by HOF : 68.69%

- Dataset used:
	* 793 videos divided into 7 classes, each video is between 3 and 10 seconds.
	* Given videos are splitted to 70% training data and 30% testing data.

- Prerequisites:
	* Python 3.6.8
	* Used video-dataset is placed in a folder with title "VisionDataSet"
	* Dataset is consisted of folders containing videos, each folder has the activity title.

- Contact:
	* theahmedyousry@gmail.com
