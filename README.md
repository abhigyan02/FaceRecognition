# Face recognition algorithm
This repository contains Python code for performing face recognition using the Olivetti Faces dataset. The code demonstrates how to use Principal Component Analysis (PCA) for dimensionality reduction and three different classifiers (Logistic Regression, Support Vector Machine, and Naive Bayes) for face recognition.

## dataset 
The Olivetti Faces dataset is a collection of 400 images of 40 different people. Each person has 10 images, and each image is a grayscale image of size 64x64 pixels.

## Methodology
The face recognition pipeline consists of the following steps:

1. Loading the Olivetti Faces dataset using sklearn.datasets.fetch_olivetti_faces.
2. Displaying a set of images for each unique person to visualize the dataset.
3. Splitting the dataset into training and test sets using train_test_split.
4. Performing PCA on the training data to reduce the dimensionality of the images.
5. Transforming both the training and test data using the PCA projection.
6. Evaluating the performance of three classifiers (Logistic Regression, Support Vector Machine, and 
   Naive Bayes) using 5-fold cross-validation.

## Results 

![image](https://github.com/abhigyan02/FaceRecognition/assets/75851981/be4adc5e-ac8e-4d79-ac76-b9ce0d13c555)

![image](https://github.com/abhigyan02/FaceRecognition/assets/75851981/e3fdac4f-ca33-4f97-bc31-1980fadb0ba7)

![image](https://github.com/abhigyan02/FaceRecognition/assets/75851981/d574118e-9104-4da2-98ac-3b1a6273581e)

The script will output the mean of the cross-validation scores for each classifier, indicating the performance of each model on unseen data. The cross-validation scores provide an estimate of how well each classifier generalizes to new face images.
