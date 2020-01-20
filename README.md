# Fishers Linear Discriminant Analysis (LDA)
This repository performs Fishers Linear Discriminant Analysis (LDA) on MNIST dataset. After you clone this repository, download the MNIST dataset from the website http://yann.lecun.com/exdb/mnist/ and put the downloaded files inside MNIST-Dataset folder.

The repository samples 200 images each from digit 5 and digit 8 from the train set in MNIST to create a training set of 400 images. It also samples 50 images each from digits 5 and 8 from the test set in MINST to create a testing set of 100 images. It then applies Fishers Linear Discriminant to project the train data to 1 dimension and estimate a threshold to separate the two categories. After that, training accuracy and test accuracy are estimated with the Fishers Linear Discriminant. 
