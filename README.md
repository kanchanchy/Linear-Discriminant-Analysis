# Linear Discriminant Analysis
This repository performs principle component analysis on MNIST dataset. After you clone this repository, download the MNIST dataset from the website http://yann.lecun.com/exdb/mnist/ and put the downloaded files inside MNIST-Dataset folder.

The repository samples 200 images each from digit 5 and digit 8 from the train set in MNIST to create a training set of 400 images. It also samples 50 images each from digits 5 and 8 from the test set in MINST to create a testing set of 100 images. It then applies PCA on the train data to reduce the dimension from 784 to 10. It also applies the same transformation to the test data to reduce its dimension to 10. It also reconstructs the original images from the PCA transformed data (from dimension 10 to 784).
