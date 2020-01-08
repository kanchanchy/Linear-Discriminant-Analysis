import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_dir = 'MNIST-Dataset/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    fd = open(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY



def fit_lda(trX, trY, tsX, tsY):
    tr5 = []
    tr8 = []
    for i in range(len(trX)):
        if trY[i][0] == 5.0:
            tr5.append(trX[i])
        else:
            tr8.append(trX[i])
    tr5 = np.array(tr5)
    tr8 = np.array(tr8)

    trainX_by_class = {}
    trainX_by_class[5] = tr5
    trainX_by_class[8] = tr8
    
    means = {}
    # calculate means for each class
    means[5] = np.mean(tr5, axis = 0)
    means[8] = np.mean(tr8, axis = 0)

    # calculate the overall mean of all the data
    overall_mean = np.mean(trX, axis = 0)

    no_feature = trX.shape[1]

    # calculate between class covariance matrix
    S_B = np.zeros((no_feature, no_feature))
    for c in means.keys():
        S_B += np.multiply(len(trainX_by_class[c]), np.outer((means[c] - overall_mean), (means[c] - overall_mean)))

    # calculate within class covariance matrix
    S_W = np.zeros(S_B.shape) 
    for c in means.keys(): 
        tmp = np.subtract(trainX_by_class[c].T, np.expand_dims(means[c], axis=1))
        S_W = np.add(np.dot(tmp, tmp.T), S_W)

    mat = np.dot(np.linalg.pinv(S_W), S_B)
    eigvals, eigvecs = np.linalg.eig(mat)
    eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

    # sort the eigvals in decreasing order
    eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

    # take the first 1 eigvector
    w = np.array([eiglist[i][1] for i in range(1)])

    return means, w

def calculate_threshold(means, w):
    tot = 0
    for c in means.keys():
        tot += np.dot(w, means[c])
    w0 = 0.5 * tot
    return w0

def findAccuracy(means, w, w0, data, label):
    c1 = 5
    c2 = 8
    mu1 = np.dot(w, means[c1])
    if (mu1 >= w0):
        class_label = '5'
    else:
        class_label = '8'

    transformed_data = np.dot(w, data.T).T
    if (class_label == '5'):
        transformed_data = [c1 if (transformed_data[i] >= w0) else c2 for i in range(len(transformed_data))]
    else:
        transformed_data = [c1 if (transformed_data[i] < w0) else c2 for i in range(len(transformed_data))]

    correct = 0
    for i in range(len(transformed_data)):
        if transformed_data[i] == label[i][0]:
            correct += 1
    accuracy = correct/len(transformed_data)

    return accuracy




def main():
    trX, trY, tsX, tsY = mnist(noTrSamples=400,
                               noTsSamples=100, digit_range=[5, 8],
                               noTrPerClass=200, noTsPerClass=50)


    trX = trX.T
    trY = trY.T
    tsX = tsX.T
    tsY = tsY.T


    means, w = fit_lda(trX, trY, tsX, tsY)
    threshold = calculate_threshold(means, w)
    print("Threshold value: " + str(threshold))

    train_accuracy = findAccuracy(means, w, threshold, trX, trY)
    print("Train Accuracy: " + str(train_accuracy))

    test_accuracy = findAccuracy(means, w, threshold, tsX, tsY)
    print("Test Accuracy: " + str(test_accuracy))



if __name__ == "__main__":
    main()
