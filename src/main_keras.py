

import kerasnn
import numpy as np
#import tensorflow as tf
import os
import pickle
import gzip
import datetime
import pandas as pd
from sklearn.metrics import f1_score
import time
import sys
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt

trainfilepath = '/home/kdcse/Documents/Second Semester/TIPR/Assignment-3/tipr-third-assignment/data/CIFAR-10'
testfilepath = '/home/kdcse/Documents/Second Semester/TIPR/Assignment-3/tipr-third-assignment/data/CIFAR-10'
outputDataPath = '/home/kdcse/Documents/Second Semester/TIPR/Assignment-3/tipr-third-assignment/output/'



if __name__ == '__main__':
    #     data = sys.argv[1]
    #     mode = 0
    #     if (data == "--test-data"):
    #         mode = 0  # simply test
    #     elif (data == "--train-data"):
    #         mode = 1  # train and test
    #     if (mode == 0):
    #         testfilepath = sys.argv[2]
    #         datasetname = sys.argv[4]
    #     elif (mode == 1):
    #         configuration = []
    #         trainfilepath = sys.argv[2]
    #         testfilepath = sys.argv[4]
    #         datasetname = sys.argv[6]
    #         con = sys.argv[8]
    #         con = con[1:-1]
    #         con = con.split(',')
    #         x = []
    #         for k in con:
    #             x.append(int(k))
    #         configuration = x
    #         print(configuration)
    #         print(len(configuration))
    #         activation = sys.argv[10]
    configuration = [3, 5]
    activation = "relu"
    datasetname = "CIFAR-10"
    if (datasetname == "CIFAR-10"):
        # region CIFAR-10
        # region Train Data
        for i in range(5):
            train_data_path = os.path.join(trainfilepath, 'data_batch_' + str(i + 1))
            with open(train_data_path, 'rb') as fo:
                datadict = pickle.load(fo, encoding='latin1')
            if (i == 0):
                D = datadict['data']
                L = datadict['labels']
            else:
                D = np.concatenate((D, datadict['data']))
                L = np.concatenate((L, datadict['labels']))
        features = D.reshape((len(D), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = L
        Data = features
        Label = labels
        LabelArray = []
        for i in range(len(Label)):
            label = Label[i]
            l = [0 for j in range(10)]
            l[label] = 1
            LabelArray.append(l)
        # endregion
        # region TestData
        test_data_path = os.path.join(testfilepath, 'test_batch')
        with open(test_data_path, 'rb') as file:
            testdatadict = pickle.load(file, encoding='latin1')

        T = testdatadict['data']
        TL = testdatadict['labels']
        features = T.reshape((len(T), 3, 32, 32)).transpose(0, 2, 3, 1)
        TestData = features
        TestLabel = TL
        TestLabelArray = []
        for i in range(len(TestLabel)):
            label = TestLabel[i]
            l = [0 for j in range(10)]
            l[label] = 1
            TestLabelArray.append(l)
        # endregion
        #trainData, trainLabel, testData, testLabel = Data, LabelArray, TestData, TestLabelArray
        trainData, trainLabel, testData, testLabel = Data, Label, TestData, TestLabel
        # (trainData, valData, trainLabel, valLabel) = train_test_split(Data,LabelArray, test_size=0.20, random_state=42)
        # testData,testLabel=TestData,TestLabelArray
        # endregion
    elif (datasetname == "Fashion-MNIST"):
        # region MNIST Fashion
        # region Training Data
        train_labels_path = os.path.join(trainfilepath, 'train-labels-idx1-ubyte.gz')
        Train_images_path = os.path.join(trainfilepath, 'train-images-idx3-ubyte.gz')
        with gzip.open(train_labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(Train_images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        images = pd.DataFrame(images)
        labels = pd.DataFrame(labels)
        Data = np.array(images)
        Label = np.array(labels)
        LabelArray = []
        for i in range(len(Label)):
            label = Label[i][0]
            l = [0 for j in range(10)]
            l[label] = 1
            LabelArray.append(l)
        # endregion

        # region Test Data
        test_labels_path = os.path.join(trainfilepath, 't10k-labels-idx1-ubyte.gz')
        Test_images_path = os.path.join(trainfilepath, 't10k-images-idx3-ubyte.gz')
        with gzip.open(test_labels_path, 'rb') as lbpath:
            labels_test = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
        with gzip.open(Test_images_path, 'rb') as imgpath:
            images_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels_test), 784)
        testimages = pd.DataFrame(images_test)
        testlabels = pd.DataFrame(labels_test)
        TestData = np.array(testimages)
        TestLabel = np.array(testlabels)
        TestLabelArray = []
        for i in range(len(TestLabel)):
            label = TestLabel[i][0]
            l = [0 for j in range(10)]
            l[label] = 1
            TestLabelArray.append(l)
        # endregion

        # (trainData, valData, trainLabel, valLabel) = train_test_split(Data,LabelArray, test_size=0.20, random_state=42)
        # testData,testLabel=TestData,TestLabelArray
        trainData, trainLabel, testData, testLabel = Data, Label, TestData, TestLabel
        # endregion
    kerasnn.MLP(trainData,trainLabel,testData,testLabel,10,500,80,"CIFAR-10")


    # region Rest Code

    # (trainData, testData, trainLabel, testLabel) = train_test_split(Data,LabelArray, test_size=0.10, random_state=42)

