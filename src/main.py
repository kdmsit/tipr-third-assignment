import gzip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-3/tipr-third-assignment"
    inputDataPath = "/data"
    outputDataPath = "/output"
    trainData=[]
    trainLabel=[]
    data = input_data.read_data_sets('data/Fashion-MNIST/', one_hot=True)
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))

    # region Fashion-MNIST
    '''f = gzip.open(path + inputDataPath + '/Fashion-MNIST/train-images-idx3-ubyte.gz', 'r')
    image_size = 28
    num_images = 1000
    buf = f.read(image_size * image_size * num_images)
    trainData = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    trainData = trainData.reshape(num_images, np.square(image_size))
    #print(trainData)

    f = gzip.open(path + inputDataPath + '/Fashion-MNIST/train-labels-idx1-ubyte.gz', 'r')
    for i in range(num_images):
        f.read(8)
        buf = f.read(1 * 32)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        trainLabel.append(labels)
    #print(trainLabel)'''
    # endregion
