import numpy as np
import tensorflow as tf
import os
import pandas as  pd
import gzip
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import train_test_split


def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
    with tf.variable_scope(name) as scope:
        # Shape of the filter-weights for the convolution
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        # Create new weights (filters) with the given shape
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

        # Create new biases, one for each filter
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

        # TensorFlow operation for convolution
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # Add the biases to the results of the convolution.
        layer += biases

        return layer, weights


def new_pool_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer


def new_relu_layer(input, name):
    with tf.variable_scope(name) as scope:
        # TensorFlow operation for convolution
        layer = tf.nn.relu(input)
    return layer


def new_fc_layer(input, num_inputs, num_outputs, name):
    with tf.variable_scope(name) as scope:
        # Create new weights and biases.
        weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

        # Multiply the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        return layer


if __name__ == '__main__':
    path = "/home/kdcse/Documents/Second Semester/TIPR/Assignment-3/tipr-third-assignment"
    inputDataPath = "../data/Fashion-MNIST"
    outputDataPath = "/output"
    #trainData=[]
    #Label=[]
    '''data = input_data.read_data_sets('data/Fashion-MNIST/', one_hot=True)
    print("DataType ",type(data))
    print("Size of:")
    print("- Training-set:\t\t{}".format(len(data.train.labels)))
    print("- Test-set:\t\t{}".format(len(data.test.labels)))
    print("- Validation-set:\t{}".format(len(data.validation.labels)))'''


    kind = 'train'
    labels_path = os.path.join(inputDataPath, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(inputDataPath, '%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    images = pd.DataFrame(images)
    labels = pd.DataFrame(labels)
    Data = (images)
    Label = np.array(labels)
    LabelArray=[]
    for i in range(len(Label)):
        label=Label[i][0]
        l = [0 for j in range(10)]
        l[label] = 1
        LabelArray.append(l)

    print(np.array(Data).shape)
    print(np.array(LabelArray).shape)


    (trainData, testData, trainLabel, testLabel) = train_test_split(Data,LabelArray, test_size=0.10, random_state=42)



    # region Description
    # Placeholder variable for the input images
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28], name='X')
    # Reshape it into [num_images, img_height, img_width, num_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Placeholder variable for the true labels associated with the images
    y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # region Layer1
    # Convolutional Layer 1
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6,name="conv1")

    # Pooling Layer 1
    layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

    # RelU layer 1
    layer_relu1 = new_relu_layer(layer_pool1, name="relu1")
    # endregion

    # region Layer2
    # Convolutional Layer 2
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16,name="conv2")

    # Pooling Layer 2
    layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

    # RelU layer 2
    layer_relu2 = new_relu_layer(layer_pool2, name="relu2")
    # endregion

    # region Classification Layers
    # Flatten Layer
    num_features = layer_relu2.get_shape()[1:4].num_elements()
    layer_flat = tf.reshape(layer_relu2, [-1, num_features])

    # Fully-Connected Layer 1
    layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

    # RelU layer 3
    layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

    # Fully-Connected Layer 2
    layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")
    # endregion

    # Use Softmax function to normalize the output
    with tf.variable_scope("Softmax"):
        y_pred = tf.nn.softmax(layer_fc2)
        y_pred_cls = tf.argmax(y_pred, dimension=1)

    # Use Cross entropy cost function
    with tf.name_scope("cross_ent"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)

    # Use Adam Optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the FileWriter
    writer = tf.summary.FileWriter("Training_FileWriter/")
    writer1 = tf.summary.FileWriter("Validation_FileWriter/")

    # Add the cost and accuracy to summary
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    num_epochs = 100
    batch_size = 100
    # endregion

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Loop over number of epochs
        for epoch in range(num_epochs):

            start_time = time.time()
            y_pred_label=[]
            train_accuracy = 0
            train_f1micro=0
            train_f1macro=0
            batchstartIndex = 0
            for batch in range(0, int(len(trainLabel) / batch_size)):
                acc=0
                batchendIndex = batchstartIndex + batch_size
                x_batch=trainData[batchstartIndex:batchendIndex]
                y_true_batch=trainLabel[batchstartIndex:batchendIndex]
                #print(y_true_batch)
                batchstartIndex=batchendIndex
                # Get a batch of images and labels
                #x_batch, y_true_batch = data.train.next_batch(batch_size)

                # Put the batch into a dict with the proper names for placeholder variables
                feed_dict_train = {x: x_batch, y_true: y_true_batch}

                # Run the optimizer using this batch of training data.
                sess.run(optimizer, feed_dict=feed_dict_train)

                # Calculate the accuracy on the batch of training data
                #train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
                acc,y_cls,y_tru = sess.run([accuracy,y_pred_cls,y_true_cls],feed_dict=feed_dict_train)
                train_f1micro +=f1_score(y_tru,y_cls,average='micro')
                train_f1macro += f1_score(y_tru, y_cls, average='macro')
                #print(y_cls)
                train_accuracy +=acc
                #y_pred_label.append(y_cls)

                # Generate summary with the current batch of data and write to file
                #summ = sess.run(merged_summary, feed_dict=feed_dict_train)
                #writer.add_summary(summ, epoch * int(len(trainLabel) / batch_size) + batch)

            train_accuracy /= int(len(trainLabel) / batch_size)
            train_f1micro  /=  int(len(trainLabel) / batch_size)
            train_f1macro /= int(len(trainLabel) / batch_size)

            #fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred_label)

            # Generate summary and validate the model on the entire validation set
            val_f1micro=0
            val_f1macro=0
            vali_accuracy,val_y_cls,val_y_tru = sess.run([accuracy,y_pred_cls,y_true_cls],feed_dict={x: testData, y_true: testLabel})
            val_f1micro += f1_score(val_y_cls, val_y_tru, average='micro')
            val_f1macro += f1_score(val_y_cls, val_y_tru, average='macro')

            end_time = time.time()

            print("Epoch " + str(epoch + 1) + " completed : Time usage " + str(int(end_time - start_time)) + " seconds")
            print("\tAccuracy:")
            print("\t- Training   Accuracy:\t{}".format(train_accuracy))
            print("\t- Training   F1_Micro:\t{}".format(train_f1micro))
            print("\t- Training   F1_Macro:\t{}".format(train_f1macro))
            print("\t- Validation Accuracy:\t{}".format(vali_accuracy))
            print("\t- Validation F1_Micro:\t{}".format(val_f1micro))
            print("\t- Validation F1_Macro:\t{}".format(val_f1macro))
