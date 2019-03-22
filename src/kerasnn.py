from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation, Flatten
import keras
import numpy as np


def MLP(X,Y,X_test,Y_tst,opdim,batchsize,epoc,datasetname):
    np.random.seed(7)
    x_train=X
    y_train = keras.utils.to_categorical(Y, 10)
    x_test=X_test
    y_test = keras.utils.to_categorical(Y_tst, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255


    if(datasetname=="Fashion-MNIST"):
        x_train = x_train.reshape(len(x_train), 28, 28)
        x_test = x_test.reshape(10000, 28, 28)

        # create model
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(256, activation='tanh', kernel_initializer='he_normal', input_shape=(28 * 28,)))
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='tanh', kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='tanh', kernel_initializer='he_normal'))
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='sigmoid', kernel_initializer='he_normal'))
        optim = keras.optimizers.SGD(lr=0.01, momentum=0.975, decay=2e-06, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train,
                            batch_size=batchsize,
                            epochs=30,
                            verbose=2,
                            validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test top 1 accuracy:', score[1])
    else:
        x_train = x_train.reshape(50000, 32 * 32 * 3)
        x_test = x_test.reshape(10000, 32 * 32 * 3)
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()

        # training
        history = model.fit(x_train, y_train,
                            batch_size=batchsize,
                            nb_epoch=epoc,
                            verbose=1,
                            validation_data=(x_test, y_test))

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', loss)
        print('Test acc:', acc)