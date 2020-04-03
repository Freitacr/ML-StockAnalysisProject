import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, Activation
import numpy as np

def createModel(input_shape, num_out_categories = 3):
    model = Sequential()
    model.add(Conv2D(16, (3,3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (6,4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dense(128))
    model.add(Conv2D(32, (6,4), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(num_out_categories))
    model.add(Activation('softmax'))

    opt = keras.optimizers.RMSprop()

    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
    return model

def trainNetwork(x_train, y_train, model, num_categories=3):
    y_data_temp = np.zeros((y_train.shape[0], num_categories))
    for i in range(len(y_train)):
        index = int(y_train[i][0])
        y_data_temp[i] = np.array([0,0,0])
        y_data_temp[i][index] = 1
    y_train_true = y_data_temp
    model.fit(x_train, y_train_true, None, 1, 1)
    return model

def evaluateNetwork(x_test, y_test, model, num_categories = 3):
    y_data_temp = np.zeros((y_test.shape[0], num_categories))
    for i in range(len(y_test)):
        index = int(y_test[i][0])
        y_data_temp[i] = np.array([0,0,0])
        y_data_temp[i][index] = 1
    y_test_true = y_data_temp
    return model.evaluate(x_test, y_test_true)