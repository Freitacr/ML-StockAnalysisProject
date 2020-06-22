import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, Activation

from multiprocessing import cpu_count


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


def trainNetwork(x_train, y_train, model: Sequential, epochs=1, verbose=1, validation_data=None):
    hist = model.fit(x_train, y_train, batch_size=20, epochs=epochs, verbose=verbose,
                     validation_data=validation_data)
    return hist


def evaluateNetwork(x_test, y_test, model: Sequential):
    return model.evaluate(x_test, y_test)