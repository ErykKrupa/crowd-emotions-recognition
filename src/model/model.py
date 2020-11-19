from keras import models, Model
from keras.layers import *
from tensorflow_core import optimizers

from config.config import *


def prepare_model() -> Model:
    model = models.Sequential()
    model.add(Conv2D(32, KERNEL_SIZE, activation=ACTIVATION, input_shape=PICTURE_SIZE + (3,)))
    model.add(MaxPooling2D(POOL_SIZE))
    model.add(Conv2D(64, KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SIZE))
    model.add(Conv2D(128, KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SIZE))
    model.add(Conv2D(128, KERNEL_SIZE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SIZE))
    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(512, activation=ACTIVATION))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.RMSprop(lr=OPTIMIZER_LEARNING_RATE),
                  metrics=['acc'])
    return model
