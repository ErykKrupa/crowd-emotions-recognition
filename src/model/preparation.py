from keras import Model, models, optimizers
from keras.layers import *

from config.config import ACTIVATION, DROPOUT_RATE, OPTIMIZER_LEARNING_RATE
from config.hidden_config import PICTURE_SHAPE, KERNEL_SHAPE, POOL_SHAPE


def prepare_whole_model() -> Model:
    model = models.Sequential()

    model.add(Conv2D(32, KERNEL_SHAPE, activation=ACTIVATION, input_shape=PICTURE_SHAPE + (3,)))
    model.add(Conv2D(32, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(32, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(64, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(64, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(64, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(Conv2D(128, KERNEL_SHAPE, activation=ACTIVATION))
    model.add(MaxPooling2D(POOL_SHAPE))
    model.add(Flatten())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(512, activation=ACTIVATION))
    model.add(Dense(3, activation='softmax'))
    _compile(model)
    return model


def prepare_end_of_model(input_dim: int) -> Model:
    model = models.Sequential()
    model.add(Dense(1024, activation=ACTIVATION, input_dim=input_dim))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(3, activation='softmax'))
    _compile(model)

    return model


def _compile(model: Model) -> None:
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizers.RMSprop(lr=OPTIMIZER_LEARNING_RATE),
                  metrics=['acc'])
