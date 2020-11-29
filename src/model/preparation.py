from keras import Model, models, optimizers
from keras.layers import *

from config.config import Config
from utils.logger import log_info, log_model_summary


def prepare_whole_model() -> Model:
    kernel_shape = Config.get('kernel_shape')
    activation = Config.get('activation')
    pool_shape = Config.get('pool_shape')

    model = models.Sequential()

    model.add(Conv2D(32, kernel_shape, activation=activation, input_shape=Config.get('input_shape')))
    model.add(Conv2D(32, kernel_shape, activation=activation))
    model.add(Conv2D(32, kernel_shape, activation=activation))
    model.add(MaxPooling2D(pool_shape))
    model.add(Conv2D(64, kernel_shape, activation=activation))
    model.add(Conv2D(64, kernel_shape, activation=activation))
    model.add(Conv2D(64, kernel_shape, activation=activation))
    model.add(MaxPooling2D(pool_shape))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(MaxPooling2D(pool_shape))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(Conv2D(128, kernel_shape, activation=activation))
    model.add(MaxPooling2D(pool_shape))
    model.add(Flatten())
    model.add(Dropout(Config.get('dropout_rate')))
    model.add(Dense(512, activation=activation))
    model.add(Dense(3, activation=activation))
    _finish(model)

    return model


def prepare_end_of_model(input_dim: int) -> Model:
    model = models.Sequential()
    model.add(Dense(1024, activation=Config.get('activation'), input_dim=input_dim))
    model.add(Dropout(Config.get('dropout_rate')))
    model.add(Dense(3, activation='softmax'))
    _finish(model)

    return model


def _finish(model: Model) -> None:
    _compile(model)
    log_model_summary(model)
    log_info('Model prepared')


def _compile(model: Model) -> None:
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=Config.get('optimizer_learning_rate')),
                  metrics=['acc'])

