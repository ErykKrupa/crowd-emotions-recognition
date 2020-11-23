import os
from datetime import datetime

import numpy as np
from keras import Model
from keras.callbacks import History

from config.config import MODELS_DIRECTORY, RESULTS_DIRECTORY


def save_model(model: Model, name: str = '') -> None:
    model.save(_prepare_model_path(name))


def save_results(history: History, name: str = '') -> None:
    np.save(_prepare_result_path(name), history.history['val_acc'])


def _prepare_model_path(name: str) -> str:
    if not os.path.exists(MODELS_DIRECTORY):
        os.makedirs(MODELS_DIRECTORY)
    return _get_path(MODELS_DIRECTORY, 'Model_', name, '.h5')


def _prepare_result_path(name: str) -> str:
    if not os.path.exists(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)
    return _get_path(RESULTS_DIRECTORY, 'Result_', name, '.npy')


def _get_path(base_dir: str, file_name: str, name: str, extension: str) -> str:
    return base_dir + '/' \
           + file_name \
           + datetime.now().strftime('%y-%m-%d_%H-%M-%S') \
           + ('-' + name if name != '' else '') \
           + extension
