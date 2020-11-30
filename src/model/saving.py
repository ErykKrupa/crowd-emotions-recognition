import os
from datetime import datetime

import numpy as np
from keras import Model
from keras.callbacks import History

from config.config import Config
from utils.logger import log_info


def save_model(model: Model, name: str = '') -> None:
    log_info('Saving model...')
    model.save(_prepare_model_path(name))
    log_info('Model saved')


def save_results(history: History, name: str = '') -> None:
    np.save(_prepare_result_path(name), history.history['val_acc'])
    log_info('Results saved')


def _prepare_model_path(name: str) -> str:
    models_directory = Config.get('models_directory')
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    return _get_path(models_directory, 'Model_', name, '.h5')


def _prepare_result_path(name: str) -> str:
    results_directory = Config.get('results_directory')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    return _get_path(results_directory, 'Result_', name, '.npy')


def _get_path(base_dir: str, file_name: str, name: str, extension: str) -> str:
    return base_dir + '/' \
           + file_name \
           + datetime.now().strftime('%y-%m-%d_%H-%M-%S') \
           + ('-' + name if name != '' else '') \
           + extension
