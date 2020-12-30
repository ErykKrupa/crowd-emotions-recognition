import os
from datetime import datetime

import numpy as np
from keras import Model
from keras.callbacks import History

from config.config import train_config as config
from utils.logger import train_log as log


def save_model(model: Model, name: str = '') -> None:
    log('Saving model...')
    model.save(_prepare_model_path(name))
    log('Model saved')


def save_results(history: History, model_name: str = '') -> None:
    np.save(_prepare_result_path('train', model_name), history.history['acc'])
    np.save(_prepare_result_path('validation', model_name), history.history['val_acc'])
    log('Results saved')


def _prepare_model_path(name: str) -> str:
    models_directory = config.get('models_directory')
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    return _get_path(models_directory, 'Model', name, '.h5')


def _prepare_result_path(file_name, model_name: str) -> str:
    results_directory = config.get('results_directory')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    return _get_path(results_directory, file_name, model_name, '.npy')


def _get_path(base_dir: str, file_name: str, name: str, extension: str) -> str:
    return base_dir + '/' \
           + file_name + '_' \
           + datetime.now().strftime('%y-%m-%d_%H-%M-%S') \
           + ('_' + name if name != '' else '') \
           + extension
