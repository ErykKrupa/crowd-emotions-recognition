import os
import shutil
from os.path import isfile, isdir
from typing import Tuple, Optional

import numpy as np
from keras import Model, applications

from config.config import Config
from data.data_set import DataSet
from data.preprocessing import get_amount_of_pictures, _get_generator
from utils.logger import log_info, log_model_summary
from utils.utils import get_flatten_output_shape

FEATURES = 'features'
LABELS = 'labels'


def extract_features(convolution_base: Model, data_set: DataSet) -> Tuple[np.ndarray, np.ndarray]:
    cached = _extract_from_cache(convolution_base.name, data_set)
    if cached is not None:
        return cached
    extracted = _extract_from_file(convolution_base, data_set)
    _save_to_cache(convolution_base.name, data_set, *extracted)
    return extracted


def _extract_from_cache(model_name: str, data_set: DataSet) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    log_info('Checking cache...')
    feature_path = _get_cache_path(model_name, data_set, FEATURES)
    label_path = _get_cache_path(model_name, data_set, LABELS)
    if isdir(_get_cache_path(model_name, data_set)) \
            and isfile(feature_path) \
            and isfile(label_path):
        log_info('Cache available. Extracting data...')
        data = np.load(feature_path), np.load(label_path)
        log_info('Data extracted')
        return data
    log_info('Cache not found')
    return None


def _extract_from_file(convolution_base: Model, data_set: DataSet) -> Tuple[np.ndarray, np.ndarray]:
    log_info('Extracting data from files...')
    output_shape = convolution_base.layers[-1].output_shape[1:]
    sample_count = get_amount_of_pictures(data_set)
    features = np.zeros(shape=(sample_count, *output_shape))
    labels = np.zeros(shape=(sample_count, 3))
    generator = _get_generator(data_set)
    i = 0
    batch_size = Config.get('batch_size')
    for inputs_batch, labels_batch in generator:
        features[i * batch_size: (i + 1) * batch_size] = convolution_base.predict(inputs_batch)
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    features = np.reshape(features, (sample_count, get_flatten_output_shape(convolution_base)))
    log_info('Files data extracted')
    return features, labels


def _save_to_cache(model_name: str, data_set, features: np.ndarray, labels: np.ndarray) -> None:
    path = _get_cache_path(model_name, data_set)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(_get_cache_path(model_name, data_set, FEATURES), features)
    np.save(_get_cache_path(model_name, data_set, LABELS), labels)
    log_info('Cache updated')


def fill_in_cache() -> None:
    convolution_base = get_pretrained()
    extract_features(convolution_base, DataSet.TRAIN)
    extract_features(convolution_base, DataSet.VALIDATION)


def get_pretrained() -> Model:
    model_constructor = getattr(applications, Config.get('pretrained_model'))
    model = model_constructor(weights='imagenet',
                              include_top=False,
                              input_shape=Config.get('input_shape'))
    log_model_summary(model)
    return model


def clear_whole_cache() -> None:
    shutil.rmtree(_get_cache_path(), ignore_errors=True)
    log_info('Cache cleared')


def clear_model_cache(model_name: str) -> None:
    shutil.rmtree(_get_cache_path(model_name), ignore_errors=True)
    log_info(model_name + ' cache cleared')


def _get_cache_path(
        model_name: str = '',
        data_set: DataSet = None,
        file_name: str = ''
) -> str:
    path = Config.get('extracted_data_cache_directory')
    if model_name == '':
        return path
    path += '/' + model_name
    if data_set is None:
        return path
    path += '/' + str(Config.get('picture_size')) + '/' + data_set.value
    if file_name is '':
        return path
    return path + '/' + file_name + '.npy'
