import os
import shutil
from os.path import isfile, isdir
from typing import Tuple, Optional, Callable

import numpy as np
from keras import Model

from config.config import BATCH_SIZE, EXTRACTED_DATA_CACHE_DIRECTORY, PICTURE_SIZE
from config.hidden_config import INPUT_SHAPE
from data.data_set import DataSet
from data.preprocessing import get_amount_of_pictures, _get_generator
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
    feature_path = _get_cache_path(model_name, data_set, FEATURES)
    label_path = _get_cache_path(model_name, data_set, LABELS)
    if isdir(_get_cache_path(model_name, data_set)) \
            and isfile(feature_path) \
            and isfile(label_path):
        return np.load(feature_path), np.load(label_path)
    return None


def _extract_from_file(convolution_base: Model, data_set: DataSet) -> Tuple[np.ndarray, np.ndarray]:
    output_shape = convolution_base.layers[-1].output_shape[1:]
    sample_count = get_amount_of_pictures(data_set)
    features = np.zeros(shape=(sample_count, *output_shape))
    labels = np.zeros(shape=(sample_count, 3))
    generator = _get_generator(data_set)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = convolution_base.predict(inputs_batch)
        features[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = features_batch
        labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] = labels_batch
        i += 1
        if i * BATCH_SIZE >= sample_count:
            break
    features = np.reshape(features, (sample_count, get_flatten_output_shape(convolution_base)))
    return features, labels


def _save_to_cache(model_name: str, data_set, features: np.ndarray, labels: np.ndarray) -> None:
    path = _get_cache_path(model_name, data_set)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(_get_cache_path(model_name, data_set, FEATURES), features)
    np.save(_get_cache_path(model_name, data_set, LABELS), labels)


def fill_in_cache(model_constructor: Callable) -> None:
    convolution_base = get_pretrained(model_constructor)
    extract_features(convolution_base, DataSet.TRAIN)
    extract_features(convolution_base, DataSet.VALIDATION)


def get_pretrained(model_constructor: Callable) -> Model:
    return model_constructor(weights='imagenet',
                             include_top=False,
                             input_shape=INPUT_SHAPE)


def clear_whole_cache() -> None:
    shutil.rmtree(_get_cache_path(), ignore_errors=True)


def clear_model_cache(model_name: str) -> None:
    shutil.rmtree(_get_cache_path(model_name), ignore_errors=True)


def _get_cache_path(
        model_name: str = '',
        data_set: DataSet = None,
        file_name: str = ''
) -> str:
    path = EXTRACTED_DATA_CACHE_DIRECTORY
    if model_name == '':
        return path
    path += '/' + model_name
    if data_set is None:
        return path
    path += '/' + str(PICTURE_SIZE) + '/' + data_set.value
    if file_name is '':
        return path
    return path + '/' + file_name + '.npy'
