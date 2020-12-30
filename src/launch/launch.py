from os import environ

from config.config import train_config as config

environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if config.get('tensorflow_log') else '3'

from math import ceil
from time import time

from keras import Model
from keras.callbacks import History

from data.data_set import DataSet
from data.preprocessing import get_amount_of_pictures, get_train_generator, get_validation_generator
from data.pretrained_preprocessing import extract_features, get_pretrained
from utils.utils import merge
from model.preparation import prepare_whole_model, prepare_end_of_model
from model.saving import save_model, save_results
from model.visualization import visualize
from utils.logger import train_log as log
from utils.utils import get_flatten_output_shape


def launch() -> None:
    start_time = time()
    if config.get('use_pretrained_model'):
        _launch_pre_trained_model()
    else:
        _launch_model()
    exec_time = time() - start_time
    log(f'Execution time: {exec_time // 60:.0f} min {exec_time % 60:.0f} sec')


def _launch_pre_trained_model() -> None:
    log('Launching pretrained model...')
    convolution_base = get_pretrained()
    model = prepare_end_of_model(get_flatten_output_shape(convolution_base))
    train_data = extract_features(convolution_base, DataSet.TRAIN)
    validation_data = extract_features(convolution_base, DataSet.VALIDATION)
    log('Training...')
    history = model.fit(
        *train_data,
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size'),
        validation_data=validation_data
    )
    _finish(history, merge(convolution_base, model), convolution_base.name)


def _launch_model() -> None:
    log('Launching model...')
    model = prepare_whole_model()
    train_generator = get_train_generator()
    validation_generator = get_validation_generator()
    log('Training...')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=ceil(get_amount_of_pictures(DataSet.TRAIN) / config.get('batch_size')),
        epochs=config.get('epochs'),
        validation_data=validation_generator,
        validation_steps=ceil(get_amount_of_pictures(DataSet.VALIDATION) / config.get('batch_size'))
    )
    _finish(history, model)


def _finish(history: History, model: Model, name: str = '') -> None:
    visualize(history, name)
    save_model(model, name)
    save_results(history, name)
