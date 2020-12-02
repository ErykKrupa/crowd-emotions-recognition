from math import ceil
from time import time

from keras import Model
from keras.callbacks import History

from config.config import Config
from data.data_set import DataSet
from data.preprocessing import get_amount_of_pictures, get_train_generator, get_validation_generator
from data.pretrained_preprocessing import extract_features, get_pretrained
from model.preparation import prepare_whole_model, prepare_end_of_model
from model.saving import save_model, save_results
from model.visualization import visualize
from utils.logger import log_info
from utils.utils import get_flatten_output_shape


def launch() -> None:
    start_time = time()
    if Config.get('use_pretrained_model'):
        _launch_pre_trained_model()
    else:
        _launch_model()
    exec_time = time() - start_time
    log_info(f'Execution time: {exec_time // 60:.0f} min {exec_time % 60:.0f} sec')


def _launch_pre_trained_model() -> None:
    log_info('Launching pretrained model...')
    convolution_base = get_pretrained()
    model = prepare_end_of_model(get_flatten_output_shape(convolution_base))
    train_data = extract_features(convolution_base, DataSet.TRAIN)
    validation_data = extract_features(convolution_base, DataSet.VALIDATION)
    log_info('Training...')
    history = model.fit(
        *train_data,
        epochs=Config.get('epochs'),
        batch_size=Config.get('batch_size'),
        validation_data=validation_data
    )
    _finish(history, model, convolution_base.name)


def _launch_model() -> None:
    log_info('Launching model...')
    model = prepare_whole_model()
    train_generator = get_train_generator()
    validation_generator = get_validation_generator()
    log_info('Training...')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=ceil(get_amount_of_pictures(DataSet.TRAIN) / Config.get('batch_size')),
        epochs=Config.get('epochs'),
        validation_data=validation_generator,
        validation_steps=ceil(get_amount_of_pictures(DataSet.VALIDATION) / Config.get('batch_size'))
    )
    _finish(history, model)


def _finish(history: History, model: Model, name: str = '') -> None:
    visualize(history, name)
    save_model(model, name)
    save_results(history, name)
