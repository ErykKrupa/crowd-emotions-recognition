from os import listdir
from os.path import isfile, isdir

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from config.config import Config
from data.data_set import DataSet
from data.emotion import Emotion

RESCALE = 1. / 255


def get_train_generator() -> DirectoryIterator:
    if not Config.get('data_augmentation'):
        return _get_generator(DataSet.TRAIN)
    data_generator: ImageDataGenerator = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=Config.get('rotation_range'),
        width_shift_range=Config.get('width_shift_range'),
        height_shift_range=Config.get('height_shift_range'),
        shear_range=Config.get('shear_range'),
        zoom_range=Config.get('zoom_range'),
        horizontal_flip=Config.get('horizontal_flip'),
        vertical_flip=Config.get('vertical_flip'),
        fill_mode=Config.get('fill_mode')
    )
    return _get_generator(DataSet.TRAIN, data_generator)


def get_validation_generator() -> DirectoryIterator:
    return _get_generator(DataSet.VALIDATION)


def _get_generator(
        data_set: DataSet,
        data_generator: ImageDataGenerator = ImageDataGenerator(RESCALE)
) -> DirectoryIterator:
    return data_generator.flow_from_directory(
        _get_data_path(data_set),
        target_size=Config.get('picture_shape'),
        batch_size=Config.get('batch_size'),
        class_mode='categorical'
    )


def _get_data_path(
        data_set: DataSet,
        emotion: Emotion = Emotion.UNSPECIFIED,
        file_name: str = ''
) -> str:
    return Config.get('data_directory') + '/' \
           + data_set.value + '/' \
           + (emotion.value + '/' if emotion != Emotion.UNSPECIFIED else '') \
           + file_name


def get_amount_of_pictures(data_set: DataSet, emotion: Emotion = Emotion.UNSPECIFIED) -> int:
    if emotion == Emotion.UNSPECIFIED:
        counter = 0
        for emotion in Emotion:
            counter += _get_amount_of_pictures_for_emotion(data_set, emotion)
        return counter
    else:
        return _get_amount_of_pictures_for_emotion(data_set, emotion)


def _get_amount_of_pictures_for_emotion(data_set: DataSet, emotion: Emotion) -> int:
    path = _get_data_path(data_set, emotion)
    if isdir(path):
        counter = 0
        for f in listdir(path):
            if isfile(path + f) and f.endswith('.jpg'):
                counter += 1
        return counter
