from os import listdir
from os.path import isfile, isdir
from typing import List

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from config.config import *
from data.data_set import DataSet
from data.emotion import Emotion
from data.picture_metadata import PictureMetadata

_data_generator = ImageDataGenerator(1. / 255)


def get_train_generator() -> DirectoryIterator:
    return _get_generator(DataSet.TRAIN, TRAIN_BATCH_SIZE)


def get_validation_generator() -> DirectoryIterator:
    return _get_generator(DataSet.VALIDATION, VALIDATION_BATCH_SIZE)


# todo
# def get_test_generator(target_size: int, batch_size: int) -> DirectoryIterator:


def _get_generator(data_set: DataSet, batch_size: int) -> DirectoryIterator:
    return _data_generator.flow_from_directory(
        _get_path(data_set),
        target_size=PICTURE_SIZE,
        batch_size=batch_size,
        class_mode='categorical'
    )


def _get_path(
        data_set: DataSet,
        emotion: Emotion = Emotion.UNSPECIFIED,
        file_name: str = ''
) -> str:
    return '../' + DATA_DIRECTORY + data_set.value + '/' \
           + (emotion.value + '/' if emotion != Emotion.UNSPECIFIED else '') \
           + file_name


def get_all_pictures_metadata(
        data_set: DataSet,
        emotion: Emotion = Emotion.UNSPECIFIED
) -> List[PictureMetadata]:
    list_ = []
    if emotion == Emotion.UNSPECIFIED:
        for emotion in Emotion:
            _add_pictures_metadata_to_set(list_, data_set, emotion)
    else:
        _add_pictures_metadata_to_set(list_, data_set, emotion)
    return list_


def _add_pictures_metadata_to_set(list_: list, data_set: DataSet, emotion: Emotion) -> None:
    path = _get_path(data_set, emotion)
    if isdir(path):
        list_.extend([PictureMetadata(f, emotion)
                      for f in listdir(path)
                      if isfile(path + f) and f.endswith('.jpg')])


def get_amount_of_pictures(data_set: DataSet, emotion: Emotion = Emotion.UNSPECIFIED) -> int:
    return len(get_all_pictures_metadata(data_set, emotion))
