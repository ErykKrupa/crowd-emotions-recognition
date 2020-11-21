from os import listdir
from os.path import isfile, isdir
from typing import List

from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

from config.config import *
from config.hidden_config import PICTURE_SHAPE
from data.data_set import DataSet
from data.emotion import Emotion
from data.picture_metadata import PictureMetadata

RESCALE = 1. / 255


def get_train_generator() -> DirectoryIterator:
    data_generator: ImageDataGenerator = ImageDataGenerator(
        rescale=RESCALE,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        vertical_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE
    )
    return _get_generator(DataSet.TRAIN, data_generator)


def get_validation_generator() -> DirectoryIterator:
    return _get_generator(DataSet.VALIDATION)


# todo
def get_test_generator() -> DirectoryIterator:
    pass


def _get_generator(
        data_set: DataSet,
        data_generator: ImageDataGenerator = ImageDataGenerator(RESCALE)
) -> DirectoryIterator:
    return data_generator.flow_from_directory(
        _get_data_path(data_set),
        target_size=PICTURE_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )


def _get_data_path(
        data_set: DataSet,
        emotion: Emotion = Emotion.UNSPECIFIED,
        file_name: str = ''
) -> str:
    return DATA_DIRECTORY + '/' \
           + data_set.value + '/' \
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
    path = _get_data_path(data_set, emotion)
    if isdir(path):
        list_.extend([PictureMetadata(f, emotion)
                      for f in listdir(path)
                      if isfile(path + f) and f.endswith('.jpg')])


def get_amount_of_pictures(data_set: DataSet, emotion: Emotion = Emotion.UNSPECIFIED) -> int:
    return len(get_all_pictures_metadata(data_set, emotion))
