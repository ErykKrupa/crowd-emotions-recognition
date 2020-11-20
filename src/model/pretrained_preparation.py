from typing import Callable

from keras import Model

from config.config import INPUT_SHAPE


def get_pretrained(model_constructor: Callable) -> Model:
    return model_constructor(weights='imagenet',
                             include_top=False,
                             input_shape=INPUT_SHAPE)
