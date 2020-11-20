from functools import reduce

from keras import Model


def get_flatten_output_shape(model: Model) -> int:
    return reduce(lambda a, b: a * b, model.layers[-1].output_shape[1:])
