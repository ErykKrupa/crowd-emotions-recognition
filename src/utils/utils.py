from functools import reduce

from keras import Model, Sequential, models, layers


def get_flatten_output_shape(model: Model) -> int:
    return reduce(lambda a, b: a * b, model.layers[-1].output_shape[1:])


def merge(base: Model, top: Sequential) -> Sequential:
    model = models.Sequential()
    model.add(base)
    model.add(layers.Flatten())
    for layer in top.layers:
        model.add(layer)
    return model
