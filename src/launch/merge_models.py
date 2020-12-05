from keras import Model, Sequential, models, layers


def merge(base: Model, top: Sequential) -> Sequential:
    model = models.Sequential()
    model.add(base)
    model.add(layers.Flatten())
    for layer in top.layers:
        model.add(layer)
    return model
