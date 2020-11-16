from datetime import datetime

from keras import Model

from config.config import MODELS_DIRECTORY


def save(model: Model) -> None:
    model.save(MODELS_DIRECTORY + datetime.now().strftime('Model_%y-%m-%d_%H-%M-%S.h5'))
