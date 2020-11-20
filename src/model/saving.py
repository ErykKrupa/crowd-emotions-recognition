from datetime import datetime

from keras import Model

from config.config import MODELS_DIRECTORY


def save(model: Model, info: str = "") -> None:
    model.save(MODELS_DIRECTORY + '/'
               + 'Model_'
               + datetime.now().strftime('%y-%m-%d_%H-%M-%S')
               + ('-' + info if info != '' else '')
               + '.h5')
