from keras import Model

from config.config import train_config, predict_config


def train_log(message) -> None:
    if train_config.get('info_log'):
        print(message)


def predict_log(message) -> None:
    if predict_config.get('info_log'):
        print(message)


def log_model_summary(model: Model) -> None:
    if train_config.get('model_summary_log'):
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        print('\n'.join(summary))
