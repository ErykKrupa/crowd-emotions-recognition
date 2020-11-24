from keras import Model

from config.config import Config


def log_info(message) -> None:
    if Config.get('info_log'):
        print(message)


def log_model_summary(model: Model) -> None:
    if Config.get('model_summary_log'):
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        print('\n'.join(summary))
