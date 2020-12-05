import json
from copy import deepcopy


class Config:
    # assignment after class initialization
    _DEFAULT_CONFIG: dict = None
    _current_config: dict = None

    @staticmethod
    def get(key: str):
        value = Config._current_config.get(key)
        if value is not None:
            return value
        if key == 'picture_shape':
            return Config.get('picture_size'), Config.get('picture_size')
        if key == 'input_shape':
            return Config.get('picture_shape') + (3, )
        if key == 'kernel_shape':
            return Config.get('kernel_size'), Config.get('kernel_size')
        if key == 'pool_shape':
            return Config.get('pool_size'), Config.get('pool_size')
        return None

    @staticmethod
    def set_default_config() -> None:
        Config._current_config = deepcopy(Config._DEFAULT_CONFIG)

    @staticmethod
    def edit_current_config(config: dict) -> None:
        Config._current_config.update(config)

    @staticmethod
    def set_config_based_on_default(config: dict) -> None:
        Config.set_default_config()
        Config.edit_current_config(config)


with open('config.json') as file:
    Config._DEFAULT_CONFIG = json.load(file)


Config.set_default_config()
