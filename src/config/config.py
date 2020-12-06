import json
from copy import deepcopy


class Config:
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._DEFAULT_CONFIG = None
        self._current_config = None

    def _create(self):
        with open(self._file_name) as file:
            self._DEFAULT_CONFIG = json.load(file)
        self.set_default_config()

    def get(self, key: str):
        if not self._DEFAULT_CONFIG:
            self._create()
        value = self._current_config.get(key)
        if value is not None:
            return value
        if key == 'picture_shape':
            return self.get('picture_size'), self.get('picture_size')
        if key == 'input_shape':
            return self.get('picture_shape') + (3, )
        if key == 'kernel_shape':
            return self.get('kernel_size'), self.get('kernel_size')
        if key == 'pool_shape':
            return self.get('pool_size'), self.get('pool_size')
        return None

    def set_default_config(self) -> None:
        if not self._DEFAULT_CONFIG:
            self._create()
        self._current_config = deepcopy(self._DEFAULT_CONFIG)

    def edit_current_config(self, config: dict) -> None:
        if not self._DEFAULT_CONFIG:
            self._create()
        self._current_config.update(config)

    def set_config_based_on_default(self, config: dict) -> None:
        if not self._DEFAULT_CONFIG:
            self._create()
        self.set_default_config()
        self.edit_current_config(config)


train_config = Config('train_config.json')
predict_config = Config('predict_config.json')
