from abc import ABCMeta, abstractmethod


class feature_selector:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.best_features = None

    @abstractmethod
    def fit(self, data, y):
        pass

    def get_features(self):
        pass


class model:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, data, y):
        pass

    @abstractmethod
    def predict(self, data):
        pass


class online:
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, data, y):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def update_model(self, data, y):
        pass
