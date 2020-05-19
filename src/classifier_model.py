from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, charset, should_model_be_saved=True):
        self.model = None
        self.charset = charset
        self.should_model_be_saved = should_model_be_saved

    @abstractmethod
    def train(self, generator, epochs, callbacks=None, start_epoch=0):
        pass

    @abstractmethod
    def _save_model(self):
        pass
