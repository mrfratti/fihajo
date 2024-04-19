"""import logging for logg info, err and warning"""

import logging

import sys
import time

from src.models import model_builders


class WeightManager:
    """loading and storing weights, and including some effects for storing and loading"""

    def __init__(self) -> None:
        self._DEFAULT_PATH = "data/models/model.weights.h5"  # pylint: disable=C0103
        self._model_path = ""
        self._current_model = ""

    @property
    def default_path(self) -> str:
        """Returns default path for weights"""
        return self._DEFAULT_PATH

    @property
    def model_path(self):
        if self._model_path is None or len(self._model_path) < 1:
            return self._DEFAULT_PATH
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        if path is None or len(path) < 1 or not isinstance(path, str):
            self._model_path = self._DEFAULT_PATH
        self._model_path = path

    @property
    def current_model(self) -> model_builders:
        """returns the current instace of model"""
        if self._current_model is None or isinstance(self._current_model, str):
            raise ValueError("No model instance set")
        return self._current_model

    @current_model.setter
    def current_model(self, model_builder):
        if not isinstance(model_builder, model_builders.ModelBuilderInterface):
            raise ValueError("weightmanager needs an instance for model")
        self._current_model = model_builder.create_model()

    def load_weights(self):
        """loads the weights"""
        try:
            self.current_model.inner.load_weights(self.model_path)
            logging.info("Model weights loaded from %s", self.model_path)
        except FileNotFoundError:
            logging.error("The specified weight file was not found: %s", self.model_path)
            sys.exit(1)
        except PermissionError:
            logging.error("Cannot open weight file missing permission to read")
            sys.exit(1)
        except IsADirectoryError:
            logging.error("Please specify a file instead of a directory")
            sys.exit(1)

    def loading_effect(self, duration=0.1, message="Evaluating"):
        """loading effect for loading weights"""
        print("\n" + message, end="")
        for _ in range(duration):
            for cursor in "|/-\\":
                sys.stdout.write(cursor)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b")
        print(" Done!")
