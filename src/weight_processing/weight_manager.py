"""import logging for logg info, err and warning"""

import logging


import sys
import time

from src.models import model_builders


class WeightManager:
    """loading and storing weights, and including some effects for storing and loading"""

    def __init__(self) -> None:
        self._DEFAULT_PATH = "data/models/model.weights.h5"  # pylint: disable=C0103
        self._current_model = ""

    @property
    def default_path(self) -> str:
        """Returns default path for weights"""
        return self._DEFAULT_PATH

    @property
    def current_model(self) -> model_builders:
        """returns the current instace of model"""
        if self._current_model is str:
            raise ValueError("No model instance set")
        return self._current_model

    @current_model.setter
    def current_model(self, model_builder):
        if not isinstance(model_builder, model_builders.ModelBuilderInterface):
            raise ValueError("weightmanager needs an instance for model")
        self._current_model = model_builder.create_model()

    def load_weights(self, model_path=None):
        """loads the weights"""
        if not model_path:
            model_path = self._DEFAULT_PATH
        self.current_model.inner.load_weights(model_path)
        logging.info("Model weights loaded from %s", model_path)

    def loading_effect(self, duration=3, message="Evaluating"):
        """loading effect for loading weights"""
        print(message, end="")
        for _ in range(duration):
            for cursor in "|/-\\":
                sys.stdout.write(cursor)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b")
        print(" Done!")