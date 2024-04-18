"""import logging for logg info, err and warning"""

import logging


import sys
import time

from src.cli.string_styling import StringStyling
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
        if self._current_model is None or isinstance(self._current_model, str):
            raise ValueError("No model instance set")
        return self._current_model

    @current_model.setter
    def current_model(self, model_builder):
        if not isinstance(model_builder, model_builders.ModelBuilderInterface):
            raise ValueError("weightmanager needs an instance for model")
        self._current_model = model_builder.create_model()

    def load_weights(self, model_path=None):
        """loads the weights"""
        try:
            if model_path is None:
                model_path = self._DEFAULT_PATH
            self.current_model.inner.load_weights(model_path)
            logging.info("Model weights loaded from %s", model_path)
        except FileNotFoundError:
            print(
                StringStyling.box_style(
                    "The specified weight file was not found: %s", model_path
                )
            )
            sys.exit(1)
        except PermissionError:
            print(
                StringStyling.box_style(
                    "Cannot open weight file missing permission to read"
                )
            )
            sys.exit(1)
        except IsADirectoryError:
            print(StringStyling("Please specify a file instead of a directory"))
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
