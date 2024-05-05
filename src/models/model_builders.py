# pylint: disable=import-error
from abc import ABC, abstractmethod
import platform
import uncertainty_wizard as uwiz
from keras import layers, optimizers, losses
from uncertainty_wizard.models import StochasticMode
from uncertainty_wizard.models.stochastic_utils.layers import (
    UwizBernoulliDropout,
    UwizGaussianDropout,
)


class ModelBuilderInterface(ABC):
    """Interface for building stochastic models for various datasets."""
    @abstractmethod
    def create_model(self):
        pass


class BaseModelBuilder(ModelBuilderInterface):
    """Base class to handle common functionalities for model builders."""
    def __init__(self, stochastic_mode: StochasticMode, optimizer="adadelta", learning_rate=None):
        self.stochastic_mode = stochastic_mode
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

    def select_optimizer(self):
        """Selects the appropriate optimizer based on the platform and specified preferences."""
        optimizer_classes = {
            "adam": optimizers.legacy.Adam if platform.system() == "Darwin" and platform.processor() == "arm" else optimizers.Adam,
            "sgd": optimizers.legacy.SGD if platform.system() == "Darwin" and platform.processor() == "arm" else optimizers.SGD,
            "adadelta": optimizers.legacy.Adadelta if platform.system() == "Darwin" and platform.processor() == "arm" else optimizers.Adadelta
        }
        optimizer_class = optimizer_classes.get(self.optimizer_name, optimizers.Adadelta)
        return optimizer_class() if self.learning_rate is None else optimizer_class(learning_rate=self.learning_rate)

    def compile_model(self, model):
        optimizer = self.select_optimizer()
        model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        return model


class MNISTModelBuilder(BaseModelBuilder):
    """
    This class provides utilities for working with the dataset including loading and preprocessing data
    and creating a stochastic model for classification.
    """

    def create_model(self):
        """
        Creates and compiles a stochastic CNN model for the MNIST dataset using uncertainty_wizard.
        :return: A compiled TensorFlow model
        """
        model = uwiz.models.StochasticSequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                UwizBernoulliDropout(0.5, stochastic_mode=self.stochastic_mode),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )
        return self.compile_model(model)


class Cifar10ModelBuilder(BaseModelBuilder):

    def create_model(self):
        model = uwiz.models.StochasticSequential([
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(256, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        return self.compile_model(model)


class FashionMnistModelBuilder(BaseModelBuilder):
    def create_model(self):
        model = uwiz.models.StochasticSequential(
            [
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                UwizGaussianDropout(0.5, stochastic_mode=self.stochastic_mode),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(10, activation="softmax"),
            ]
        )
        return self.compile_model(model)
