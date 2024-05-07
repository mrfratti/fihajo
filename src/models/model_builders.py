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
    @abstractmethod
    def create_model(self, stochastic_mode: StochasticMode):
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

        optimizer = self.select_optimizer()

        model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

    def select_optimizer(self):
        if platform.system() == "Darwin" and platform.processor() == "arm":
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.legacy.Adam()
                    if self.learning_rate is None
                    else optimizers.legacy.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.legacy.SGD()
                    if self.learning_rate is None
                    else optimizers.legacy.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.legacy.Adadelta()  # pylint: disable=E1101
        else:
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.Adam()
                    if self.learning_rate is None
                    else optimizers.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.SGD()
                    if self.learning_rate is None
                    else optimizers.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.Adadelta()
        return opt


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

        optimizer = self.select_optimizer()

        model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

    def select_optimizer(self):
        if platform.system() == "Darwin" and platform.processor() == "arm":
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.legacy.Adam()
                    if self.learning_rate is None
                    else optimizers.legacy.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.legacy.SGD()
                    if self.learning_rate is None
                    else optimizers.legacy.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.legacy.Adadelta()  # pylint: disable=E1101
        else:
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.Adam()
                    if self.learning_rate is None
                    else optimizers.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.SGD()
                    if self.learning_rate is None
                    else optimizers.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.Adadelta()
        return opt


class FashionMnistModelBuilder(ModelBuilderInterface):
    def __init__(self, stochastic_mode: StochasticMode, optimizer="adadelta", learning_rate=None):
        self.stochastic_mode = stochastic_mode
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

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

        optimizer = self.select_optimizer()

        model.compile(
            loss=losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

    def select_optimizer(self):
        if platform.system() == "Darwin" and platform.processor() == "arm":
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.legacy.Adam()
                    if self.learning_rate is None
                    else optimizers.legacy.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.legacy.SGD()
                    if self.learning_rate is None
                    else optimizers.legacy.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.legacy.Adadelta()  # pylint: disable=E1101
        else:
            if self.optimizer_name == "adam":
                opt = (
                    optimizers.Adam()
                    if self.learning_rate is None
                    else optimizers.Adam(learning_rate=self.learning_rate)
                )
            elif self.optimizer_name == "sgd":
                opt = (
                    optimizers.SGD()
                    if self.learning_rate is None
                    else optimizers.SGD(learning_rate=self.learning_rate)
                )
            else:
                opt = optimizers.Adadelta()
        return opt
