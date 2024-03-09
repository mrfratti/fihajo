from abc import ABC, abstractmethod
import platform
import tensorflow as tf
import uncertainty_wizard as uwiz
from uncertainty_wizard.models import StochasticMode
from uncertainty_wizard.models.stochastic_utils.layers import UwizBernoulliDropout, UwizGaussianDropout, \
    UwizGaussianNoise


class ModelBuilderInterface(ABC):
    @abstractmethod
    def create_model(self, stochastic_mode: StochasticMode):
        pass


class MNISTModelBuilder(ModelBuilderInterface):
    """
    This class provides utilities for working with the dataset including loading and preprocessing data
    and creating a stochastic model for classification.
    """

    def __init__(self, stochastic_mode: StochasticMode):
        self.stochastic_mode = stochastic_mode

    def create_model(self):
        """
        Creates and compiles a stochastic CNN model for the MNIST dataset using uncertainty_wizard.
        :return: A compiled TensorFlow model
        """
        model = uwiz.models.StochasticSequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            UwizBernoulliDropout(0.5, stochastic_mode=self.stochastic_mode),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            opt = tf.keras.optimizers.Adadelta()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model


class Cifar10ModelBuilder(ModelBuilderInterface):
    def __init__(self, stochastic_mode: StochasticMode):
        self.stochastic_mode = stochastic_mode

    def create_model(self):
        model = uwiz.models.StochasticSequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            UwizBernoulliDropout(0.25, stochastic_mode=self.stochastic_mode),

            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            opt = tf.keras.optimizers.legacy.Adam()
        else:
            opt = tf.keras.optimizers.Adam()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model


class FashionMnistModelBuilder(ModelBuilderInterface):
    def __init__(self, stochastic_mode: StochasticMode):
        self.stochastic_mode = stochastic_mode

    def create_model(self):
        model = uwiz.models.StochasticSequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            UwizGaussianDropout(0.5, stochastic_mode=self.stochastic_mode),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            opt = tf.keras.optimizers.Adadelta()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        return model
