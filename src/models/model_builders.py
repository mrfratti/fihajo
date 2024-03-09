from abc import ABC, abstractmethod
import platform
import tensorflow as tf
import uncertainty_wizard as uwiz
from uncertainty_wizard.models import StochasticMode
from uncertainty_wizard.models.stochastic_utils.layers import UwizBernoulliDropout, UwizGaussianDropout


class ModelBuilderInterface(ABC):
    @abstractmethod
    def create_model(self, stochastic_mode: StochasticMode):
        pass


class MNISTModelBuilder(ModelBuilderInterface):
    """
    This class provides utilities for working with the dataset including loading and preprocessing data
    and creating a stochastic model for classification.
    """

    def __init__(self, stochastic_mode: StochasticMode, optimizer='adadelta', learning_rate=None):
        self.stochastic_mode = stochastic_mode
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

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

        optimizer = self.select_optimizer()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def select_optimizer(self):
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.legacy.Adam() if self.learning_rate is None else tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.legacy.SGD() if self.learning_rate is None else tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.Adam() if self.learning_rate is None else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.SGD() if self.learning_rate is None else tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.Adadelta()
        return opt


class Cifar10ModelBuilder(ModelBuilderInterface):
    def __init__(self, stochastic_mode: StochasticMode, optimizer='adadelta', learning_rate=None):
        self.stochastic_mode = stochastic_mode
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

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

        optimizer = self.select_optimizer()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def select_optimizer(self):
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.legacy.Adam() if self.learning_rate is None else tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.legacy.SGD() if self.learning_rate is None else tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.Adam() if self.learning_rate is None else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.SGD() if self.learning_rate is None else tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.Adadelta()
        return opt


class FashionMnistModelBuilder(ModelBuilderInterface):
    def __init__(self, stochastic_mode: StochasticMode, optimizer='adadelta', learning_rate=None):
        self.stochastic_mode = stochastic_mode
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate

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

        optimizer = self.select_optimizer()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        return model

    def select_optimizer(self):
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.legacy.Adam() if self.learning_rate is None else tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.legacy.SGD() if self.learning_rate is None else tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            if self.optimizer_name == 'adam':
                opt = tf.keras.optimizers.Adam() if self.learning_rate is None else tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            elif self.optimizer_name == 'sgd':
                opt = tf.keras.optimizers.SGD() if self.learning_rate is None else tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            else:
                opt = tf.keras.optimizers.Adadelta()
        return opt
