"""pylint"""

from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import to_categorical


class DatasetHandler:
    def load_and_preprocess(self):
        raise NotImplementedError("Subclasses must implement this method")


class MnistDatasetHandler(DatasetHandler):
    def load_and_preprocess(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype("float32") / 255).reshape(x_train.shape[0], 28, 28, 1)
        x_test = (x_test.astype("float32") / 255).reshape(x_test.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)
        return (x_train, y_train), (x_test, y_test)


class Cifar10DatasetHandler(DatasetHandler):
    def load_and_preprocess(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        return (x_train, y_train), (x_test, y_test)


class FashionMnistDatasetHandler(DatasetHandler):
    def load_and_preprocess(self):
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        return (x_train, y_train), (x_test, y_test)
