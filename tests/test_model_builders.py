# pylint: skip-file
import unittest
from unittest.mock import patch
import tensorflow as tf
from keras import layers
from src.models.model_builders import MNISTModelBuilder, Cifar10ModelBuilder, FashionMnistModelBuilder
from uncertainty_wizard.models import StochasticMode

class TestModelBuilders(unittest.TestCase):
    def setUp(self):
        self.stochastic_mode = StochasticMode()

    def test_mnist_model_creation(self):
        builder = MNISTModelBuilder(self.stochastic_mode, 'adam', 0.001)
        model = builder.create_model()
        self.assertIsInstance(model.inner, tf.keras.models.Sequential)
        self.assertEqual(len(model.inner.layers), 7)
        self.assertIsInstance(model.inner.layers[0], layers.Conv2D)

    def test_cifar10_model_creation(self):
        builder = Cifar10ModelBuilder(self.stochastic_mode, 'sgd', 0.01)
        model = builder.create_model(stochastic_mode=StochasticMode)
        self.assertIsInstance(model.inner, tf.keras.models.Sequential)
        self.assertEqual(len(model.inner.layers), 14)
        self.assertEqual(model.inner.layers[0].output_shape, (None, 32, 32, 16))
        self.assertEqual(model.inner.layers[-1].output_shape, (None, 10))

    def test_fashion_mnist_model_creation(self):
        builder = FashionMnistModelBuilder(self.stochastic_mode, 'adadelta', 0.1)
        model = builder.create_model()
        self.assertIsInstance(model.inner, tf.keras.models.Sequential)
        self.assertEqual(len(model.inner.layers), 7)
        self.assertIsInstance(model.inner.layers[0], layers.Conv2D)

    @patch('platform.system')
    @patch('platform.processor')
    def test_optimizer_selection_arm(self, mock_processor, mock_system):
        mock_system.return_value = "Darwin"
        mock_processor.return_value = "arm"
        builder = MNISTModelBuilder(self.stochastic_mode, 'adam', 0.001)
        optimizer = builder.select_optimizer()
        self.assertIsInstance(optimizer, tf.keras.optimizers.legacy.Adam)

    @patch('platform.system')
    @patch('platform.processor')
    def test_optimizer_selection_non_arm(self, mock_processor, mock_system):
        mock_system.return_value = "Linux"
        mock_processor.return_value = "x86_64"
        builder = Cifar10ModelBuilder(self.stochastic_mode, 'sgd', 0.01)
        optimizer = builder.select_optimizer()
        self.assertIsInstance(optimizer, tf.keras.optimizers.SGD)

if __name__ == '__main__':
    unittest.main()
