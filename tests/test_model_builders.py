import unittest
from ddt import ddt, data, unpack
from unittest.mock import patch
import tensorflow as tf
from keras import layers
from src.models.model_builders import MNISTModelBuilder, Cifar10ModelBuilder, FashionMnistModelBuilder
from uncertainty_wizard.models import StochasticMode

@ddt
class TestModelBuilders(unittest.TestCase):

    def setUp(self):
        self.stochastic_mode = StochasticMode()

    @data(
        ('MNIST', MNISTModelBuilder, 7, layers.Conv2D, (None, 28, 28, 1)),
        ('CIFAR10', Cifar10ModelBuilder, 14, layers.Conv2D, (None, 32, 32, 3)),
        ('FashionMNIST', FashionMnistModelBuilder, 7, layers.Conv2D, (None, 28, 28, 1))
    )
    @unpack
    def test_model_creation(self, name, builder_class, num_layers, first_layer_type, input_shape):
        if name == 'CIFAR10':
            builder = builder_class(self.stochastic_mode, 'adam', 0.001)
            model = builder.create_model(self.stochastic_mode)
        else:
            builder = builder_class(self.stochastic_mode, 'adam', 0.001)
            model = builder.create_model()
        self.assertEqual(len(model.inner.layers), num_layers, f"{name} model should have {num_layers} layers.")
        self.assertIsInstance(model.inner.layers[0], first_layer_type,
                              f"{name} model's first layer should be {first_layer_type.__name__}.")
        self.assertEqual(model.inner.layers[0].input_shape, input_shape,
                         f"{name} model's first layer should accept input shape {input_shape}.")

    @patch('platform.system', return_value="Darwin")
    @patch('platform.processor', return_value="arm")
    def test_optimizer_selection_arm(self, mock_processor, mock_system):
        builder = MNISTModelBuilder(self.stochastic_mode, 'adam', 0.001)
        optimizer = builder.select_optimizer()
        self.assertIsInstance(optimizer, tf.keras.optimizers.legacy.Adam, "ARM should use legacy Adam optimizer.")


if __name__ == '__main__':
    unittest.main()
