import unittest
import numpy as np

from unittest.mock import patch
from keras.datasets import mnist
from src.datasets.dataset_handler import MnistDatasetHandler


class TestMnistDatasetHandler(unittest.TestCase):
    @patch('keras.datasets.mnist.load_data')
    def test_load_and_preprocess(self, mock_load_data):
        mock_load_data.return_value = ((np.zeros((60000, 28, 28)), np.zeros(60000)),
                                       (np.zeros((10000, 28, 28)), np.zeros(10000)))

        handler = MnistDatasetHandler()
        (x_train, y_train), (x_test, y_test) = handler.load_and_preprocess()

        self.assertEqual(x_train.shape, (60000, 28, 28, 1))
        self.assertEqual(x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(x_train.max(), 0.0)


if __name__ == '__main__':
    unittest.main()
