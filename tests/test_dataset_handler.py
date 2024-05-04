import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.datasets.dataset_handler import MnistDatasetHandler


class TestMnistDatasetHandler(unittest.TestCase):
    def setUp(self):
        self.handler = MnistDatasetHandler()

    @patch('keras.datasets.mnist.load_data')
    def test_load_and_preprocess(self, mock_load_data):
        # Generate random 8-bit grayscale values, excluding absolute extremes
        x_train = np.random.randint(1, 254, (60000, 28, 28), dtype=np.uint8)
        x_train[0, 0, 0] = 0  # Set the lowest pixel value to confirm normalization to 0.0
        x_train[0, 0, 1] = 255  # Set the highest pixel value to confirm normalization to 1.0
        y_train = np.random.randint(0, 10, (60000,), dtype=np.uint8)

        x_test = np.random.randint(1, 254, (10000, 28, 28), dtype=np.uint8)
        x_test[0, 0, 0] = 0  # Ensure presence of 0 in test set for normalization check
        x_test[0, 0, 1] = 255  # Ensure presence of 255 in test set for normalization check
        y_test = np.random.randint(0, 10, (10000,), dtype=np.uint8)

        mock_load_data.return_value = ((x_train, y_train), (x_test, y_test))

        # Execute the method under test
        (train_data, train_labels), (test_data, test_labels) = self.handler.load_and_preprocess()

        # Assertions to validate preprocessing
        self.assertEqual(train_data.shape, (60000, 28, 28, 1), "Train data shape incorrect")
        self.assertEqual(test_data.shape, (10000, 28, 28, 1), "Test data shape incorrect")
        self.assertTrue(np.isclose(train_data.max(), 1.0), "Train data max value should be 1.0")
        self.assertTrue(np.isclose(train_data.min(), 0.0), "Train data min value should be 0.0")
        self.assertTrue(np.isclose(test_data.max(), 1.0), "Test data max value should be 1.0")
        self.assertTrue(np.isclose(test_data.min(), 0.0), "Test data min value should be 0.0")
        self.assertEqual(train_labels.shape, (60000, 10), "Train labels shape incorrect")
        self.assertEqual(test_labels.shape, (10000, 10), "Test labels shape incorrect")
        self.assertTrue(np.all((train_labels.sum(axis=1) == 1)), "Train labels one-hot encoding failed")
        self.assertTrue(np.all((test_labels.sum(axis=1) == 1)), "Test labels one-hot encoding failed")


if __name__ == '__main__':
    unittest.main()
