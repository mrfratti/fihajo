import unittest
from unittest.mock import patch, MagicMock
from src.models.train import Trainer
from src.models.model_builders import MNISTModelBuilder


class TestTrainer(unittest.TestCase):
    def setUp(self):
        # Mock the model builder and dataset
        self.model_builder = MagicMock(spec=MNISTModelBuilder)
        self.model_builder.create_model.return_value = MagicMock()
        self.train_dataset = (MagicMock(), MagicMock())  # Mocked (x_train, y_train)
        self.test_dataset = (MagicMock(), MagicMock())  # Mocked (x_test, y_test)

        # Command line arguments setup
        self.args = MagicMock(
            dataset='mnist',
            adv=False,
            batch=64,
            epochs=5,
            eps=0.3,
            report=False,
            save_path=None
        )

        # Trainer initialization with mocked dependencies
        self.trainer = Trainer(self.model_builder, self.train_dataset, self.test_dataset, self.args)

    @patch('src.models.train.WeightManager')
    def test_initialization(self, mock_weight_manager):
        """ Test the initialization of the Trainer class. """
        self.assertIsNotNone(self.trainer.model)
        self.assertEqual(self.trainer.train_dataset, self.train_dataset)
        self.assertEqual(self.trainer.test_dataset, self.test_dataset)

    @patch('src.models.train.Trainer.training')
    @patch('src.models.train.Trainer.adversarial_training')
    def test_training_execution(self, mock_adversarial_training, mock_training):
        """ Test the execution of the training method. """
        self.trainer.train()
        if self.args.adv:
            mock_adversarial_training.assert_called_once()
            mock_training.assert_not_called()
        else:
            mock_training.assert_called_once()
            mock_adversarial_training.assert_not_called()

    @patch('src.models.train.VisualizeTraining')
    @patch('src.models.train.logging')
    def test_standard_training(self, mock_logging, mock_visualize):
        """Test the standard training flow of the Trainer."""
        # Setup model's fit method to return a mock history with mock data
        mock_history = MagicMock()
        mock_history.history = {'loss': [0.5, 0.4, 0.3], 'accuracy': [0.8, 0.85, 0.9]}
        self.model_builder.create_model.return_value.fit = MagicMock(return_value=mock_history)

        # Mock visualizer to avoid actual plotting
        mock_visualize_instance = mock_visualize.return_value
        mock_visualize_instance.plot_training_results = MagicMock()

        # Execute the train method
        self.trainer.train()

        # Check that fit was called
        self.model_builder.create_model.return_value.fit.assert_called_once()
        # Ensure plotting was called with the correct data
        mock_visualize_instance.plot_training_results.assert_called_once_with(mock_history)
        mock_logging.info.assert_called_with("Starting training.")

    @patch('src.models.train.logging')
    def test_training_adversarial_mode(self, mock_logging):
        """Test the adversarial training flow of the Trainer."""
        self.args.adv = True
        with patch.object(self.trainer, 'adversarial_training') as mock_adv_training:
            self.trainer.train()
            mock_adv_training.assert_called_once()

    @patch('os.makedirs')
    @patch('builtins.input', side_effect=[''])
    @patch('src.models.train.logging')
    def test_save_model_default_path(self, mock_logging, mock_input, mock_makedirs):
        """Test that save_model uses the default path when no path is provided by the user."""
        self.trainer.save_model()
        default_directory = '/'.join(self.trainer._default_save_path().split('/')[:-1])
        mock_makedirs.assert_called_with(default_directory, exist_ok=True)
        self.trainer.model.inner.save_weights.assert_called_with(self.trainer._default_save_path())

    @patch('os.makedirs')
    @patch('builtins.input', side_effect=['path/models/model.h5'])
    @patch('src.models.train.logging')
    def test_save_model_custom_path(self, mock_logging, mock_input, mock_makedirs):
        """Test that save_model uses a custom path when provided by the user."""
        self.trainer.save_model()
        expected_path = 'path/models/model.h5'
        expected_directory = 'path/models'
        mock_makedirs.assert_called_with(expected_directory, exist_ok=True)
        self.trainer.model.inner.save_weights.assert_called_with(expected_path)


if __name__ == '__main__':
    unittest.main()
