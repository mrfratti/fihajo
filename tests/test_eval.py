import unittest
from unittest.mock import patch, MagicMock
from src.models.eval import Evaluator
from src.models.model_builders import MNISTModelBuilder


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock the model builder and dataset
        self.model_builder = MagicMock(spec=MNISTModelBuilder)
        self.model_builder.create_model.return_value = MagicMock()
        self.dataset = (MagicMock(), (MagicMock(), MagicMock()))  # Mocked (x_test, y_test)

        # cmd arguments setup
        self.args = MagicMock(
            model_path='path/to/model',
            adv_eval=False,
            eps=0.3
        )

        # Evaluator initialization with mocked dependencies
        self.evaluator = Evaluator(self.model_builder, self.dataset, self.args)

    @patch('src.models.eval.WeightManager')
    def test_initialization(self, mock_weight_manager):
        """ Test the initialization of the Evaluator class. """
        self.assertIsNotNone(self.evaluator.model)
        self.assertEqual(self.evaluator.dataset, self.dataset)

    @patch('src.models.eval.VisualizeEvaluation')
    @patch('src.models.eval.logging')
    def test_evaluation(self, mock_logging, mock_visualize):
        """ Test the standard evaluation flow of the Evaluator. """
        # Setup model's evaluate and predict methods
        self.model_builder.create_model.return_value.evaluate = MagicMock(return_value=(0.5, 0.9))
        self.model_builder.create_model.return_value.predict = MagicMock(return_value=MagicMock())

        # Mock visualizer to avoid actual plotting
        mock_visualize_instance = mock_visualize.return_value
        mock_visualize_instance.plot_predictions = MagicMock()
        mock_visualize_instance.plot_confusion_matrix = MagicMock()
        mock_visualize_instance.plot_classification_report = MagicMock()

        # Execute the evaluation method
        self.evaluator.evaluate()

        # Check that evaluate was called
        self.model_builder.create_model.return_value.evaluate.assert_called()
        mock_logging.info.assert_called()

    @patch('src.models.eval.fast_gradient_method', return_value=MagicMock())
    @patch('src.models.eval.projected_gradient_descent', return_value=MagicMock())
    @patch('src.models.eval.VisualizeEvaluation')
    @patch('src.models.eval.logging')
    def test_adversarial_evaluation(self, mock_logging, mock_visualize, mock_pgd, mock_fgm):
        """ Test the adversarial evaluation flow of the Evaluator. """
        # Setup model's evaluate and predict methods
        self.model_builder.create_model.return_value.evaluate = MagicMock(return_value=(0.5, 0.9))
        self.model_builder.create_model.return_value.predict = MagicMock(return_value=MagicMock())

        # Enabling adversarial evaluation
        self.args.adv_eval = True

        # Execute the adversarial evaluation method
        self.evaluator.evaluate()

        # Verify adversarial methods were called
        mock_fgm.assert_called_once()
        mock_pgd.assert_called_once()

        mock_visualize.return_value.plot_adversarial_examples.assert_called()
        mock_visualize.return_value.plot_accuracy_comparison.assert_called()

        mock_logging.info.assert_called()


if __name__ == '__main__':
    unittest.main()
