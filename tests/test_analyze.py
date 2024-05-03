import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.uncertainty.analyze import Analyzer
from src.models.model_builders import MNISTModelBuilder
from uncertainty_wizard.models import StochasticMode
from src.visualization.visualization import VisualizeUncertainty


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        # Set up the StochasticMode and ModelBuilder
        self.stochastic_mode = StochasticMode()
        self.model_builder = MNISTModelBuilder(self.stochastic_mode)
        # Mock the model that create_model returns
        self.model = MagicMock()
        self.model_builder.create_model = MagicMock(return_value=self.model)
        self.model.predict_quantified.return_value = [
            (np.array([0.95]), 'pcs'),
            (np.array([0.85]), 'mean_softmax'),
            (np.array([0.10]), 'predictive_entropy')
        ]
        self.model.predict.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

        # Set up the dataset
        self.dataset = ((np.array([1]), np.array([0])), (np.array([1, 2, 3]), np.array([1, 2, 3])))
        # cmd arguments setup
        self.args = MagicMock(model_path='path/to/model', batch=64, report=True)
        # Initialize the Analyzer
        self.analyzer = Analyzer(self.model_builder, self.dataset, self.args.batch, self.args)

    def test_initialization(self):
        """Test that the Analyzer initializes correctly with a model and dataset."""
        self.assertIsNotNone(self.analyzer.model)
        self.assertEqual(self.analyzer.dataset, self.dataset)

    @patch('src.uncertainty.analyze.WeightManager')
    def test_load_weights(self, mock_weight_manager):
        """Test if the Analyzer loads weights upon initialization."""
        mock_weight_manager_instance = mock_weight_manager.return_value
        Analyzer(self.model_builder, self.dataset, self.args.batch, self.args)
        mock_weight_manager_instance.load_weights.assert_called_once()

    @patch('src.visualization.visualization.VisualizeUncertainty.plot_pcs_mean_softmax')
    @patch('src.visualization.visualization.VisualizeUncertainty.plot_pcs_ms_inverse')
    @patch('src.uncertainty.analyze.Analyzer.analyze_entropy')
    @patch('src.uncertainty.analyze.Analyzer.table_generator')
    def test_analyze(self,
                     mock_table_generator,
                     mock_analyze_entropy,
                     mock_plot_pcs_ms_inverse,
                     mock_plot_pcs_mean_softmax):
        self.analyzer.analyze()
        mock_plot_pcs_mean_softmax.assert_called()
        mock_plot_pcs_ms_inverse.assert_called()
        mock_analyze_entropy.assert_called()
        mock_table_generator.assert_called()

    def test_predict_quantified(self):
        """Test predict_quantified method functionality."""
        x_test = np.array([1, 2, 3])
        self.analyzer.run_quantified(x_test)
        self.model.predict_quantified.assert_called_once_with(
            x_test,
            quantifier=["pcs", "mean_softmax", "predictive_entropy"],
            batch_size=64,
            sample_size=32,
            verbose=1
        )


if __name__ == '__main__':
    unittest.main()
