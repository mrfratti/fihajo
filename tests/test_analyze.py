import unittest
from unittest.mock import patch, MagicMock
from src.uncertainty.analyze import Analyzer
from src.models.model_builders import MNISTModelBuilder, ModelBuilderInterface


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        # Mock the model builder as a subclass of ModelBuilderInterface
        self.model_builder = MagicMock(spec=ModelBuilderInterface)
        mock_model = MagicMock()
        self.model_builder.create_model.return_value = mock_model

        # Ensure predict_quantified and other model methods are ready to be called
        mock_model.predict_quantified = MagicMock()
        mock_model.predict = MagicMock(return_value=MagicMock())
        mock_model.inner = MagicMock()
        mock_model.inner.load_weights = MagicMock()
        self.dataset = (MagicMock(), (MagicMock(), MagicMock()))  # Mocked (x_test, y_test)

        # cmd arguments setup
        self.args = MagicMock(
            model_path='path/to/model',
            batch=64,
            report=False
        )

        # Analyzer initialization with mocked dependencies
        self.analyzer = Analyzer(self.model_builder, self.dataset, self.args.batch, self.args)

    def test_initialization(self):
        """ Test the initialization of the Analyzer class. """
        self.assertIsNotNone(self.analyzer.model)
        self.assertEqual(self.analyzer.dataset, self.dataset)
        self.assertEqual(self.analyzer.batch, 64)

    @patch('src.uncertainty.analyze.WeightManager')
    def test_load_weights(self, mock_weight_manager):
        """ Test that load_weights is correctly invoked. """
        # Setup
        weight_manager_instance = mock_weight_manager.return_value
        weight_manager_instance.model_path = 'path/to/model'
        weight_manager_instance.current_model = self.model_builder.create_model()
        weight_manager_instance.load_weights = MagicMock()

        # Test
        self.analyzer._weightmanager = weight_manager_instance
        self.analyzer._weightmanager.load_weights()
        weight_manager_instance.load_weights.assert_called_once()

    @patch('src.uncertainty.analyze.Analyzer.analyze')
    def test_analyze_execution(self, mock_analyze):
        """ Test the execution of the analyze method. """
        self.analyzer.analyze()
        mock_analyze.assert_called_once()

    @patch('src.uncertainty.analyze.Analyzer.run_quantified')
    @patch('src.uncertainty.analyze.Analyzer.pcs_mean_softmax')
    @patch('src.uncertainty.analyze.Analyzer.analyze_entropy')
    @patch('src.uncertainty.analyze.Analyzer.table_generator')
    def test_analyze(self, mock_table_generator, mock_analyze_entropy, mock_pcs_mean_softmax, mock_run_quantified):
        """ Test the analyze method runs all components. """
        self.analyzer.analyze()
        mock_run_quantified.assert_called_once()
        mock_pcs_mean_softmax.assert_called_once()
        mock_analyze_entropy.assert_called_once()
        mock_table_generator.assert_called_once()

    def test_predict_quantified(self):
        """ Test predict_quantified method of the Analyzer class. """
        x_test = MagicMock()
        self.analyzer.run_quantified(x_test)
        self.analyzer.model.predict_quantified.assert_called_once_with(
            x_test,
            quantifier=["pcs", "mean_softmax", "predictive_entropy"],
            batch_size=64,
            sample_size=32,
            verbose=1
        )


if __name__ == '__main__':
    unittest.main()
