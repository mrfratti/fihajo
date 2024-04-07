import argparse
import logging
import os.path
import datetime
import numpy as np
import pandas as pd
from src.visualization.visualization import VisualizeUncertainty
from src.weight_processing.weight_manager import WeightManager


class Analyzer:
    """
    The Analyzer class is responsible for analyzing a trained model.
    It quantifies uncertainty using various metrics and visualizes them.
    """

    def __init__(self, model_builder, dataset, batch: int, args: argparse.Namespace):
        """
        Initializes the Analyzer with the model path and batch size.
        :param model_path: Path to the trained model weights.
        :param batch: Batch size for processing data.
        """
        self.args = args
        self.dataset = dataset
        self.batch = batch
        self._weightmanager = WeightManager()
        self._weightmanager.current_model = model_builder
        self._weightmanager.load_weights(args.model_path)
        self.model = self._weightmanager.current_model
        self.quantified_results = None
        self.pcs_mean_softmax_scores = None
        self.entropy_scores = None
        self.mean_softmax_scores = None
        self.pcs_scores = None
        self._plot_file_names = {}

    @property
    def default_path(self) -> str:
        """returns the default path of weights storing location"""
        return self._weightmanager.default_path

    def analyze(self):
        """
        Runs the complete analysis, including quantified uncertainty analysis,
        PCS and mean softmax calculation, entropy analysis, and table generation.
        """
        _, (x_test, y_test) = self.dataset

        self._weightmanager.loading_effect(duration=3, message="Loading model weights")
        self.run_quantified(x_test)
        self.pcs_mean_softmax()
        self.analyze_entropy(x_test)
        self.table_generator(x_test, y_test)

    def run_quantified(self, x_test):
        """
        Performs quantified predictions using the model to obtain both confidence and uncertainty metrics.
        :param x_test: (np.array): Test dataset inputs.
        """
        if self.quantified_results is None:
            quantifiers = ["pcs", "mean_softmax", "predictive_entropy"]
            self.quantified_results = self.model.predict_quantified(
                x_test,
                quantifier=quantifiers,
                batch_size=self.batch,
                sample_size=32,
                verbose=1,
            )

            # Extract the results from the quantified tuples
            self.pcs_scores = self.quantified_results[0][1]
            self.mean_softmax_scores = self.quantified_results[1][1]
            self.pcs_mean_softmax_scores = self.pcs_scores, self.mean_softmax_scores
            self.entropy_scores = self.quantified_results[2][1]

    def pcs_mean_softmax(self):
        """
        Visualizes the PCS and mean softmax scores distribution and their relationships.
        """
        visualizer = VisualizeUncertainty()
        visualizer.plot_pcs_mean_softmax(self.pcs_mean_softmax_scores)
        visualizer.plot_distribution_pcs_ms_scores(self.pcs_mean_softmax_scores)
        visualizer.plot_pcs_ms_inverse(self.pcs_mean_softmax_scores)
        self._plot_file_names.update(visualizer.plot_file_names)

    def analyze_entropy(self, x_test):
        """
        Analyzes and visualizes the entropy of model predictions alongside other metrics.
        :param x_test: (np.array): Test dataset inputs.
        """
        predictive_confidence = np.max(self.model.predict(x_test), axis=1)

        # Compute statistics for the entropy scores
        print("Statistics for Entropy scores:")
        print("Mean:", np.mean(self.entropy_scores))
        print("Median:", np.median(self.entropy_scores))
        print("Standard Deviation:", np.std(self.entropy_scores))
        print("Min:", np.min(self.entropy_scores))
        print("Max:", np.max(self.entropy_scores))

        # Print the average confidence score
        print("\nAverage confidence score:", np.mean(predictive_confidence))

        # Print confidence and entropy for the most uncertain samples
        print("\nMost uncertain samples:")
        for i in np.argsort(self.entropy_scores)[-5:]:
            print(
                f"Entropy: {self.entropy_scores[i]:.2f}, Confidence: {predictive_confidence[i]:.2f}"
            )

        # Print confidence and entropy for the first 5 samples
        print("\nConfidence and entropy score for the first 5 samples:")
        for i in range(5):
            print(
                f"Entropy: {self.entropy_scores[i]:.2f}, Confidence: {predictive_confidence[i]:.2f}"
            )

        visualizer = VisualizeUncertainty()
        visualizer.plot_dist_entropy_scores(self.entropy_scores)
        visualizer.high_uncertain_inputs(self.entropy_scores, x_test, num_samples=25)
        visualizer.plot_predictive_conf_entropy_scores(
            predictive_confidence, self.entropy_scores
        )
        self._plot_file_names.update(visualizer.plot_file_names)

    def table_generator(self, x_test, y_test):
        """
        Generates a table summarizing true labels, predicted labels, and various uncertainty metrics.
        :param x_test: (np.array): Test dataset inputs.
        :param y_test:(np.array): Test dataset labels.
        """
        true_labels = np.argmax(y_test, axis=1) if np.ndim(y_test) > 1 else y_test
        predicted_labels = np.argmax(
            self.model.predict(x_test), axis=1
        )  # y_pred_classes
        # self.pcs_scores = -self.pcs_scores
        # self.mean_softmax_scores = -self.mean_softmax_scores
        # predictive_confidence = np.max(self.model.predict(x_test), axis=1) # Mean Softmax Probability

        # Create a table with the following columns
        table = pd.DataFrame(
            {
                "True Label": true_labels,
                "Predicted Label": predicted_labels,
                "Predictive Confidence": self.pcs_scores,
                "Mean Softmax Probability": self.mean_softmax_scores,
                "Prediction Entropy": self.entropy_scores,
            }
        )

        # sort the table by predictive entropy
        table = table.sort_values(by="Prediction Entropy", ascending=False)
        print(table.head(10))
        try:
            save_path = input(
                "Enter the path to save the tables or press Enter to use the default path: "
            ).strip()
            output_dir = save_path if save_path else "data/tables"

        except EOFError as e:
            logging.error("analyze: error with input from user console: %s", e)
            output_dir = "data/tables"
            pass

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_csv = f"table_{timestamp}.csv"
        filename_xlsx = f"table_{timestamp}.xlsx"

        table.to_csv(os.path.join(output_dir, filename_csv))
        table.to_excel(os.path.join(output_dir, filename_xlsx))
        logging.info("Tables saved to %s", output_dir)

        return table

    @property
    def plot_file_names(self) -> dict:
        return self._plot_file_names
