import argparse
import logging
import os.path
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.manifold import TSNE
from src.models.model_utils import ModelUtils
from src.visualization.visualization import VisualizeUncertainty
from uncertainty_wizard.models import StochasticMode


class Analyzer:
    """
    The Analyzer class is responsible for analyzing a trained model.
    It quantifies uncertainty using various metrics and visualizes them.
    """
    def __init__(self, model_path: str, batch: int):
        """
        Initializes the Analyzer with the model path and batch size.
        :param model_path: Path to the trained model weights.
        :param batch: Batch size for processing data.
        """
        self.model_path = model_path
        self.batch = batch
        self.utils = ModelUtils()
        self.model = self.create_and_load_model(self.model_path)
        self.quantified_results = None
        self.pcs_mean_softmax_scores = None
        self.entropy_scores = None
        self.mean_softmax_scores = None
        self.pcs_scores = None

    def create_and_load_model(self, model_path: str):
        """
        Creates and laods a MNIST model with weights from a specified path.
        :param model_path: Path to the model weights file.
        :return: A TensorFlow model with loaded weights.
        """
        stochastic_mode = StochasticMode()
        model = self.utils.create_mnist_model(stochastic_mode)
        logging.info(f"Loading model weights from {model_path}")
        model.inner.load_weights(model_path)
        return model

    @staticmethod
    def _default_load_path():
        return 'data/models/model_weights.h5'

    def analyze(self):
        """
        Runs the complete analysis, including quantified uncertainty analysis,
        PCS and mean softmax calculation, entropy analysis, and table generation.
        """
        _, (x_test, y_test) = self.utils.load_and_preprocess_mnist()
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
            quantifiers = ['pcs', 'mean_softmax', 'predictive_entropy']
            self.quantified_results = self.model.predict_quantified(x_test,
                                                                    quantifier=quantifiers,
                                                                    batch_size=self.batch,
                                                                    sample_size=32,
                                                                    verbose=1)

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

    def analyze_entropy(self, x_test):
        """
        Analyzes and visualizes the entropy of model predictions alongside other metrics.
        :param x_test: (np.array): Test dataset inputs.
        """
        predictive_confidence = np.max(self.model.predict(x_test), axis=1)

        # Extracting features from second last layer (dense) for t-SNE visualization
        feature_model = tf.keras.Model(inputs=self.model.inner.inputs,
                                       outputs=self.model.inner.get_layer('dense').output)

        features = feature_model.predict(x_test)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(features)

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
            print(f"Entropy: {self.entropy_scores[i]:.2f}, Confidence: {predictive_confidence[i]:.2f}")

        # Print confidence and entropy for the first 5 samples
        print("\nConfidence and entropy score for the first 5 samples:")
        for i in range(5):
            print(f"Entropy: {self.entropy_scores[i]:.2f}, Confidence: {predictive_confidence[i]:.2f}")

        visualizer = VisualizeUncertainty()
        # Plot the distribution of entropy scores
        visualizer.plot_dist_entropy_scores(self.entropy_scores)
        visualizer.high_uncertain_inputs(self.entropy_scores, x_test, num_samples=25)
        visualizer.plot_predictive_conf_entropy_scores(predictive_confidence, self.entropy_scores)
        visualizer.plot_tsne_entropy(tsne_results, self.entropy_scores)

    def table_generator(self, x_test, y_test):
        """
        Generates a table summarizing true labels, predicted labels, and various uncertainty metrics.
        :param x_test: (np.array): Test dataset inputs.
        :param y_test:(np.array): Test dataset labels.
        """
        true_labels = np.argmax(y_test, axis=1) if np.ndim(y_test) > 1 else y_test
        predicted_labels = np.argmax(self.model.predict(x_test), axis=1)  # y_pred_classes
        #self.pcs_scores = -self.pcs_scores
        #self.mean_softmax_scores = -self.mean_softmax_scores
        # predictive_confidence = np.max(self.model.predict(x_test), axis=1) # Mean Softmax Probability

        # Create a table with the following columns
        table = pd.DataFrame({'True Label': true_labels,
                              'Predicted Label': predicted_labels,
                              'Predictive Confidence': self.pcs_scores,
                              'Mean Softmax Probability': self.mean_softmax_scores,
                              'Prediction Entropy': self.entropy_scores
                              })

        # sort the table by predictive entropy
        table = table.sort_values(by='Prediction Entropy', ascending=False)
        print(table.head(10))

        save_path = input("Enter the path to save the tables or press Enter to use the default path: ").strip()
        output_dir = save_path if save_path else 'data/tables'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename_csv = f"table_{timestamp}.csv"
        filename_xlsx = f"table_{timestamp}.xlsx"

        table.to_csv(os.path.join(output_dir, filename_csv))
        table.to_excel(os.path.join(output_dir, filename_xlsx))
        logging.info(f"Tables saved to {output_dir}")

        return table


if __name__ == "__main__":
    pass
