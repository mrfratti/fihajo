"""arguments for saving and loding model"""

import argparse

import logging
import numpy as np

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

from src.visualization.visualization import VisualizeEvaluation
from src.weight_processing.weight_manager import WeightManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message).80s"
)


class Evaluator:
    """
    Evaluator class for evaluating a trained model, including support for adversarial evaluation.
    """

    def __init__(self, model_builder, dataset, args: argparse.Namespace) -> None:
        """
        Initializes the Evaluator object with command-line arguments and loads the trained model.
        :param args: Command-line arguments specifying evaluation parameters.
        """
        self.args = args
        self.dataset = dataset
        self._weightmanager = WeightManager()
        self._weightmanager.model_path = args.model_path
        self._weightmanager.current_model = model_builder
        self._weightmanager.load_weights()
        self.model = self._weightmanager.current_model
        self.adversarial_evaluated = args.adv_eval
        self._plot_file_names = {}
        self._interactive_plot_file_names = {}

    @property
    def default_path(self) -> str:
        """returns the default path of weights storing location"""
        return self._weightmanager.default_path

    def evaluate(self):
        """
        Organizes the evaluation process, including standard and adversarial evaluation if specified.
        """
        _, (x_test, y_test) = self.dataset

        self._weightmanager.loading_effect(duration=15, message="Loading model weights")
        self.evaluation(x_test, y_test, plot_results=not self.args.adv_eval)
        if self.args.adv_eval:
            self.adversarial_evaluation(x_test, y_test)

    def evaluation(self, x_test, y_test, plot_results=True):
        # Evaluate the model
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        logging.info(f"Evaluation - Loss: {loss:.2f}%, Accuracy: {acc * 100:.2f}%")

        if plot_results:
            visualizer = VisualizeEvaluation()
            y_pred = self.model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)

            class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            if self.args.dataset == "fashion_mnist":
                class_names = [
                    "T-shirt/top",
                    "Trouser",
                    "Pullover",
                    "Dress",
                    "Coat",
                    "Sandal",
                    "Shirt",
                    "Sneaker",
                    "Bag",
                    "Ankle boot",
                ]
            elif self.args.dataset == "cifar10":
                class_names = [
                    "airplane",
                    "automobile",
                    "bird",
                    "cat",
                    "deer",
                    "dog",
                    "frog",
                    "horse",
                    "ship",
                    "truck",
                ]

            list_class = [str(i) for i in class_names]

            visualizer.plot_predictions(
                self.model.inner, x_test, y_true, classes=list_class, num_samples=25
            )
            visualizer.plot_confusion_matrix(y_true, y_pred_classes, classes=list_class)
            visualizer.plot_classification_report(
                y_true, y_pred_classes, classes=list_class
            )
            self._plot_file_names.update(visualizer.plot_file_names)

            if self.args.interactive:
                visualizer.plot_predictions(
                    self.model.inner,
                    x_test,
                    y_true,
                    classes=list_class,
                    num_samples=25,
                    filename_text="interactive_plot_file_names",
                )
                visualizer.plot_interactive_confusion_matrix(
                    y_true, y_pred_classes, classes=list_class
                )
                visualizer.plot_interactive_classification_report(
                    y_true, y_pred_classes, classes=list_class
                )
                self._interactive_plot_file_names.update(
                    visualizer.interactive_plot_file_names
                )

    def adversarial_evaluation(self, x_test, y_test):
        self.evaluation(x_test, y_test, plot_results=True)
        # Fast Gradient Sign Method
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        x_adv_fgsm = fast_gradient_method(
            self.model.inner, x_test, self.args.eps, np.inf, clip_min=0.0, clip_max=1.0
        )
        loss_fgsm, acc_fgsm = self.model.evaluate(x_adv_fgsm, y_test, verbose=1)
        # predictions_fgsm = self.model.predict(x_adv_fgsm)
        logging.info(
            f"Evaluation on FGSM - Loss: {loss_fgsm:.2f}%, Accuracy: {acc_fgsm * 100:.2f}%"
        )

        # Projected Gradient Descent
        x_adv_pgd = projected_gradient_descent(
            self.model.inner,
            x_test,
            self.args.eps,
            0.01,
            40,
            np.inf,
            clip_min=0.0,
            clip_max=1.0,
        )
        loss_pgd, acc_pgd = self.model.evaluate(x_adv_pgd, y_test, verbose=1)
        # predictions_pgd = self.model.predict(x_adv_pgd)
        logging.info(
            f"Evaluation on PGD - Loss: {loss_pgd:.2f}%, Accuracy: {acc_pgd * 100:.2f}%"
        )

        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        if self.args.dataset == "fashion_mnist":
            class_names = [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        elif self.args.dataset == "cifar10":
            class_names = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]

        list_class = [str(i) for i in class_names]

        visualizer = VisualizeEvaluation()
        visualizer.plot_adversarial_examples(
            self.model, x_test, self.args.eps, classes=list_class, num_samples=25
        )
        accuracies = [acc * 100, acc_fgsm * 100, acc_pgd * 100]
        labels = ["Clean", "FGSM", "PGD"]
        visualizer.plot_accuracy_comparison(accuracies, labels=labels)

        self._plot_file_names.update(visualizer.plot_file_names)

        if self.args.interactive:
            visualizer.plot_adversarial_examples(
                self.model,
                x_test,
                self.args.eps,
                classes=list_class,
                num_samples=25,
                filename_text="adversarial_examples",
            )
            visualizer.plot_interactive_accuracy_comparison(accuracies, labels=labels)
            self._interactive_plot_file_names.update(
                visualizer.interactive_plot_file_names
            )

    @property
    def plot_file_names(self) -> dict:
        return self._plot_file_names

    @property
    def interactive_plot_file_names(self) -> dict:
        return self._interactive_plot_file_names
