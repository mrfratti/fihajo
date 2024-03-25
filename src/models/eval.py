import argparse
import logging
import sys
import time
from src.visualization.visualization import VisualizeEvaluation
import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
        self.model = model_builder.create_model()
        self.dataset = dataset

        self.load_weights(args.model_path)

    @staticmethod
    def _default_load_path() -> str:
        return "data/models/model.weights.h5"

    def load_weights(self, model_path=None):
        if not model_path:
            model_path = self._default_load_path()
        self.model.inner.load_weights(model_path)
        logging.info(f"Model weights loaded from {model_path}")

    def evaluate(self):
        """
        Orchestrates the evaluation process, including standard and adversarial evaluation if specified.
        """
        _, (x_test, y_test) = self.dataset

        self.loading_effect(duration=15, message="Loading model weights")
        self.evaluation(x_test, y_test, plot_results=not self.args.adv_eval)
        if self.args.adv_eval:
            self.adversarial_evaluation(x_test, y_test)

    def evaluation(self, x_test, y_test, plot_results=True):
        # Evaluate the model
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        logging.info(
            f"Evaluation - Loss: {loss * 100:.2f}%, Accuracy: {acc * 100:.2f}%"
        )

        if plot_results:
            visualizer = VisualizeEvaluation()
            y_pred = self.model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)

            visualizer.plot_predictions(
                self.model.inner, x_test, y_true, num_samples=25
            )
            visualizer.plot_confusion_matrix(
                y_true, y_pred_classes, classes=[str(i) for i in range(10)]
            )
            visualizer.plot_classification_report(y_true, y_pred_classes)
            

    def adversarial_evaluation(self, x_test, y_test):
        # Fast Gradient Sign Method
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        x_adv_fgsm = fast_gradient_method(
            self.model.inner, x_test, self.args.eps, np.inf, clip_min=0.0, clip_max=1.0
        )
        loss_fgsm, acc_fgsm = self.model.evaluate(x_adv_fgsm, y_test, verbose=1)
        predictions_fgsm = self.model.predict(x_adv_fgsm)
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
        predictions_pgd = self.model.predict(x_adv_pgd)
        logging.info(
            f"Evaluation on PGD - Loss: {loss_pgd * 100:.2f}%, Accuracy: {acc_pgd * 100:.2f}%"
        )

        visualizer = VisualizeEvaluation()
        # Plotting for adversarial evaluation
        visualizer.plot_adversarial_examples(
            self.model, x_test, self.args.eps, num_samples=25
        )
        accuracies = [acc * 100, acc_fgsm * 100, acc_pgd * 100]
        labels = ["Clean", "FGSM", "PGD"]
        visualizer.plot_accuracy_comparison(accuracies, labels=labels)

    def loading_effect(self, duration=3, message="Evaluating"):
        print(message, end="")
        for _ in range(duration):
            for cursor in "|/-\\":
                sys.stdout.write(cursor)
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write("\b")
        print(" Done!")
