import argparse
import logging
import os.path
import pandas as pd
from sklearn.metrics import classification_report
from .model_utils import load_and_preprocess_mnist, create_mnist_model
from src.visualization.visualization import VisualizeEvaluation
import numpy as np
from uncertainty_wizard.models import StochasticMode
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Evaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = self.create_and_load_model(args.model_path)

    def create_and_load_model(self, model_path: str):
        stochastic_mode = StochasticMode()
        model = create_mnist_model(stochastic_mode)
        model_path = self.args.model_path or self._default_load_path()
        logging.info(f"Loading model weights from {model_path}")
        model.inner.load_weights(model_path)
        return model

    @staticmethod
    def _default_load_path() -> str:
        return 'data/models/model_weights.h5'

    def evaluate(self):
        _, (x_test, y_test) = load_and_preprocess_mnist()

        self.evaluation(x_test, y_test, plot_results=not self.args.adv_eval)
        if self.args.adv_eval:
            self.adversarial_evaluation(x_test, y_test)

    def evaluation(self, x_test, y_test, plot_results=True):
        # Evaluate the model
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        logging.info(f"Evaluation - Loss: {loss * 100:.2f}%, Accuracy: {acc * 100:.2f}%")
        if plot_results:
            y_pred = self.model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)

            visualizer = VisualizeEvaluation()
            visualizer.plot_predictions(self.model.inner, x_test, y_true, num_samples=25)
            visualizer.plot_confusion_matrix(y_true, y_pred_classes, classes=[str(i) for i in range(10)])
            visualizer.plot_classification_report(y_true, y_pred_classes)

    def adversarial_evaluation(self, x_test, y_test):
        # Fast Gradient Sign Method
        loss, acc = self.model.evaluate(x_test, y_test, verbose=1)
        x_adv_fgsm = fast_gradient_method(self.model.inner, x_test, self.args.eps, np.inf, clip_min=0., clip_max=1.)
        loss_fgsm, acc_fgsm = self.model.evaluate(x_adv_fgsm, y_test, verbose=1)
        predictions_fgsm = self.model.predict(x_adv_fgsm)
        logging.info(f"Evaluation on FGSM - Loss: {loss_fgsm:.2f}%, Accuracy: {acc_fgsm * 100:.2f}%")

        # Projected Gradient Descent
        x_adv_pgd = projected_gradient_descent(self.model.inner, x_test,
                                               self.args.eps, 0.01, 40,
                                               np.inf, clip_min=0., clip_max=1.)
        loss_pgd, acc_pgd = self.model.evaluate(x_adv_pgd, y_test, verbose=1)
        predictions_pgd = self.model.predict(x_adv_pgd)
        logging.info(f"Evaluation on PGD - Loss: {loss_pgd * 100:.2f}%, Accuracy: {acc_pgd * 100:.2f}%")

        visualizer = VisualizeEvaluation()
        # Plotting for adversarial evaluation
        visualizer.plot_adversarial_examples(self.model, x_test, self.args.eps, num_samples=25)
        accuracies = [acc * 100, acc_fgsm * 100, acc_pgd * 100]
        labels = ['Clean', 'FGSM', 'PGD']
        visualizer.plot_accuracy_comparison(accuracies, labels=labels)


if __name__ == "__main__":
    pass
