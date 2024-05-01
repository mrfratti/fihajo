import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from src.cli.string_styling import StringStyling

import json



class VisualizeTraining:
    """_summary_
    generate plot and storing plots
    """

    def __init__(self, plot_dir="report/reports/data/plots/training"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}


    def _plot_results(self, history, mode, title, ylabel="", xlabel="Epoch", historytags=[]):
        """Plot training & validation accuracy"""
        if mode == "accuracy":
            plt.subplot(1, 2, 1)
            historytags.append({"data": "accuracy", "label": "Train Accuracy"})
            historytags.append({"data": "val_accuracy", "label": "Validation Accuracy"})
            ylabel = "Accuracy"
        if mode == "loss":
            plt.subplot(1, 2, 2)
            historytags.append({"data": "loss", "label": "Train Loss"})
            historytags.append({"data": "val_loss", "label": "Validation Loss"})
            ylabel = "Loss"
        for tag in historytags:
            if isinstance(history, dict):
                plt.plot(history[tag["data"]], label=tag["label"])
            else:
                plt.plot(history.history[tag["data"]], label=tag["label"])
        historytags.clear()
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.legend()

        # --- Interactive Chart --- |

        history_data = history.history
        
        x_value = []
        for x_axis_nr in range(1, len(history_data["accuracy"]) + 1):
            x_value.append(x_axis_nr)

        data_info = {
            "x": x_value,
            "y1": history_data["accuracy"],
            "y2": history_data["val_accuracy"],
            "y3": history_data["loss"],
            "y4": history_data["val_loss"]
        }

        date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        file_path = 'report/reports/data/plots/training/'
        # file_path_2 = f"{file_path}val_acc_and_loss_{date_time}.json"
        file_path_2 = f"{file_path}val_acc_and_loss.json"
        
        full_file_path = os.path.join(os.getcwd(), file_path_2)
        with open(full_file_path, 'w') as file:
            json.dump(data_info, file, indent=4)



    def plot_training_results(self, history):
        # Plot training & validation accuracy and loss
        plt.figure(dpi=1200)
        # Accuracy
        self._plot_results(
            history,
            mode="accuracy",
            title="Training Accuracy",
        )
        # Loss
        self._plot_results(history, mode="loss", title="Model loss")
        filename = self._save_plot("val_acc_and_loss")
        plt.show()
        self._plot_file_names["training"] = filename

    def plot_adversarial_training_results(self, history):
        """_summary_"""
        plt.figure(dpi=1200)
        # Accuracy
        self._plot_results(history, mode="accuracy", title="Adversarial Training Accuracy")
        # loss
        self._plot_results(history, mode="loss", title="Adversarial Training Loss")
        filename = self._save_plot("adv_train_acc_loss")
        plt.show()
        self._plot_file_names["adversarialTraining"] = filename

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of filenames"""
        if not isinstance(self._plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for filname should be dict"
                )
            )
        if len(self._plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing filnames in dict")
            )
        return self._plot_file_names


class VisualizeEvaluation:
    def __init__(self, plot_dir="report/reports/data/plots/evaluation"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}

    def plot_predictions(self, model, x_test, y_true, num_samples=25):
        predictions = model.predict(x_test[:num_samples])
        predicted_labels = np.argmax(predictions, axis=1)
        # true_label = np.argmax(y_test, axis=1) if np.ndim(y_test) > 1 else y_test
        plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
            plt.title(f"True: {y_true[i]}, Predicted: {predicted_labels[i]}")
            plt.axis("off")
        plt.tight_layout()
        filename = self._save_plot("predictions")
        plt.show()
        self._plot_file_names["predictions"] = filename

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(20, 15))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        filename = self._save_plot("confusion_matrix")
        plt.show()
        self._plot_file_names["confusion_matrix"] = filename


        # --- Confusion Matrix --- |

        data_info = {
            "x": cm,
        }

        date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        file_path = 'report/reports/data/plots/training/'
        # file_path_2 = f"{file_path}val_acc_and_loss_{date_time}.json"
        file_path_2 = f"{file_path}confusion_matrix.json"
        
        full_file_path = os.path.join(os.getcwd(), file_path_2)
        with open(full_file_path, 'w') as file:
            json.dump(data_info, file, indent=4)




    def plot_classification_report(self, y_true, y_pred_classes, output_dict=True):
        report = classification_report(
            y_true, y_pred_classes, output_dict=output_dict, zero_division=0
        )
        df_report = pd.DataFrame(report).transpose()
        df_report.drop("support", errors="ignore", inplace=True)
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            df_report[["precision", "recall", "f1-score"]].T,
            annot=True,
            cmap="viridis",
            fmt=".2f",
        )
        plt.title("Classification Report")
        filename = self._save_plot("classification_report")
        plt.show()
        self._plot_file_names["classification_report"] = filename

    def plot_adversarial_examples(self, model, x_test, eps, num_samples=25):
        # Generate FGSM adversarial examples
        x_adv_fgsm = fast_gradient_method(model.inner, x_test[:num_samples], eps, np.inf)
        predictions_fgsm = np.argmax(model.predict(x_adv_fgsm), axis=1)

        # predictions_clean = np.argmax(model.predict(x_test[:num_samples]), axis=1)

        # Generate PGD adversarial examples
        x_adv_pgd = projected_gradient_descent(
            model.inner, x_test[:num_samples], eps, 0.01, 40, np.inf
        )
        predictions_pgd = np.argmax(model.predict(x_adv_pgd), axis=1)

        plt.figure(figsize=(2 * num_samples, 6))

        # plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            # Original images
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(x_test[i], cmap="gray")
            plt.title(
                f"Clean\nPred: {np.argmax(model.predict(x_test[i:i + 1]), axis=1)[0]}",
                fontsize=9,
            )
            plt.axis("off")

            # Plot FGSM adversarial images
            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.imshow(x_adv_fgsm[i], cmap="gray")
            plt.title(f"FGSM\nPred: {predictions_fgsm[i]}", fontsize=9)
            plt.axis("off")

            # Plot PGD adversarial images
            plt.subplot(3, num_samples, 2 * num_samples + i + 1)
            plt.imshow(x_adv_pgd[i], cmap="gray")
            plt.title(f"PGD\nPred: {predictions_pgd[i]}", fontsize=9)
            plt.axis("off")

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        filename = self._save_plot("adv_examples")
        plt.show()
        self._plot_file_names["adverserial_ex"] = filename

    def plot_accuracy_comparison(self, accuracies, labels=["Clean", "FGSM", "PGD"]):
        plt.figure(figsize=(8, 6))
        bar_positions = np.arange(len(accuracies))
        plt.bar(bar_positions, accuracies, color=["blue", "green", "red"])
        plt.xticks(bar_positions, labels)
        plt.ylabel("Accuracy (%)")
        plt.title("Model Accuracy: Clean vs Adversarial Examples")
        plt.ylim(0, max(accuracies) + 10)
        # plt.ylim(0, 110)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 2, f"{acc:.2f}", ha="center", va="bottom")

        filename = self._save_plot("accuracy_comparison")
        plt.show()
        self._plot_file_names["accuracy_comparison"] = filename

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of filenames"""
        if not isinstance(self._plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for filname should be dict"
                )
            )
        if len(self._plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing filnames in dict")
            )
        return self._plot_file_names


class VisualizeUncertainty:
    def __init__(self, plot_dir="report/reports/data/plots/analyze"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}

    def plot_pcs_mean_softmax(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        # Results is a list of tuples, where each tuple contains (predictions, scores)
        # pcs_scores, mean_softmax_scores = results[0][1], results[1][1]

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.histplot(pcs_scores, bins=50, kde=True, color="skyblue")
        plt.xlabel("PCS Score", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of PCS Scores")
        plt.subplot(1, 2, 2)
        sns.histplot(mean_softmax_scores, bins="auto", kde=True, color="lightgreen")
        plt.xlabel("Mean Softmax Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of Mean Softmax Scores")
        plt.tight_layout()
        filename = self._save_plot("pcs_meansoftmax")
        plt.show()
        self._plot_file_names["pcs_meansoftmax"] = filename

    def plot_distribution_pcs_ms_scores(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        plt.figure(figsize=(10, 10))
        sns.histplot(pcs_scores, bins=50, alpha=0.7, color="blue", kde=True, label="PCS")
        sns.histplot(
            mean_softmax_scores,
            bins=50,
            alpha=0.7,
            color="red",
            kde=True,
            label="Mean Softmax",
        )
        plt.xlabel("Predictive Confidence Score")
        plt.ylabel("Frequency")
        plt.title("Distribution of PCS and Mean Softmax Scores")
        plt.legend()
        filename = self._save_plot("dist_pcs_meansoftmax")
        plt.show()
        self._plot_file_names["distrubution_meansoftmax"] = filename

    def plot_pcs_ms_inverse(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        pcs_inverse = 1 - pcs_scores
        uncertainty_threshold_pcs = np.percentile(pcs_inverse, 95)
        mean_softmax_inverse = 1 - mean_softmax_scores
        uncertainty_threshold_mean_softmax = np.percentile(mean_softmax_inverse, 95)

        plt.figure(figsize=(20, 10))
        # First subplot for PCS scores
        plt.subplot(1, 2, 1)
        sns.histplot(pcs_inverse, bins="auto", kde=True, color="skyblue")
        plt.axvline(
            uncertainty_threshold_pcs,
            color="r",
            linestyle="dashed",
            linewidth=2,
            label=f"95th percentile: {uncertainty_threshold_pcs:.2f}",
        )
        plt.axvline(
            np.mean(pcs_inverse),
            color="k",
            linestyle="dashed",
            label=f"Mean PCS: {np.mean(pcs_scores):.2f}",
        )
        plt.xlabel("Prediction Confidence Score (PCS)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of PCS Scores as Uncertainty", fontsize=18)
        plt.legend()

        # Second subplot for Mean Softmax scores
        plt.subplot(1, 2, 2)
        sns.histplot(mean_softmax_inverse, bins="auto", kde=True, color="lightgreen")
        plt.axvline(
            uncertainty_threshold_mean_softmax,
            color="r",
            linestyle="dashed",
            label=f"95th percentile: {uncertainty_threshold_mean_softmax:.2f}",
        )
        plt.axvline(
            np.mean(mean_softmax_inverse),
            color="k",
            linestyle="dashed",
            label=f"Mean Softmax: {np.mean(mean_softmax_scores):.2f}",
        )
        plt.xlabel("Mean Softmax Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Mean Softmax Scores as Uncertainty", fontsize=18)
        plt.legend()
        plt.tight_layout()
        filename = self._save_plot("pcs_ms_inverse")
        plt.show()
        self._plot_file_names["pcs_inverse"] = filename

    def plot_dist_entropy_scores(self, entropy_scores):
        plt.figure(figsize=(20, 10))
        sns.histplot(entropy_scores, bins=50, kde=True, color="blue", label="Clean Data")
        plt.axvline(
            np.mean(entropy_scores),
            color="k",
            linestyle="dashed",
            linewidth=2,
            label="Mean",
        )
        plt.xlabel("Predictive Entropy", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Histogram of Predictive Entropy", fontsize=24)
        plt.legend()
        filename = self._save_plot("dist_entropy")
        plt.show()
        self._plot_file_names["entropy_distrubution"] = filename

    def high_uncertain_inputs(self, entropy_scores, x_test, num_samples=25):
        # Sort the indices of the entropy scores in descending order
        sorted_indices = np.argsort(entropy_scores)[::-1]

        # Plot the most uncertain examples
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[sorted_indices[i]], cmap="gray")
            plt.title(f"Entropy: {entropy_scores[sorted_indices[i]]:.2f}")
            plt.axis("off")
        plt.tight_layout()
        filename = self._save_plot("high_uncertain_inputs")
        plt.show()
        self._plot_file_names["higly_uncertain_inputs"] = filename

    def plot_predictive_conf_entropy_scores(self, predictive_confidence, entropy_scores):
        plt.figure(figsize=(20, 10))
        plt.scatter(
            predictive_confidence,
            entropy_scores,
            c=entropy_scores,
            cmap="viridis",
            alpha=0.5,
        )
        plt.colorbar(label="Predictive Entropy")
        plt.xlabel("Predictive Confidence")
        plt.ylabel("Entropy Score")
        plt.title("Predictive Confidence vs Entropy Score")
        filename = self._save_plot("pred_vs_entropy")
        plt.show()
        self._plot_file_names["prediction_vs_entrophy"] = filename

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of filenames"""
        if not isinstance(self._plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for filname should be dict"
                )
            )
        if len(self._plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing filnames in dict")
            )
        return self._plot_file_names
