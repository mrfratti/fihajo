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
import plotly
import plotly.graph_objects as plotly_graph_objects
import plotly.figure_factory as plotly_figure_factory
from plotly.subplots import make_subplots
from src.visualization.echart_html import html_accuracy_loss_chart, html_heatmap_chart, html_heatmap_chart_2


class VisualizeTraining:
    """Generates and stores training plots."""

    def __init__(self, plot_dir="src/report/reports/data/plots/training"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}
        self._interactive_plot_file_names = {}

    def _plot_results(self, history, mode, title, ylabel="", xlabel="Epoch", historytags: list = []) -> None:
        """Plot training: validation accuracy & loss"""
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

    def plot_training_results(self, history: dict) -> None:
        """Plots training & validation accuracy and loss."""
        plt.figure(figsize=(20, 10))
        self._plot_results(history, mode="accuracy", title="Training Accuracy")
        self._plot_results(history, mode="loss", title="Model loss")
        filename = self._save_plot("val_acc_and_loss")
        self._plot_file_names["training"] = filename

    def plot_interactive_training_results(self, history):
        """Plots interactive training & validation accuracy and loss."""
        history_data = history.history

        list_x_value = []
        list_accuracy = []
        list_val_accuracy = []
        list_loss = []
        list_val_loss = []

        rounded_accuracy = 3
        rounded_loss = 3
        for x_axis_nr in range(0, len(history_data["accuracy"])):
            list_x_value.append(x_axis_nr + 1)
            list_accuracy.append(round(history_data["accuracy"][x_axis_nr], rounded_accuracy))
            list_val_accuracy.append(round(history_data["val_accuracy"][x_axis_nr], rounded_accuracy))
            list_loss.append(round(history_data["loss"][x_axis_nr], rounded_loss))
            list_val_loss.append(round(history_data["val_loss"][x_axis_nr], rounded_loss))

        data_info = {
            "x_nr": list_x_value,
            "accuracy": list_accuracy,
            "val_accuracy": list_val_accuracy,
            "loss": list_loss,
            "val_loss": list_val_loss
        }

        data_info_html = html_accuracy_loss_chart(data_info, "Training Accuracy & Loss")
        filename = self._save_interactive_plot_html_2("val_acc_and_loss", data_info_html)
        self._interactive_plot_file_names["training"] = filename

    def plot_adversarial_training_results(self, history:dict) -> None:
        """Plots adversarial training accuracy and loss."""
        plt.figure(figsize=(20, 10))
        self._plot_results(history, mode="accuracy", title="Adversarial Training Accuracy")
        self._plot_results(history, mode="loss", title="Adversarial Training Loss")
        filename = self._save_plot("adv_train_acc_loss")
        self._plot_file_names["adversarialTraining"] = filename

    def plot_interactive_adversarial_training_results(self, history):
        """Plots interactive adversarial training accuracy and loss."""
        history_data = history

        list_x_value = []
        list_accuracy = []
        list_val_accuracy = []
        list_loss = []
        list_val_loss = []

        rounded_accuracy = 3
        rounded_loss = 3
        for x_axis_nr in range(0, len(history_data["accuracy"])):
            list_x_value.append(x_axis_nr + 1)
            list_accuracy.append(round(history_data["accuracy"][x_axis_nr].item(), rounded_accuracy))
            list_val_accuracy.append(round(history_data["val_accuracy"][x_axis_nr].item(), rounded_accuracy))
            list_loss.append(round(history_data["loss"][x_axis_nr].item(), rounded_loss))
            list_val_loss.append(round(history_data["val_loss"][x_axis_nr].item(), rounded_loss))

        data_info = {
            "x_nr": list_x_value,
            "accuracy": list_accuracy,
            "val_accuracy": list_val_accuracy,
            "loss": list_loss,
            "val_loss": list_val_loss
        }

        data_info_html = html_accuracy_loss_chart(data_info, "Adversarial Training Accuracy & Loss")
        filename = self._save_interactive_plot_html_2("adv_train_acc_loss", data_info_html)
        self._interactive_plot_file_names["adversarialTraining"] = filename

    def _save_plot(self, filename: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot_html(self, filename: str, data_info) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.html"
        plotly.offline.plot(data_info, filename=os.path.join(self.plot_dir, filename), include_plotlyjs=True,
                            auto_open=False)

        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot_html_2(self, filename, data_info):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.html"

        full_file_path = os.path.join(os.getcwd(), f"{self.plot_dir}/{filename}")

        with open(full_file_path, 'w') as file:
            file.write(data_info)

        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of plot filenames."""
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

    @property
    def interactive_plot_file_names(self) -> dict:
        """Returns a dictionary of interactive plot filenames."""
        if not isinstance(self._interactive_plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(message="visualizer: Wrong datatype for filname should be dict"))
        if len(self._interactive_plot_file_names) < 1:
            raise ValueError(StringStyling.box_style(message="visualizer: missing filnames in dict"))
        return self._interactive_plot_file_names


class VisualizeEvaluation:
    """Generates and stores evaluation plots."""
    def __init__(self, plot_dir="src/report/reports/data/plots/evaluation"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}
        self._interactive_plot_file_names = {}

    def plot_predictions(self, model, x_test: np.ndarray, y_true: np.ndarray, classes: list,
                         num_samples: int = 25, filename_text: str = "plot_file_names") -> None:
        predictions = model.predict(x_test[:num_samples])
        predicted_labels = np.argmax(predictions, axis=1)
        plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            if x_test.shape[-1] == 1:  # Grayscale images
                plt.imshow(x_test[i].reshape(x_test.shape[1], x_test.shape[2]), cmap="gray")
            else:   # RGB images
                plt.imshow(x_test[i].reshape(x_test.shape[1], x_test.shape[2], x_test.shape[3]))
            plt.title(f"True: {classes[y_true[i]]}, Predicted: {classes[predicted_labels[i]]}")
            plt.axis("off")
        plt.tight_layout()

        if filename_text == "plot_file_names":
            filename = self._save_plot("predictions")
            self._plot_file_names["predictions"] = filename
        elif filename_text == "interactive_plot_file_names":
            filename = self._save_interactive_plot("predictions")
            self._interactive_plot_file_names["predictions"] = filename

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> None:
        """Plots confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 15))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix", fontsize=20)
        plt.ylabel("True Label", fontsize=18)
        plt.xlabel("Predicted Label", fontsize=18)
        filename = self._save_plot("confusion_matrix")
        self._plot_file_names["confusion_matrix"] = filename

    def plot_interactive_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        data_info = [{"value": int(cm[x][y]), "row": classes[x], "column": classes[y]}
                     for x in range(len(cm))
                     for y in range(len(cm[x]))]
        data_info_html = html_heatmap_chart(data_info, classes)
        filename = self._save_interactive_plot_html_2("confusion_matrix", data_info_html)
        self._interactive_plot_file_names["confusion_matrix"] = filename

    def plot_classification_report(self, y_true: np.ndarray, y_pred_classes: np.ndarray,
                                   classes: list, output_dict: bool = True) -> None:
        """Plots a classification report."""
        report = classification_report(y_true, y_pred_classes,
                                       output_dict=output_dict, zero_division=0, target_names=classes)
        df_report = pd.DataFrame(report).transpose().drop("support", errors="ignore")
        plt.figure(figsize=(20, 16))
        sns.heatmap(df_report[["precision", "recall", "f1-score"]].T, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Classification Report", fontsize=20)
        filename = self._save_plot("classification_report")
        self._plot_file_names["classification_report"] = filename

    def plot_interactive_classification_report(self, y_true, y_pred_classes, classes, output_dict=True):
        report = classification_report(
            y_true, y_pred_classes, output_dict=output_dict, zero_division=0, target_names=classes
        )
        df_report = pd.DataFrame(report).transpose()
        df_report.drop("support", errors="ignore", inplace=True)

        data_info = {
            "value": df_report.values.tolist(),
            "columns": list(df_report.columns),
            "rows": list(df_report.index)
        }

        data_info_html = html_heatmap_chart_2(data_info)
        filename = self._save_interactive_plot_html_2("classification_report", data_info_html)
        self._interactive_plot_file_names["classification_report"] = filename

    def plot_adversarial_examples(self, model, x_test: np.ndarray, eps: float, classes: list,
                                  num_samples: int = 25, filename_text: str = "plot_file_names") -> None:
        """Plots adversarial examples."""
        # Generate FGSM adversarial examples
        x_adv_fgsm = fast_gradient_method(model.inner, x_test[:num_samples], eps, np.inf)
        predictions_fgsm = np.argmax(model.predict(x_adv_fgsm), axis=1)

        # Generate PGD adversarial examples
        x_adv_pgd = projected_gradient_descent(
            model.inner, x_test[:num_samples], eps, 0.01, 40, np.inf
        )
        predictions_pgd = np.argmax(model.predict(x_adv_pgd), axis=1)

        plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            # Original images
            plt.subplot(3, num_samples, i + 1)
            plt.imshow(x_test[i], cmap="gray")
            plt.title(f"Clean\nPred:\n {classes[np.argmax(model.predict(x_test[i:i + 1]), axis=1)[0]]}", fontsize=10)
            plt.axis("off")

            # Plot FGSM adversarial images
            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.imshow(x_adv_fgsm[i], cmap="gray")
            plt.title(f"FGSM\nPred:\n {classes[predictions_fgsm[i]]}", fontsize=10)
            plt.axis("off")

            # Plot PGD adversarial images
            plt.subplot(3, num_samples, 2 * num_samples + i + 1)
            plt.imshow(x_adv_pgd[i], cmap="gray")
            plt.title(f"PGD\nPred:\n {classes[predictions_pgd[i]]}", fontsize=10)
            plt.axis("off")

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        if filename_text == "plot_file_names":
            filename = self._save_plot("adversarial_examples")
            self._plot_file_names["adversarial_examples"] = filename
        elif filename_text == "interactive_plot_file_names":
            filename = self._save_interactive_plot("adversarial_examples")
            self._interactive_plot_file_names["adversarial_examples"] = filename

    def plot_accuracy_comparison(self, accuracies: list, labels: list = ["Clean", "FGSM", "PGD"]) -> None:
        """Plots accuracy comparison for different evaluation methods."""
        plt.figure(figsize=(10, 8))
        bar_positions = np.arange(len(accuracies))
        plt.bar(bar_positions, accuracies, color=["blue", "green", "red"])
        plt.xticks(bar_positions, labels)
        plt.ylabel("Accuracy (%)", fontsize=18)
        plt.title("Model Accuracy: Clean vs Adversarial Examples", fontsize=20)
        plt.ylim(0, max(accuracies) + 10)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 2, f"{acc:.2f}", ha="center", va="bottom")
        filename = self._save_plot("accuracy_comparison")
        self._plot_file_names["accuracy_comparison"] = filename

    def plot_interactive_accuracy_comparison(self, accuracies, labels=["Clean", "FGSM", "PGD"]):
        """Plots interactive accuracy comparison for different evaluation methods."""
        bar_positions = np.arange(len(accuracies))
        fig = plotly_graph_objects.Figure([
            plotly_graph_objects.Bar(
                x=labels,
                y=accuracies,
                marker_color=["blue", "green", "red"]
            )
        ])

        fig.update_layout(
            title_text="Model Accuracy: Clean vs Adversarial Examples",
            title_font_size=20,
            xaxis=dict(
                title="Model",
                tickmode="array",
                tickvals=bar_positions,
                ticktext=labels
            ),
            yaxis=dict(
                title="Accuracy (%)"
            )
        )

        filename = self._save_interactive_plot_html("accuracy_comparison", fig)
        self._interactive_plot_file_names["accuracy_comparison"] = filename

    def _save_plot(self, filename: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot(self, filename: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot_html(self, filename: str, data_info) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.html"
        plotly.offline.plot(data_info, filename=os.path.join(self.plot_dir, filename), include_plotlyjs=True,
                            auto_open=False)
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot_html_2(self, filename: str, data_info: str) -> str:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.html"
        full_file_path = os.path.join(os.getcwd(), f"{self.plot_dir}/{filename}")
        with open(full_file_path, 'w') as file:
            file.write(data_info)
        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of plot file names"""
        if not isinstance(self._plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for file name should be dict"
                )
            )
        if len(self._plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing file names in dict")
            )
        return self._plot_file_names

    @property
    def interactive_plot_file_names(self) -> dict:
        """Returns a dictionary of interactive plot file names"""
        if not isinstance(self._interactive_plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for file name should be dict"
                )
            )
        if len(self._interactive_plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing file names in dict")
            )
        return self._interactive_plot_file_names


class VisualizeUncertainty:
    """Generates and stores uncertainty analysis plots."""
    def __init__(self, plot_dir="src/report/reports/data/plots/analyze"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self._plot_file_names = {}
        self._interactive_plot_file_names = {}

    def plot_pcs_mean_softmax(self, pcs_mean_softmax_scores: tuple) -> None:
        """Plots the distribution of PCS and Mean Softmax Scores."""
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.histplot(pcs_scores, bins=50, kde=True, color="skyblue")
        plt.xlabel("PCS Score", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of PCS Scores", fontsize=20)
        plt.subplot(1, 2, 2)
        sns.histplot(mean_softmax_scores, bins="auto", kde=True, color="lightgreen")
        plt.xlabel("Mean Softmax Score", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of Mean Softmax Scores", fontsize=20)
        plt.tight_layout()
        filename = self._save_plot("pcs_meansoftmax")
        self._plot_file_names["pcs_meansoftmax"] = filename

    def plot_interactive_pcs_mean_softmax(self, pcs_mean_softmax_scores):
        """Plots interactive distribution of PCS and Mean Softmax Scores."""
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Distribution of PCS Scores", "Distribution of Mean Softmax Scores"))
        fig.add_trace(plotly_graph_objects.Histogram(
            x=pcs_scores,
            name="PCS Scores",
            marker_color="skyblue"),
            row=1, col=1)
        fig.update_xaxes(title_text="PCS Scores", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.add_trace(plotly_graph_objects.Histogram(
            x=mean_softmax_scores,
            name="Mean Softmax Score",
            marker_color="lightgreen"),
            row=2, col=1)
        fig.update_xaxes(title_text="Mean Softmax Score", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_layout(title_text="", xaxis_title_text="", yaxis_title_text="", bargap=0.2)
        filename = self._save_interactive_plot_html("pcs_meansoftmax", fig)
        self._interactive_plot_file_names["pcs_meansoftmax"] = filename

    def plot_distribution_pcs_ms_scores(self, pcs_mean_softmax_scores: tuple) -> None:
        """Plots the distribution of PCS and Mean Softmax Scores."""
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        plt.figure(figsize=(20, 10))
        sns.histplot(pcs_scores, bins=50, alpha=0.7, color="blue", kde=True, label="PCS")
        sns.histplot(
            mean_softmax_scores,
            bins=50,
            alpha=0.7,
            color="red",
            kde=True,
            label="Mean Softmax",
        )
        plt.xlabel("Predictive Confidence Score & Mean Softmax Scores", fontsize=20)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of PCS and Mean Softmax Scores", fontsize=20)
        plt.legend()
        filename = self._save_plot("dist_pcs_meansoftmax")
        self._plot_file_names["distrubution_meansoftmax"] = filename

    def plot_interactive_distribution_pcs_ms_scores(self, pcs_mean_softmax_scores):
        """Plots interactive distribution of PCS and Mean Softmax Scores."""
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        fig = make_subplots(subplot_titles=(""))
        fig.add_trace(plotly_graph_objects.Histogram(
            x=pcs_scores,
            name="PCS",
            marker_color="blue",
            nbinsx=100))
        fig.add_trace(plotly_graph_objects.Histogram(
            x=mean_softmax_scores,
            name="Mean Softmax",
            marker_color="red",
            nbinsx=100))
        fig.update_layout(
            title_text="Distribution of PCS and Mean Softmax Scores",
            xaxis_title_text="Predictive Confidence Score & Mean Softmax Scores",
            yaxis_title_text="Frequency", bargap=0.1)
        filename = self._save_interactive_plot_html("dist_pcs_meansoftmax", fig)
        self._interactive_plot_file_names["distrubution_meansoftmax"] = filename

    def plot_pcs_ms_inverse(self, pcs_mean_softmax_scores: tuple) -> None:
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
        plt.xlabel("Prediction Confidence Score (PCS)", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of PCS Scores as Uncertainty", fontsize=20)
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
        plt.xlabel("Mean Softmax Score", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Distribution of Mean Softmax Scores as Uncertainty", fontsize=20)
        plt.legend()
        plt.tight_layout()
        filename = self._save_plot("pcs_ms_inverse")
        self._plot_file_names["pcs_inverse"] = filename

    def plot_interactive_pcs_ms_inverse(self, pcs_mean_softmax_scores):
        """Plots interactive inverse of PCS and Mean Softmax Scores."""
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        pcs_inverse = 1 - pcs_scores
        uncertainty_threshold_pcs = np.percentile(pcs_inverse, 95)
        mean_softmax_inverse = 1 - mean_softmax_scores
        uncertainty_threshold_mean_softmax = np.percentile(mean_softmax_inverse, 95)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Distribution of PCS Scores as Uncertainty",
                                                            "Distribution of Mean Softmax Scores as Uncertainty"))

        fig.add_trace(plotly_graph_objects.Histogram(x=pcs_inverse, name='PCS Scores', marker_color='skyblue'), row=1,
                      col=1)

        fig.add_trace(plotly_graph_objects.Scatter(
            x=[uncertainty_threshold_pcs, uncertainty_threshold_pcs],
            y=[0, max(np.histogram(pcs_scores, bins='auto')[0])],
            mode="lines", name=f'95th percentile: {uncertainty_threshold_pcs:.2f}',
            line=dict(color='red', dash='dash')), row=1, col=1)

        fig.add_trace(plotly_graph_objects.Scatter(
            x=[np.mean(pcs_inverse), np.mean(pcs_inverse)],
            y=[0, max(np.histogram(pcs_scores, bins='auto')[0])],
            mode="lines", name=f'Mean PCS: {np.mean(pcs_scores):.2f}',
            line=dict(color='yellow', dash='dash')), row=1, col=1)

        fig.add_trace(plotly_graph_objects.Histogram(x=mean_softmax_inverse, name='Mean Softmax Scores',
                                                     marker_color='lightgreen'), row=2, col=1)

        fig.add_trace(plotly_graph_objects.Scatter(
            x=[uncertainty_threshold_mean_softmax, uncertainty_threshold_mean_softmax],
            y=[0, max(np.histogram(mean_softmax_scores, bins='auto')[0])],
            mode="lines", name=f'95th percentile: {uncertainty_threshold_mean_softmax:.2f}',
            line=dict(color='red', dash='dash')), row=2, col=1)

        fig.add_trace(plotly_graph_objects.Scatter(
            x=[np.mean(mean_softmax_inverse), np.mean(mean_softmax_inverse)],
            y=[0, max(np.histogram(mean_softmax_scores, bins='auto')[0])],
            mode="lines", name=f'Mean Softmax: {np.mean(mean_softmax_scores):.2f}',
            line=dict(color='yellow', dash='dash')), row=2, col=1)

        fig.update_xaxes(title_text="Scores", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)

        fig.update_xaxes(title_text="Scores", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

        fig.update_layout(title_text='Interactive Distribution Plots', xaxis_title_text='',
                          yaxis_title_text='Frequency', bargap=0.2)

        filename = self._save_interactive_plot_html("pcs_ms_inverse", fig)
        self._interactive_plot_file_names["pcs_ms_inverse"] = filename

    def plot_dist_entropy_scores(self, entropy_scores: list) -> None:
        """Plots the distribution of entropy scores."""
        plt.figure(figsize=(20, 10))
        sns.histplot(entropy_scores, bins=50, kde=True, color="red", label="Entropy Scores")
        plt.axvline(np.mean(entropy_scores), color="k", linestyle="dashed", linewidth=2, label="Mean")
        plt.xlabel("Predictive Entropy", fontsize=18)
        plt.ylabel("Frequency", fontsize=18)
        plt.title("Histogram of Predictive Entropy", fontsize=20)
        plt.legend()
        filename = self._save_plot("dist_entropy")
        self._plot_file_names["entropy_distrubution"] = filename

    def plot_interactive_dist_entropy_scores(self, entropy_scores):
        fig = plotly_graph_objects.Figure()

        fig.add_trace(plotly_graph_objects.Histogram(
            x=entropy_scores,
            name="Predictive Entropy",
            marker_color="blue"),
        )

        fig.update_layout(title_text="Histogram of Predictive Entropy",
                          xaxis_title_text="Predictive Entropy",
                          yaxis_title_text="Frequency",
                          bargap=0.2)

        filename = self._save_interactive_plot_html("dist_entropy", fig)
        self._interactive_plot_file_names["entropy_distrubution"] = filename

    def high_uncertain_inputs(self, entropy_scores: list, x_test: np.ndarray, num_samples: int = 25) -> None:
        """Plots the most uncertain inputs based on entropy scores."""
        num_samples = min(num_samples, len(x_test))
        # Sort the indices of the entropy scores in descending order
        sorted_indices = np.argsort(entropy_scores)[::-1]

        # Plot the most uncertain examples
        plt.figure(figsize=(8, 8))
        for i in range(num_samples):
            index = sorted_indices[i]
            plt.subplot(5, 5, i + 1)
            if x_test.shape[-1] == 1:  # Grayscale images
                plt.imshow(x_test[i].reshape(x_test.shape[1], x_test.shape[2]), cmap="gray")
            else:   # RGB images
                plt.imshow(x_test[i].reshape(x_test.shape[1], x_test.shape[2], x_test.shape[3]))
            plt.title(f"Entropy: {entropy_scores[sorted_indices[i]]:.2f}")
            plt.axis("off")
        plt.tight_layout()
        filename = self._save_plot("high_uncertain_inputs")
        self._plot_file_names["higly_uncertain_inputs"] = filename

    def high_interactive_uncertain_inputs(self, entropy_scores, x_test, num_samples=25):
        num_samples = min(num_samples, len(x_test))
        # Sort the indices of the entropy scores in descending order
        sorted_indices = np.argsort(entropy_scores)[::-1]
        plt.figure(figsize=(8, 8))
        for i in range(num_samples):
            index = sorted_indices[i]
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
            plt.title(f"Entropy: {entropy_scores[sorted_indices[i]]:.2f}")
            plt.axis("off")
        plt.tight_layout()

        filename = self._save_interactive_plot("high_uncertain_inputs")
        self._interactive_plot_file_names["higly_uncertain_inputs"] = filename

    def plot_predictive_conf_entropy_scores(self, predictive_confidence: list, entropy_scores: list) -> None:
        """Plots predictive confidence vs entropy scores."""
        plt.figure(figsize=(20, 10))
        plt.scatter(
            predictive_confidence,
            entropy_scores,
            c=entropy_scores,
            cmap="viridis",
            alpha=0.5,
        )
        plt.colorbar(label="Predictive Entropy")
        plt.xlabel("Predictive Confidence", fontsize=18)
        plt.ylabel("Entropy Score", fontsize=18)
        plt.title("Predictive Confidence vs Entropy Score", fontsize=18)
        filename = self._save_plot("pred_vs_entropy")
        #plt.show()
        self._plot_file_names["prediction_vs_entrophy"] = filename

    def plot_interactive_predictive_conf_entropy_scores(self, predictive_confidence, entropy_scores):
        fig = plotly_graph_objects.Figure(
            data=plotly_graph_objects.Scatter(
                x=predictive_confidence,
                y=entropy_scores,
                mode='markers',
                marker=dict(
                    color=entropy_scores,
                    colorbar=dict(title='Predictive Entropy'),
                    colorscale='Viridis'
                )
            )
        )

        fig.add_annotation(
            x=max(predictive_confidence),
            y=max(entropy_scores),
            text=f"Total plots: {len(predictive_confidence)}",
            showarrow=False,
            align="center",
            borderwidth=5,
            borderpad=5,
            arrowcolor="rgb(71, 71, 71)",
            bordercolor="rgb(71, 71, 71)",
            bgcolor="rgb(255, 184, 0)",
        )

        index_x_max = np.argmax(predictive_confidence)
        x_max = predictive_confidence[index_x_max]
        y_x_max = entropy_scores[index_x_max]

        fig.add_annotation(
            x=x_max,
            y=y_x_max,
            text=f"x:{x_max:.2f}, y:{y_x_max:.2f}",
            showarrow=True,
            align="center",
            borderwidth=5,
            borderpad=5,
            arrowcolor="rgb(71, 71, 71)",
            bordercolor="rgb(71, 71, 71)",
            bgcolor="rgb(255, 184, 0)",
        )

        index_y_max = np.argmax(entropy_scores)
        y_max = entropy_scores[index_y_max]
        x_y_max = predictive_confidence[index_y_max]

        fig.add_annotation(
            x=x_y_max,
            y=y_max,
            text=f"x:{x_y_max:.2f}, y:{y_max:.2f}",
            showarrow=True,
            align="center",
            borderwidth=5,
            borderpad=5,
            arrowcolor="rgb(71, 71, 71)",
            bordercolor="rgb(71, 71, 71)",
            bgcolor="rgb(255, 184, 0)",
        )

        fig.update_layout(
            title="Predictive Confidence vs Entropy Score",
            xaxis_title="Predictive Confidence",
            yaxis_title="Entropy Score",
            updatemenus=[{
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.1,
                "yanchor": "top"
            }],
            xaxis=dict(rangeslider=dict(visible=True), type="linear")
        )

        filename = self._save_interactive_plot_html("prediction_vs_entrophy", fig)
        self._interactive_plot_file_names["prediction_vs_entrophy"] = filename

    def _save_plot(self, filename: str) -> str:
        """Saves the plot to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot(self, filename: str) -> str:
        """Saves an interactive plot to a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
        return f"{self.plot_dir}/{filename}"

    def _save_interactive_plot_html(self, filename: str, data_info) -> str:
        """Saves an interactive plot to an HTML file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.html"
        plotly.offline.plot(data_info, filename=os.path.join(self.plot_dir, filename), include_plotlyjs=True,
                            auto_open=False)

        return f"{self.plot_dir}/{filename}"

    @property
    def plot_file_names(self) -> dict:
        """Returns a dictionary of plot file names."""
        if not isinstance(self._plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for file name should be dict"
                )
            )
        if len(self._plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing file names in dict")
            )
        return self._plot_file_names

    @property
    def interactive_plot_file_names(self) -> dict:
        """Returns a dictionary of interactive plot file names."""
        if not isinstance(self._interactive_plot_file_names, dict):
            raise ValueError(
                StringStyling.box_style(
                    message="visualizer: Wrong datatype for file name should be dict"
                )
            )
        if len(self._interactive_plot_file_names) < 1:
            raise ValueError(
                StringStyling.box_style(message="visualizer: missing file names in dict")
            )
        return self._interactive_plot_file_names
