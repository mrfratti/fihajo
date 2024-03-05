import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


class VisualizeTraining:
    def __init__(self, plot_dir='data/plots/training'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_training_results(self, history):
        # Plot training & validation accuracy and loss
        # Accuracy
        plt.figure(dpi=1200)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        self._save_plot('val_acc_and_loss')
        plt.show()

    def plot_adversarial_training_results(self, history):
        plt.figure(dpi=1200)
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Adversarial Training Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Adversarial Training Loss')
        plt.legend()
        self._save_plot('adv_train_acc_loss')
        plt.show()

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()


class VisualizeEvaluation:
    def __init__(self, plot_dir='data/plots/evaluation'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_predictions(self, model, x_test, y_true, num_samples=25):
        predictions = model.predict(x_test[:num_samples])
        predicted_labels = np.argmax(predictions, axis=1)
        # true_label = np.argmax(y_test, axis=1) if np.ndim(y_test) > 1 else y_test
        plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            plt.title(f'True: {y_true[i]}, Predicted: {predicted_labels[i]}')
            plt.axis('off')
        plt.tight_layout()
        self._save_plot('predictions')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(20, 15))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        self._save_plot('confusion_matrix')
        plt.show()

    def plot_classification_report(self, y_true, y_pred_classes, output_dict=True):
        report = classification_report(y_true, y_pred_classes, output_dict=output_dict)
        df_report = pd.DataFrame(report).transpose()
        df_report.drop('support', errors='ignore', inplace=True)
        plt.figure(figsize=(20, 10))
        sns.heatmap(df_report[['precision', 'recall', 'f1-score']].T, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Classification Report')
        self._save_plot('classification_report')
        plt.show()

    def plot_adversarial_examples(self, model, x_test, eps, num_samples=25):
        # Generate FGSM adversarial examples
        x_adv_fgsm = fast_gradient_method(model.inner, x_test[:num_samples], eps, np.inf)
        predictions_fgsm = np.argmax(model.predict(x_adv_fgsm), axis=1)

        # predictions_clean = np.argmax(model.predict(x_test[:num_samples]), axis=1)

        # Generate PGD adversarial examples
        x_adv_pgd = projected_gradient_descent(model.inner, x_test[:num_samples], eps, 0.01, 40, np.inf)
        predictions_pgd = np.argmax(model.predict(x_adv_pgd), axis=1)

        plt.figure(figsize=(2 * num_samples, 6))

        # plt.figure(figsize=(20, 10))
        for i in range(num_samples):
            # Original images
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(x_test[i], cmap='gray')
            plt.title(f'Clean\nPred: {np.argmax(model.predict(x_test[i:i + 1]), axis=1)[0]}', fontsize=9)
            plt.axis('off')

            # Plot FGSM adversarial images
            plt.subplot(3, num_samples, num_samples + i + 1)
            plt.imshow(x_adv_fgsm[i], cmap='gray')
            plt.title(f'FGSM\nPred: {predictions_fgsm[i]}', fontsize=9)
            plt.axis('off')

            # Plot PGD adversarial images
            plt.subplot(3, num_samples, 2 * num_samples + i + 1)
            plt.imshow(x_adv_pgd[i], cmap='gray')
            plt.title(f'PGD\nPred: {predictions_pgd[i]}', fontsize=9)
            plt.axis('off')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

        self._save_plot('adv_examples')
        plt.show()

    def plot_accuracy_comparison(self, accuracies, labels=['Clean', 'FGSM', 'PGD']):
        plt.figure(figsize=(8, 6))
        bar_positions = np.arange(len(accuracies))
        plt.bar(bar_positions, accuracies, color=['blue', 'green', 'red'])
        plt.xticks(bar_positions, labels)
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy: Clean vs Adversarial Examples')
        plt.ylim(0, max(accuracies) + 10)
        # plt.ylim(0, 110)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 2, f"{acc:.2f}", ha='center', va='bottom')

        self._save_plot('accuracy_comparison')
        plt.show()

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()


class VisualizeUncertainty:
    def __init__(self, plot_dir='./data/plots/analyze'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_pcs_mean_softmax(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        # Results is a list of tuples, where each tuple contains (predictions, scores)
        # pcs_scores, mean_softmax_scores = results[0][1], results[1][1]

        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.histplot(pcs_scores, bins=50, kde=True, color='skyblue')
        plt.xlabel('PCS Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of PCS Scores')
        plt.subplot(1, 2, 2)
        sns.histplot(mean_softmax_scores, bins='auto', kde=True, color='lightgreen')
        plt.xlabel('Mean Softmax Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Mean Softmax Scores')
        plt.tight_layout()
        self._save_plot('pcs_meansoftmax')
        plt.show()

    def plot_distribution_pcs_ms_scores(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        plt.figure(figsize=(10, 10))
        sns.histplot(pcs_scores, bins=50, alpha=0.7, color='blue', kde=True, label='PCS')
        sns.histplot(mean_softmax_scores, bins=50, alpha=0.7, color='red', kde=True, label='Mean Softmax')
        plt.xlabel('Predictive Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of PCS and Mean Softmax Scores')
        plt.legend()
        self._save_plot('dist_pcs_meansoftmax')
        plt.show()

    def plot_pcs_ms_inverse(self, pcs_mean_softmax_scores):
        pcs_scores, mean_softmax_scores = pcs_mean_softmax_scores
        pcs_inverse = 1 - pcs_scores
        uncertainty_threshold_pcs = np.percentile(pcs_inverse, 95)
        mean_softmax_inverse = 1 - mean_softmax_scores
        uncertainty_threshold_mean_softmax = np.percentile(mean_softmax_inverse, 95)

        plt.figure(figsize=(20, 10))
        # First subplot for PCS scores
        plt.subplot(1, 2, 1)
        sns.histplot(pcs_inverse, bins='auto', kde=True, color='skyblue')
        plt.axvline(uncertainty_threshold_pcs, color='r', linestyle='dashed', linewidth=2,
                    label=f'95th percentile: {uncertainty_threshold_pcs:.2f}')
        plt.axvline(np.mean(pcs_inverse), color='k', linestyle='dashed', label=f'Mean PCS: {np.mean(pcs_scores):.2f}')
        plt.xlabel('Prediction Confidence Score (PCS)', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of PCS Scores as Uncertainty', fontsize=18)
        plt.legend()

        # Second subplot for Mean Softmax scores
        plt.subplot(1, 2, 2)
        sns.histplot(mean_softmax_inverse, bins='auto', kde=True, color='lightgreen')
        plt.axvline(uncertainty_threshold_mean_softmax, color='r', linestyle='dashed',
                    label=f'95th percentile: {uncertainty_threshold_mean_softmax:.2f}')
        plt.axvline(np.mean(mean_softmax_inverse), color='k', linestyle='dashed',
                    label=f'Mean Softmax: {np.mean(mean_softmax_scores):.2f}')
        plt.xlabel('Mean Softmax Score', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Distribution of Mean Softmax Scores as Uncertainty', fontsize=18)
        plt.legend()
        plt.tight_layout()
        self._save_plot('pcs_ms_inverse')
        plt.show()

    def plot_dist_entropy_scores(self, entropy_scores):
        plt.figure(figsize=(20, 10))
        sns.histplot(entropy_scores, bins=50, kde=True, color='blue', label='Clean Data')
        plt.axvline(np.mean(entropy_scores), color='k', linestyle='dashed', linewidth=2, label='Mean')
        plt.xlabel('Predictive Entropy', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title('Histogram of Predictive Entropy', fontsize=18)
        plt.legend()
        self._save_plot('dist_entropy')
        plt.show()

    def high_uncertain_inputs(self, entropy_scores, x_test, num_samples=25):
        # Sort the indices of the entropy scores in descending order
        sorted_indices = np.argsort(entropy_scores)[::-1]

        # Plot the most uncertain examples
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[sorted_indices[i]], cmap='gray')
            plt.title(f'Entropy: {entropy_scores[sorted_indices[i]]:.2f}')
            plt.axis('off')
        plt.tight_layout()
        self._save_plot('high_uncertain_inputs')
        plt.show()

    def plot_predictive_conf_entropy_scores(self, predictive_confidence, entropy_scores):
        plt.figure(figsize=(20, 10))
        plt.scatter(predictive_confidence, entropy_scores, c=entropy_scores, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Predictive Entropy')
        plt.xlabel('Predictive Confidence')
        plt.ylabel('Entropy Score')
        plt.title('Predictive Confidence vs Entropy Score')
        self._save_plot('pred_vs_entropy')
        plt.show()

    def plot_tsne_entropy(self, tsne_results, entropy_scores):
        # Plot the t-SNE results
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=entropy_scores, cmap='viridis_r', alpha=0.6)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Entropy Score')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('t-SNE Visualization of Predictive Entropy')
        self._save_plot('tsne_entropy')
        plt.show()

    def _save_plot(self, filename):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{filename}_{timestamp}.png"
        plt.savefig(os.path.join(self.plot_dir, filename))
        plt.close()
