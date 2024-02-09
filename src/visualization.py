import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_training_results(history):
    # Plot training & validation accuracy and loss
    # Accuracy
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

    # Generate a filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"val_acc_and_loss_{timestamp}.png"

    # Save the plot
    plot_dir = './data/plots/train'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_predictions(model, x_test, num_samples=25):
    predictions = model.predict(x_test[:num_samples])
    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_labels[i]}')
        plt.axis('off')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"predictions_{timestamp}.png"

    plot_dir = './data/plots/evaluate'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"confusion_matrix_{timestamp}.png"

    plot_dir = './data/plots/evaluate'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_uncertainty_distribution(results):
    uncertainty_scores = results[1][1]
    plt.figure(figsize=(8, 6))
    plt.hist(uncertainty_scores, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Uncertainty Scores')
    plt.xlabel('Uncertainty Score')
    plt.ylabel('Frequency')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"uncertainty_{timestamp}.png"

    plot_dir = './data/plots/analyze'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_distribution_pcs_ms(results):
    # Results is a list of tuples, where each tuple contains (predictions, scores)
    pcs_scores, mean_softmax_scores = results[0][1], results[1][1]

    plt.subplot(1, 2, 1)
    plt.hist(pcs_scores, bins=50, alpha=0.7)
    plt.xlabel('PCS Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of PCS Scores')

    plt.subplot(1, 2, 2)
    plt.hist(mean_softmax_scores, bins=50, alpha=0.7)
    plt.xlabel('Mean Softmax Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mean Softmax Scores')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"distribution_pcs_ms_{timestamp}.png"

    plot_dir = './data/plots/analyze'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_pcs_ms_scores(results, num_samples=25):
    # PCS scores
    pcs_scores, mean_softmax_scores = results[0][1][:num_samples], results[1][1][:num_samples]
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.bar(range(num_samples), pcs_scores)
    plt.xlabel('Sample')
    plt.ylabel('PCS Score')
    plt.title('PCS Scores for Test Samples')

    # Mean Softmax scores
    plt.subplot(1, 2, 2)
    plt.bar(range(num_samples), mean_softmax_scores)
    plt.xlabel('Sample')
    plt.ylabel('Mean Softmax Score')
    plt.title('Mean Softmax Scores for Test Samples')
    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"pcs_ms_scores_{timestamp}.png"

    plot_dir = './data/plots/analyze'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_distribution_pcs_ms_scores(results):
    pcs_predictions, pcs_confidences = results[0][0], results[0][1]
    mean_softmax_predictions, mean_softmax_confidences = results[1][0], results[1][1]

    plt.hist(pcs_confidences, bins=50, alpha=0.7, color='blue', label='PCS')
    plt.hist(mean_softmax_confidences, bins=50, alpha=0.7, color='red', label='Mean Softmax')
    plt.xlabel('Predictive Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predictive Confidence Scores')
    plt.legend()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"distribution_pcs_ms_scores_{timestamp}.png"

    plot_dir = './data/plots/analyze'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()
