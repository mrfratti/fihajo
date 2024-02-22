import datetime
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


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
    y_pred = np.argmax(model.predict(x_test), axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_pred[i]}, Predicted: {predicted_labels[i]}')
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
    plt.figure(figsize=(10, 10))
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


def plot_adversarial_examples(model, x_test, eps, num_samples=25):

    # Generate FGSM adversarial examples
    x_adv_fgsm = fast_gradient_method(model.inner, x_test[:num_samples], eps, np.inf)
    predictions_fgsm = np.argmax(model.predict(x_adv_fgsm), axis=1)


    #predictions_clean = np.argmax(model.predict(x_test[:num_samples]), axis=1)

    # Generate PGD adversarial examples
    x_adv_pgd = projected_gradient_descent(model.inner, x_test[:num_samples], eps, 0.01, 40, np.inf)
    predictions_pgd = np.argmax(model.predict(x_adv_pgd), axis=1)

    plt.figure(figsize=(2 * num_samples, 6))

    #plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        # Original images
        plt.subplot(2, num_samples, i+1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f'Clean\nPred: {np.argmax(model.predict(x_test[i:i+1]), axis=1)[0]}', fontsize=9)
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

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"org_vs_adv_{timestamp}.png"

    plot_dir = './data/plots/evaluate'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))

    plt.show()


def plot_accuracy_comparison(accuracies, labels=['Clean', 'FGSM', 'PGD']):
    plt.figure(figsize=(8, 6))
    bar_positions = np.arange(len(accuracies))
    plt.bar(bar_positions, accuracies, color=['blue', 'green', 'red'])
    plt.xticks(bar_positions, labels)
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy: Clean vs Adversarial Examples')
    plt.ylim(0, 110)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 2, f"{acc:.2f}", ha='center', va='bottom')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"compare_clean_vs_adv_{timestamp}.png"

    plot_dir = './data/plots/evaluate'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, filename))


    plt.show()


def plot_uncertainty_distribution(results):
    uncertainty_scores = results[1][1]
    plt.figure(figsize=(10, 10))
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

    plt.figure(figsize=(10,10))
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


def plot_distribution_pcs_ms_scores(results):
    pcs_predictions, pcs_confidences = results[0][0], results[0][1]
    mean_softmax_predictions, mean_softmax_confidences = results[1][0], results[1][1]

    plt.figure(figsize=(10, 10))
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
