import tensorflow as tf
from .model_utils import load_and_preprocess_mnist, create_mnist_model
from src.visualization.visualization import plot_confusion_matrix, plot_predictions, plot_adversarial_examples, plot_accuracy_comparison
import numpy as np
from uncertainty_wizard.models import StochasticMode
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


def evaluate_model(args):
    # Load the test data
    _, (x_test, y_test) = load_and_preprocess_mnist()

    # If y_test is one-hot encoded
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)  # Convert to sparse format

    # Recreate model architecture
    stochastic_mode = StochasticMode()
    model = create_mnist_model(stochastic_mode)

    # Load weights into the model
    model.inner.load_weights('data/model_weights.h5')
    #test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    # Standard evaluation
    test_accuracy = SparseCategoricalAccuracy()

    # Evaluate on clean examples
    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128):
        predictions = model.predict(x_batch)
        test_accuracy.update_state(y_batch, predictions)

    accuracy_clean = test_accuracy.result().numpy() * 100 # Convert to percentage
    print(f'\nTest accuracy on clean examples: {accuracy_clean}')

    # Adversarial evaluation settings
    eps = args.eps # Epsilon for adversarial perturbation
    test_accuracy_fgsm = SparseCategoricalAccuracy()
    test_accuracy_pgd = SparseCategoricalAccuracy()

    # Evaluate on adversarial examples (FGSM and PGD)
    for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128):
        # FGSM examples
        x_adv_fgsm = fast_gradient_method(model.inner, x_batch, eps, np.inf)
        predictions_fgsm = model.predict(x_adv_fgsm)
        test_accuracy_fgsm.update_state(y_batch, predictions_fgsm)

        # PGD examples
        x_adv_pdg = projected_gradient_descent(model.inner, x_batch, eps, 0.01, 40, np.inf)
        predictions_pdg = model.predict(x_adv_pdg)
        test_accuracy_pgd.update_state(y_batch, predictions_pdg)

    accuracy_fgsm = test_accuracy_fgsm.result().numpy() * 100  # Convert to percentage
    accuracy_pgd = test_accuracy_pgd.result().numpy() * 100  # Convert to percentage

    print(f'\nTest accuracy on FGSM adversarial examples: {accuracy_fgsm}')
    print(f'\nTest accuracy on PDG adversarial examples: {accuracy_pgd}')

    #print('\nTest accuracy:', test_acc)
    #print('\nTest loss', test_loss)

    # Plot the accuracies:
    accuracies = [accuracy_clean, accuracy_fgsm, accuracy_pgd]
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    #y_true = np.argmax(y_test, axis=1)
    plot_predictions(model.inner, x_test)
    plot_confusion_matrix(y_test, y_pred_classes, classes=[str(i) for i in range(10)])
    plot_adversarial_examples(model, x_test, args.eps, num_samples=25)
    plot_accuracy_comparison(accuracies)