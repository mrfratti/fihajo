from model_utils import load_and_preprocess_mnist, create_mnist_model
from visualization import plot_confusion_matrix, plot_predictions
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode

def evaluate_model(args):
    # Load the test data
    _, (x_test, y_test) = load_and_preprocess_mnist()

    # Recreate model architecture
    stochastic_mode = StochasticMode()
    model = create_mnist_model(stochastic_mode)

    # Load weights into the model
    model.inner.load_weights('data/model_weights.h5')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    print('\nTest accuracy:', test_acc)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    plot_predictions(model.inner, x_test)
    plot_confusion_matrix(y_true, y_pred_classes, classes=[str(i) for i in range(10)])