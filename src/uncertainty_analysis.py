from model_utils import create_mnist_model, load_and_preprocess_mnist
from visualization import (plot_uncertainty_distribution, plot_predictions, plot_distribution_pcs_ms,
                           plot_pcs_ms_scores,
                           plot_distribution_pcs_ms_scores)
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode


def analyze_uncertainty(args):
    _, (x_test, _) = load_and_preprocess_mnist()

    # Recreate the model architecture
    stochastic_mode = StochasticMode()
    model = create_mnist_model(stochastic_mode)

    # Load the weights into the model
    model.inner.load_weights('data/model_weights.h5')

    # Define quantifiers and perform quantified prediction
    quantifiers = ['pcs', 'mean_softmax']
    results = model.predict_quantified(x_test,
                                       quantifier=quantifiers,
                                       batch_size=args.batch,
                                       sample_size=32,
                                       verbose=1)

    # Plot results
    plot_uncertainty_distribution(results)
    plot_predictions(model.inner, x_test)
    plot_distribution_pcs_ms(results)
    plot_pcs_ms_scores(results)
    plot_distribution_pcs_ms_scores(results)
