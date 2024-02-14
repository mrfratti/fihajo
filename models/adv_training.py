import platform
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from uncertainty_wizard.models import StochasticMode
from tensorflow.keras.utils import Progbar
from tensorflow.keras.metrics import CategoricalAccuracy
from .model_utils import create_adv_mnist_model, load_and_preprocess_mnist


def adv_training(args):
    (x_train, y_train), _ = load_and_preprocess_mnist()
    stochastic_mode = StochasticMode()
    model = create_adv_mnist_model(stochastic_mode)

    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        optimizer = tf.keras.optimizers.legacy.Adam()
    else:
        optimizer = tf.keras.optimizers.Adam()

    loss_object = CategoricalCrossentropy(from_logits=True)
    train_loss = Mean(name='train_loss')
    train_accuracy = CategoricalAccuracy(name='train_accuracy')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Initialization of progress bar
        progbar = Progbar(target=len(x_train) // args.batch, unit_name='batch')
        train_accuracy.reset_states()

        for batch_index, (x_batch, y_batch) in enumerate(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch)):
            with tf.GradientTape() as tape:
                # Generate adversarial examples
                x_batch_adv = projected_gradient_descent(model.inner, x_batch, args.eps, 0.01, 40, np.inf)
                predictions = model.inner(x_batch_adv, training=True)
                loss = loss_object(y_batch, predictions)

            gradients = tape.gradient(loss, model.inner.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.inner.trainable_variables))

            train_loss(loss)
            train_accuracy(y_batch, predictions)

            # Update metrics
            #loss_metric.update_state(loss)
            #accuracy_metric.update_state(y_batch, logits)

            # Update progress bar
            progbar.update(batch_index + 1, values=[("loss", train_loss.result()),
                                                    ("accuracy", train_accuracy.result())])

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}')

        # Reset metrics at the end of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        print(f"End of Epoch {epoch + 1}")

    # Save the model or weights if needed
    model.inner.save_weights('data/adv_model_weights.h5')
    print("Adversarial training completed and model saved.")
