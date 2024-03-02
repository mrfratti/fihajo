import logging
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LambdaCallback
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.utils import Progbar
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from .model_utils import create_mnist_model, load_and_preprocess_mnist
from .visualization import plot_training_results
from uncertainty_wizard.models import StochasticMode


def train_model(args):
    (x_train, y_train), _ = load_and_preprocess_mnist()
    # Recreate model architecture
    stochastic_mode = StochasticMode()
    model = create_mnist_model(stochastic_mode)

    if args.save_path is None:
        args.save_path = 'data/models/adv_model_weights.h5' if args.adv_train else 'data/models/model_weights.h5'

    if args.adv_train:
        print("Starting adversarial training...")

        loss_object = CategoricalCrossentropy(from_logits=True)
        train_loss = Mean(name='train_loss')
        train_accuracy = CategoricalAccuracy(name='train_accuracy')

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            # Initialize progress bar
            progbar = Progbar(target=len(x_train) // args.batch, unit_name='batch')
            train_loss.reset_states()
            train_accuracy.reset_states()

            for batch_index, (x_batch, y_batch) in enumerate(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch)):
                with tf.GradientTape() as tape:
                    # Generate adversarial examples
                    x_batch_adv = projected_gradient_descent(model.inner, x_batch, args.eps, 0.01, 40, np.inf)
                    predictions = model.inner(x_batch_adv, training=True)
                    loss = loss_object(y_batch, predictions)

                gradients = tape.gradient(loss, model.inner.trainable_variables)
                model.inner.optimizer.apply_gradients(zip(gradients, model.inner.trainable_variables))

                train_loss(loss)
                train_accuracy(y_batch, predictions)

                # Update progress bar
                progbar.update(batch_index + 1, values=[("loss", train_loss.result()),
                                                        ("accuracy", train_accuracy.result())])

            print(f"Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()}")

            # Reset metrics at the end of each epoch
            train_loss.reset_states()
            train_accuracy.reset_states()

            print(f"End of Epoch {epoch+1}")
            print("Adversarial training completed. Saving model weights to: {}".format(args.save_path))

    else:
        print("Starting standard training...")
        callbacks = [
            EarlyStopping(patience=3, verbose=1),
            TensorBoard(log_dir='./data/logs', histogram_freq=1),
            LambdaCallback(on_epoch_end=lambda epoch, logs: logging.info(
                f"Epoch {epoch + 1} completed. Loss: {logs['loss']}, Accuracy: {logs['accuracy']}"))
        ]

        history = model.fit(x_train,
                            y_train,
                            validation_split=0.1,
                            batch_size=args.batch,
                            epochs=args.epochs,
                            verbose=1,
                            callbacks=callbacks)

        print("Training completed. Saving model weights to: {}".format(args.save_path))
        plot_training_results(history)

    weights_dir = os.path.dirname(args.save_path)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    model.inner.save_weights(args.save_path)
    print("Model weights saved to: {}".format(args.save_path))
