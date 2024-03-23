import argparse
import logging
import os
import platform
import sys
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LambdaCallback
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy, CategoricalAccuracy
from tensorflow.keras.utils import Progbar
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from src.visualization.visualization import VisualizeTraining

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Trainer:
    """
    Trainer class for training and evaluating a model using both normal and adversarial training methods.
    """

    def __init__(self, model_builder, train_dataset, test_dataset, args: argparse.Namespace) -> None:
        """
        Initializes the Trainer object with the model builder, dataset handler, and command-line arguments,
        and prepares the model for training
        :param args: command-line arguments specifying training parameters.
        :param model_builder: An instance responsible for building the model
        :param train_dataset: A tuple containing the training data and labels (x_train, y_train)
        :param test_dataset: A tuple containing the test data and labels (x_test, y_test)
        """
        self.args = args
        self.model = model_builder.create_model()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_object = CategoricalCrossentropy(from_logits=False)

    def train(self):
        """
        Selects the training method based on whether adversarial training is enabled via command-line arguments.
        """
        if self.args.adv:
            message = "Adversarial training enabled.\n"
            self.loading_effect(duration=15, message=message)
            logging.info(f"Starting adversarial training on {self.args.dataset} dataset")
            self.adversarial_training()
        else:
            message = f"Getting ready for training the model on {self.args.dataset} dataset\n"
            self.loading_effect(duration=15, message=message)
            logging.info("Starting training.")
            self.training()

    def training(self):
        """
        Executes standard training procedure for the MNIST model, including callbacks for early stopping and logging.
        """

        x_train, y_train = self.train_dataset

        callbacks = [
            EarlyStopping(patience=3, verbose=1),
            TensorBoard(log_dir="./data/logs", histogram_freq=1),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: logging.info(
                    f"Epoch {epoch + 1} completed. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                )
            ),
        ]

        history = self.model.fit(
            x_train,
            y_train,
            validation_split=0.1,
            batch_size=self.args.batch,
            epochs=self.args.epochs,
            verbose=1,
            callbacks=callbacks,
        )

        visualizer = VisualizeTraining()
        visualizer.plot_training_results(history)

    def adversarial_training(self):
        """
        Executes adversarial training, incorporating attacks during training to improve model robustness.
        """
        logging.info("Starting adversarial training...")
        adv_training_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        train_loss = Mean(name="train_loss")
        train_accuracy = CategoricalAccuracy(name="train_accuracy")
        val_loss = Mean(name="val_loss")
        val_accuracy = CategoricalAccuracy(name="val_accuracy")

        if platform.system() == "Darwin" and platform.processor() == "arm":
            optimizer = tf.keras.optimizers.legacy.Adadelta()
        else:
            optimizer = tf.keras.optimizers.Adadelta()

        x_train, y_train = self.train_dataset
        x_val, y_val = self.test_dataset

        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            # Initialize progress bar
            progbar = Progbar(target=len(x_train) // self.args.batch, unit_name="batch")
            train_loss.reset_states()
            train_accuracy.reset_states()

            for batch_index, (x_batch, y_batch) in enumerate(
                tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
                    self.args.batch
                )
            ):
                with tf.GradientTape() as tape:
                    # Generate adversarial examples
                    x_batch_adv = projected_gradient_descent(
                        self.model.inner, x_batch, self.args.eps, 0.01, 40, np.inf
                    )
                    predictions = self.model.inner(x_batch_adv, training=True)
                    loss = self.loss_object(y_batch, predictions)

                gradients = tape.gradient(loss, self.model.inner.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, self.model.inner.trainable_variables)
                )

                train_loss(loss)
                train_accuracy(y_batch, predictions)

                # Update progress bar
                progbar.update(
                    batch_index + 1,
                    values=[
                        ("loss", train_loss.result()),
                        ("accuracy", train_accuracy.result()),
                    ],
                )

            adv_training_history["loss"].append(train_loss.result().numpy())
            adv_training_history["accuracy"].append(train_accuracy.result().numpy())

            # Validation phase after each epoch
            for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
                (x_val, y_val)
            ).batch(self.args.batch):
                x_batch_adv = projected_gradient_descent(
                    self.model.inner, x_batch, self.args.eps, 0.01, 40, np.inf
                )
                val_predictions = self.model.inner(
                    x_batch_adv, training=False
                )  # False for evaluation
                batch_val_loss = self.loss_object(y_batch, val_predictions)
                val_loss(batch_val_loss)
                val_accuracy(y_batch, val_predictions)

            adv_training_history["val_loss"].append(val_loss.result().numpy())
            adv_training_history["val_accuracy"].append(val_accuracy.result().numpy())

            logging.info(
                f"Epoch {epoch + 1} completed. Loss: {train_loss.result().numpy():.3f}, "
                f"Accuracy: {train_accuracy.result().numpy():.3f}, "
                f"Validation Loss: {val_loss.result().numpy():.3f}, "
                f"Validation Accuracy: {val_accuracy.result().numpy():.3f}"
            )

            # Reset metrics at the end of each epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()

        logging.info("Adversarial training completed")

        visualizer = VisualizeTraining()
        visualizer.plot_adversarial_training_results(adv_training_history)

    def save_model(self):
        user_input = input(
            "Enter a path to save the model or press Enter to use the default path: "
        ).strip()
        save_path = user_input if user_input else self._default_save_path()
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.inner.save_weights(save_path)
            logging.info(f"Model weights saved to: {save_path}")
        except Exception as e:
            logging.error(f"Error saving the model weights: {e}")
            raise

    def _default_save_path(self) -> str:
        """Generate a default save path for the model based on training type"""
        base_path = "data/models"
        file_name = "adv_model.weights.h5" if self.args.adv else "model.weights.h5"
        return os.path.join(base_path, file_name)

    def loading_effect(self, duration=30, message=""):
        print(message, end="")
        for _ in range(duration):
            for cursor in "|/-\\":
                sys.stdout.write(cursor)
                sys.stdout.flush()
                time.sleep(0.1)  # Adjust sleep time as needed
                sys.stdout.write("\b")
        print("Done!")


if __name__ == "__main__":
    pass
