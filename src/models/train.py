import argparse
import logging
import os
import platform
import sys
import tensorflow as tf
import numpy as np

from keras.callbacks import (EarlyStopping, TensorBoard, LambdaCallback)
from keras.losses import CategoricalCrossentropy
from keras.metrics import (Mean, CategoricalAccuracy)
from keras.utils import Progbar
from keras import optimizers
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from src.cli.string_styling import StringStyling
from src.visualization.visualization import VisualizeTraining
from src.weight_processing.weight_manager import WeightManager

from report_interactive.test2_render_html_for_eChart import build_list_info, create_cheack_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message).80s")


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
        self._weightmanager = WeightManager()
        self._weightmanager.current_model = model_builder
        self.model = self._weightmanager.current_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_object = CategoricalCrossentropy(from_logits=False)
        self._plot_file_names = {}

    @property
    def plot_file_names(self) -> dict:
        """List of plot filenames"""
        return self._plot_file_names

    def train(self):
        """
        Selects the training method based on whether adversarial training is enabled via command-line arguments.
        """
        create_cheack_file()
        if self.args.adv:
            message = "Adversarial training enabled.\n"
            self._weightmanager.loading_effect(duration=15, message=message)
            logging.info("Starting adversarial training on %s dataset", self.args.dataset)
            build_list_info("adversarial_training")
            self.adversarial_training()
        else:
            message = f"Getting ready for training the model on {self.args.dataset} dataset\n"
            self._weightmanager.loading_effect(duration=15, message=message)
            logging.info("Starting training.")
            build_list_info("training")
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
                    f"\n Epoch {epoch + 1} completed. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}"
                )
            )
        ]

        history = self.model.fit(x_train, y_train,
                                 validation_split=0.1,
                                 batch_size=self.args.batch,
                                 epochs=self.args.epochs,
                                 verbose=1,
                                 callbacks=callbacks)

        visualizer = VisualizeTraining()
        visualizer.plot_training_results(history)
        self._plot_file_names.update(visualizer.plot_file_names)

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
            optimizer = optimizers.legacy.Adadelta()  # pylint: disable=E1101
        else:
            optimizer = optimizers.Adadelta()

        x_train, y_train = self.train_dataset
        x_val, y_val = self.test_dataset

        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            # Initialize progress bar
            progbar = Progbar(target=len(x_train) // self.args.batch, unit_name="batch")
            train_loss.reset_state()
            train_accuracy.reset_state()

            for batch_index, (x_batch, y_batch) in enumerate(
                    tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.args.batch)):
                with tf.GradientTape() as tape:
                    # Generate adversarial examples
                    x_batch_adv = projected_gradient_descent(
                        self.model.inner, x_batch, self.args.eps, 0.01, 40, np.inf)
                    predictions = self.model.inner(x_batch_adv, training=True)
                    loss = self.loss_object(y_batch, predictions)

                gradients = tape.gradient(loss, self.model.inner.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.inner.trainable_variables))

                train_loss(loss)
                train_accuracy(y_batch, predictions)

                # Update progress bar
                progbar.update(
                    batch_index + 1,
                    values=[
                        ("loss", np.array(train_loss.result())),
                        ("accuracy", np.array(train_accuracy.result())),
                    ]
                )

            adv_training_history["loss"].append(np.array(train_loss.result()))
            adv_training_history["accuracy"].append(np.array(train_accuracy.result()))

            # Validation phase after each epoch
            for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.args.batch):
                x_batch_adv = projected_gradient_descent(
                    self.model.inner, x_batch, self.args.eps, 0.01, 40, np.inf)
                val_predictions = self.model.inner(x_batch_adv, training=False)
                batch_val_loss = self.loss_object(y_batch, val_predictions)
                val_loss(batch_val_loss)
                val_accuracy(y_batch, val_predictions)

            adv_training_history["val_loss"].append(np.array(val_loss.result()))
            adv_training_history["val_accuracy"].append(np.array(val_accuracy.result()))

            logging.info(
                f"\n Epoch {epoch + 1} completed. Loss: {np.array(train_loss.result()):.3f}"
                f"Accuracy: {np.array(train_accuracy.result()):.3f}\n "
                f"Validation Loss: {np.array(val_loss.result()):.3f}\n "
                f"Validation Accuracy: {np.array(val_accuracy.result()):.3f} \n"
            )

            # Reset metrics at the end of each epoch
            train_loss.reset_state()
            train_accuracy.reset_state()
            val_loss.reset_state()
            val_accuracy.reset_state()

        logging.info("Adversarial training completed")

        visualizer = VisualizeTraining()
        visualizer.plot_adversarial_training_results(adv_training_history)
        self._plot_file_names.update(visualizer.plot_file_names)

    def save_model(self):
        """saving model weights"""

        try:
            user_input = input("Enter a path to save the model or press Enter to use the default path: ").strip()
            save_path = user_input if user_input else self._default_save_path()

        except EOFError as e:
            logging.error("train: error with input from user console, using default path: %s", e)
            save_path = self._default_save_path()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.inner.save_weights(save_path)
            logging.info("Model weights saved to: %s", save_path)

        except FileNotFoundError:
            print(StringStyling.box_style("File path not found or cannot open weight file"))
            sys.exit(1)
        except PermissionError:
            print(StringStyling.box_style("Missing writing permissions, cannot write weight file"))
            sys.exit(1)

    def _default_save_path(self) -> str:
        """Generate a default save path for the model based on training type"""
        base_path = "data/models"
        file_name = "adv_model.weights.h5" if self.args.adv else "model.weights.h5"
        return os.path.join(base_path, file_name)
