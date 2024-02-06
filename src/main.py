import argparse
import os.path
import datetime
import platform

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import uncertainty_wizard as uwiz
import matplotlib.pyplot as plt
from uncertainty_wizard.models.stochastic_utils.layers import UwizBernoulliDropout, UwizGaussianDropout, UwizGaussianNoise
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


class LoadData:
    @staticmethod
    def load_and_process_mnist():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train.astype('float32') / 255).reshape(x_train.shape[0], 28, 28, 1)
        x_test = (x_test.astype('float32') / 255).reshape(x_test.shape[0], 28, 28, 1)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        return (x_train, y_train), (x_test, y_test)

class BuildModel:
    @staticmethod
    def create_mnist_model(stochastic_mode):
        # Creating a stochastic model using uncertainty_wizard
        model = uwiz.models.StochasticSequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(tf.keras.layers.Dropout(0.5))
        model.add(UwizBernoulliDropout(0.5, stochastic_mode=stochastic_mode))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        if platform.system() == "Darwin" and platform.processor() == "arm":
            opt = tf.keras.optimizers.legacy.Adadelta()
        else:
            opt = tf.keras.optimizers.Adadelta()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])

        model.summary()

        return model


class Trainer:
    def __init__(self, model):
        self.model = model

    def train_model(self, x_train, y_train, epochs):
        history = self.model.fit(x_train,
                                 y_train,
                                 validation_split=0.1,
                                 batch_size=32,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
        return history

    @staticmethod
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

        # Generate a unique filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"val_acc_and_loss_{timestamp}.png"

        # Save the plot
        plot_dir = './data/plots/train'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename))

        plt.show()


class Evaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, x_test, y_test):
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=2)
        print('\nTest accuracy:', test_acc)
        return test_loss, test_acc

    def plot_confusion_matrix(self, y_test, x_test):
        y_pred = self.model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"confusion_matrix_{timestamp}.png"

        # Save the plot
        plot_dir = './data/plots/evaluate'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename))
        plt.show()

    def plot_predictions(self, x_test, num_samples=25):
        predictions = self.model.predict(x_test[:num_samples])
        predicted_labels = np.argmax(predictions, axis=1)

        plt.figure(figsize=(10, 10))
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


class UncertaintyAnalyzer:
    def __init__(self, model):
        self.model = model

    def analyze_uncertainty(self, x_test, sample_size=32):
        quantifiers = ['pcs', 'mean_softmax']
        results = self.model.predict_quantified(x_test,
                                                quantifier=quantifiers,
                                                batch_size=64,
                                                sample_size=sample_size,
                                                verbose=1)
        return results

    @staticmethod
    def plot_uncertainty(results):
        uncertainty_scores = results[1][1] # ??? adjust per result structure ??
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

    @staticmethod
    def plot_pcs_and_mean_softmax(results, x_test, num_samples=25):
        pcs_scores, mean_softmax_scores = results[0][1], results[1][1]
        for i in range(num_samples):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
            plt.title(f'PCS: {pcs_scores[i]:.2f}\nMS: {mean_softmax_scores[i]:.2f}')
            plt.axis('off')
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"pcs_and_mean_softmax_{timestamp}.png"

        plot_dir = './data/plots/evaluate'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename))
        plt.show()

    @staticmethod
    def plot_pcs_softmax_bars(results):
        pcs_scores, mean_softmax_scores = results[0][1], results[1][1]
        plt.figure(figsize=(10, 4))
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
        filename = f"pcs_mean_softmax_bars_{timestamp}.png"

        plot_dir = './data/plots/analyze'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, filename))
        plt.show()


class AIUncertaintyTool:
    def __init__(self, epochs=5):
        self.epochs = epochs
        self.model = None

    def run_train(self):
        # Load and preprocess data
        (x_train, y_train), _ = LoadData.load_and_process_mnist()

        # Create and train model
        stochastic_mode = StochasticMode()
        self.model = BuildModel.create_mnist_model(stochastic_mode)
        trainer = Trainer(self.model)
        history = trainer.train_model(x_train, y_train, self.epochs)
        Trainer.plot_training_results(history)
        self.model.inner.save_weights('data/mnist_model_stochastic.h5')

    def run_evaluate(self):
        #if self.model is None:
        #    print("Model not found. Please train the model first")
        #    return

        # Load test data
        _, (x_test, y_test) = LoadData.load_and_process_mnist()

        # Evaluate model
        self.model = BuildModel.create_mnist_model(StochasticMode())
        self.model.inner.load_weights('data/mnist_model_stochastic.h5')
        evaluator = Evaluator(self.model)
        evaluator.evaluate(x_test, y_test)
        evaluator.plot_confusion_matrix(y_test, x_test)
        evaluator.plot_predictions(x_test, 25)

    def run_analyze(self):
        # Load test data
        _, (x_test, _) = LoadData.load_and_process_mnist()

        # Analyze uncertainty
        self.model = BuildModel.create_mnist_model(StochasticMode())
        self.model.inner.load_weights('data/mnist_model_stochastic.h5')
        analyzer = UncertaintyAnalyzer(self.model)
        results = analyzer.analyze_uncertainty(x_test)
        analyzer.plot_uncertainty(results)
        analyzer.plot_pcs_and_mean_softmax(results, x_test, 25)
        analyzer.plot_pcs_softmax_bars(results)


def main():
    parser = argparse.ArgumentParser(description="AI Model Uncertainty Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand for training
    parser_train = subparsers.add_parser('train', help='Train the model')
    parser_train.add_argument('--epochs', type=int, default=5, help='Number of epochs for training the model')
    parser_train.set_defaults(func=lambda args: train(args.epochs))

    # Subcommand for evaluation
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate the model')
    parser_evaluate.set_defaults(func=evaluate)

    # Subcommand for uncertainty analysis
    parser_analyze = subparsers.add_parser('analyze', help='Analyze model uncertainty')
    parser_analyze.set_defaults(func=analyze)

    args = parser.parse_args()

    if args.command == 'train':
        AIUncertaintyTool(epochs=args.epochs).run_train()
    elif args.command == 'evaluate':
        AIUncertaintyTool().run_evaluate()
    elif args.command == 'analyze':
        AIUncertaintyTool().run_analyze()


def train(epochs):
    tool = AIUncertaintyTool(epochs)
    tool.run_train()


def evaluate():
    tool = AIUncertaintyTool()
    tool.run_evaluate()


def analyze():
    tool = AIUncertaintyTool()
    tool.run_analyze()


if __name__ == "__main__":
    main()
