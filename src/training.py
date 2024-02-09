import logging
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, LambdaCallback
from model_utils import load_and_preprocess_mnist, create_mnist_model
from visualization import plot_training_results
from uncertainty_wizard.models._stochastic._stochastic_mode import StochasticMode


def train_model(args):
    (x_train, y_train), _ = load_and_preprocess_mnist()

    # Recreate model architecture
    stochastic_mode = StochasticMode()
    model = create_mnist_model(stochastic_mode)

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

    print("Training completed. Saving model weights as 'model_weights.h5")
    model.inner.save_weights('data/model_weights.h5')
    print('\n')
    print("Model weights saved")
    plot_training_results(history)
