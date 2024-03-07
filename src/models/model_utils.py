import platform
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import uncertainty_wizard as uwiz
from uncertainty_wizard.models.stochastic_utils.layers import UwizBernoulliDropout, UwizGaussianDropout, \
    UwizGaussianNoise


def load_and_preprocess_mnist():
    # Load mnist data and reshape for CNN
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32') / 255).reshape(x_train.shape[0], 28, 28, 1)
    x_test = (x_test.astype('float32') / 255).reshape(x_test.shape[0], 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)


def create_mnist_model(stochastic_mode):
    # Creating a stochastic model using uncertainty_wizard
    model = uwiz.models.StochasticSequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        UwizBernoulliDropout(0.5, stochastic_mode=stochastic_mode),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        opt = tf.keras.optimizers.legacy.Adadelta()
    else:
        opt = tf.keras.optimizers.Adadelta()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model