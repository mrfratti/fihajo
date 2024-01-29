import tensorflow as tf
import pandas as pd
import numpy as np

# LOAD
print("LOAD...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# MERGE 
print("MERGE...")
x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# RESHAPE
print("RESHAPE...")
images_number = x_all.shape[0]
images_reshaped = x_all.reshape(images_number, -1)
data = pd.DataFrame(images_reshaped)
data['solution'] = y_all

# SAVE
print("SAVE...")
data.to_csv('mnist.csv', index=False)

print("DONE!")