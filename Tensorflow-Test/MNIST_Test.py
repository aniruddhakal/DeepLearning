import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import sys

"""
Example created as shown in:
/
Original Full example video:
https://www.youtube.com/watch?v=wQ8BIBpya2k&t=925s
"""

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy')

N_EPOCHS = 1

if len(sys.argv) > 1:
    N_EPOCHS = int(sys.argv[1])

print('Running for %d epochs' % N_EPOCHS)

model.fit(x_train, y_train, epochs=N_EPOCHS)

predictions = np.argmax(model.predict(x_test, verbose=True), axis=1)
accuracy = accuracy_score(y_test, predictions)
#
print('Classification Accuracy: %.2f%%' % ((accuracy) * 100))
print("Finished...")
