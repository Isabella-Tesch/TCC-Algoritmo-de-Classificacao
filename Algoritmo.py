import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

x_train_flattened = x_train.reshape(len(x_train), 28*28)
x_test_flattened = x_test.reshape(len(x_test), 28*28)

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flattened, y_train, epochs=5)

model.evaluate(x_test_flattened, y_test)

y_predicted = model.predict(x_test_flattened)
print(y_predicted[1])

print(np.argmax(y_predicted[1]))

y_predicted_labels = [np.argmax(i) for i in y_predicted]
print(y_predicted_labels[:5])

confusion_matrix = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
print(confusion_matrix)


