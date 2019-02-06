import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from numpy.random import seed
from tensorflow import set_random_seed
import random as rn
import tensorflow as tf
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

seed(1)

set_random_seed(2)

rn.seed(12345)

tf.set_random_seed(1234)

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

neural_model = Sequential([
    Dense(2, input_shape=(13,), activation="relu"),
    Dense(1, activation="linear")
])

neural_model.compile(SGD(lr = .003), "mean_squared_error", \
                     metrics=["accuracy"])

np.random.seed(0)
run_hist_1 = neural_model.fit(x_train, y_train, epochs=4000,\
                              validation_data=(x_test, y_test), \
                              verbose=True, shuffle=False)

neural_network_dropouts = Sequential()
neural_network_dropouts.add(Dense(2, activation='relu', input_shape=(13,)))
neural_network_dropouts.add(Dropout(0.1))
neural_network_dropouts.add(Dense(1, activation='linear'))

neural_network_dropouts.compile(SGD(lr = .003), "mean_squared_error", \
                     metrics=["accuracy"])

np.random.seed(0)
run_hist_2 = neural_network_dropouts.fit(x_train, y_train, epochs=4000,\
                              validation_data=(x_test, y_test), \
                              verbose=True, shuffle=False)

print("Mean squared error of a neural model: %.2f" %
      mean_squared_error(y_test, neural_model.predict(x_test)))

print('Variance score: %.2f' % r2_score(y_test, neural_model.predict(x_test)))

plt.plot(run_hist_1.history["loss"],'r', marker='.', label="Train Loss")
plt.plot(run_hist_1.history["val_loss"],'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error")
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('Error')
plt.grid()
plt.show()


print("Mean squared error of a neural model with dropouts: %.2f" %
      mean_squared_error(y_test, neural_network_dropouts.predict(x_test)))

print('Variance score of neural model with dropouts: %.2f' % r2_score(y_test, neural_network_dropouts.predict(x_test)))

plt.plot(run_hist_2.history["loss"],'r', marker='.', label="Train Loss")
plt.plot(run_hist_2.history["val_loss"],'b', marker='.', label="Validation Loss")
plt.title("Train loss and validation error with dropouts")
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('Error')
plt.grid()
plt.show()