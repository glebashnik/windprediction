import os
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

class RNN:

    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(32, return_sequences=True,
                                batch_input_shape=(self.batch_size, input_shape[0], input_shape[1]),
                                stateful=True))
        self.model.add(LSTM(16))
        self.model.add(Dense(1))
        self.model.summary()

    def train_network(self, x_train, y_train):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        # Self-written early_stopping
        patience = 10
        epochs_since_best = 0
        best_val_loss = float('inf')

        for i in range(self.epochs):
            log = self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, validation_split=0.2, epochs = 1, verbose=1, shuffle=False)
            
            # Early stopping
            epochs_since_best += 1
            if log.history['val_loss'][0] < best_val_loss:
                self.model.save(os.path.join('checkpoint_model.h5'))
                print("Improved model, saving...")
                epochs_since_best = 0
                best_val_loss = log.history['val_loss']
            elif epochs_since_best >= patience:
                print("Exceeded patience, halting training...")
                break
                            
            #Resetting states
            self.model.reset_states()
            print('Epoch: %.d' % (i + 1))

    def evaluate(self, model_path, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])
        self.model.load_weights(model_path)

        evaluation = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)

        return evaluation, self.model.metrics_names

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
