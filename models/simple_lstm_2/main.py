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
from keras.regularizers import l2

class RNN:

    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(16, return_sequences=True, input_shape=input_shape, activation='softsign'))
        self.model.add(LSTM(8, activation='softsign'))
        self.model.add(Dense(1))
        self.model.summary()

    def train_network(self, x_train, y_train):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        early_stopping = EarlyStopping(monitor='val_loss', patience=30)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, callbacks=[early_stopping, checkpoint], validation_split=0.2, epochs = self.epochs, verbose=1, shuffle=False)

    def evaluate(self, model_path, x_test, y_test):
        # Load best found model
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])
        self.model.load_weights(model_path)

        evaluation = self.model.evaluate(x_test, y_test, batch_size=self.batch_size)

        return evaluation, self.model.metrics_names

    def predict(self, model_path, x_test):
        # Load best found model
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])
        self.model.load_weights(model_path)

        return self.model.predict(x_test, batch_size=self.batch_size)

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))