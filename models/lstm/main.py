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

from keras import optimizers

class LSTM:

    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = 300

        self.model_path = os.path.join('models','lstm','checkpoint.h5')

    # Builds a simple lstm network
    def build_model(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(16, return_sequences=True, input_shape=input_shape, activation='softsign'))
        self.model.add(LSTM(8, activation='softsign'))
        self.model.add(Dense(1))
        self.model.summary()

    # Builds an lstm model with given number of layers
    def build_model_general(self,input_shape, layers):

        self.model = Sequential()

        self.model.add(LSTM(layers[0], return_sequences=True, input_shape=input_shape, 
        activation='softsign'))

        depth = len(layers)

        if depth >= 3:

            for i in range(depth-2):
                self.model.add(LSTM(layers[i+1], return_sequences=True, activation='softsign'))

        self.model.add(LSTM(layers[-1], activation='softsign'))
        self.model.add(Dense(1))
        self.model.summary()

    def train_network(self, x_train, y_train, lr = 0.01, opt = 'Adam'):

        self.model.compile(loss='mae', optimizer=opt, metrics=['mae','mse',self.rmse])

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, callbacks=[early_stopping], validation_split=0.2, epochs = self.epochs, verbose=1, shuffle=False)

        return self.model

    def evaluate(self, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',
                           metrics=['mae', 'mse', self.rmse])
        self.model.load_weights(self.model_path)

        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation, self.model.metrics_names

    def predict(self, x_test):
        # Load best found model
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])
        self.model.load_weights(self.model_path)

        return self.model.predict(x_test, batch_size=self.batch_size)

    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))