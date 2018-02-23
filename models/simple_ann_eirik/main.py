from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

class NN:

    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1))
        self.model.summary()

    def train_network(self, x_train, y_train):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae','mse',self.rmse])

        early_stopping = EarlyStopping(monitor='val_loss', patience=500)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopping, checkpoint], epochs = self.epochs, verbose=2, shuffle=True)

    def evaluate(self, model_path, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae','mse',self.rmse])
        self.model.load_weights(model_path)
        
        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation, self.model.metrics_names

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))