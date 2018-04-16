from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

class Conv_NN:

    def __init__(self, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs

    def build_model(self, history_length, rest_input_dim, l2_reg=0):
        
        production_input = Input(shape=(history_length, 1), name='production_input')

        prod_conv = Conv1D(filters=5, kernel_size=5, activation='relu', activity_regularizer = l2(l2_reg))(production_input)
        prod_pool = MaxPool1D(pool_size=2)(prod_conv)
        flat = Flatten()(prod_pool)

        rest_input = Input(shape=(rest_input_dim,), name='rest_input')

        total_input = concatenate([flat, rest_input])
        x = Dense(64, activation='relu', activity_regularizer = l2(l2_reg))(total_input)
        x = Dense(32, activation='relu', activity_regularizer = l2(l2_reg))(x)
        x = Dense(16, activation='relu', activity_regularizer = l2(l2_reg))(x)
        output = Dense(1)(x)

        self.model = Model(inputs=[production_input, rest_input], outputs=[output])
        return self.model


    def train_network(self, x_prod_train, x_rest_train, y_train, opt='adam'):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=500)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        self.model.fit([x_prod_train, x_rest_train], [y_train],
                        batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopping, checkpoint],
                        epochs = self.epochs, verbose=2, shuffle=True)

    def evaluate(self, model_path, x_prod_test, x_rest_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae','mse',self.rmse])
        self.model.load_weights(model_path)
        
        evaluation = self.model.evaluate([x_prod_test, x_rest_test], y_test)
        return evaluation, self.model.metrics_names

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))