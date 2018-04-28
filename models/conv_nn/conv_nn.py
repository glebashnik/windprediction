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
from keras.regularizers import l2, l1

class Conv_NN:

    def __init__(self, model_path, batch_size=32, epochs=100):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path

    def build_model(self, production_length, forecast_length, num_features):
        
        production_input = Input(shape=(production_length, 1), name='history_input')

        prod_conv = Conv1D(filters=8, kernel_size=2, padding='same')(production_input)
        prod_conv = LeakyReLU(alpha=0.2)(prod_conv)
        prod_pool = MaxPooling1D(pool_size=4)(prod_conv)
        # prod_conv = Conv1D(filters=16, kernel_size=2, padding='same')(prod_pool)
        # prod_conv = LeakyReLU(alpha=0.2)(prod_conv)
        # prod_pool = MaxPooling1D(pool_size=2)(prod_conv)
        prod = Flatten()(prod_pool)
        prod = Dropout(0.2)(prod)

        forecast_input = Input(shape=(forecast_length, 2), name='forecast_input')

        forecast_conv = Conv1D(filters=8, kernel_size=2, padding='same')(forecast_input)
        forecast_conv = LeakyReLU(alpha=0.2)(forecast_conv)
        forecast_pool = MaxPooling1D(pool_size=4)(forecast_conv)
        # forecast_conv = Conv1D(filters=16, kernel_size=2, padding='same')(forecast_pool)
        # forecast_conv = LeakyReLU(alpha=0.2)(forecast_conv)
        # forecast_pool = MaxPooling1D(pool_size=2)(forecast_conv)
        forecast = Flatten()(forecast_pool)
        forecast = Dropout(0.2)(forecast)       

        single_input = Input(shape=(num_features,))
        total_input = concatenate([prod, forecast, single_input]) 
        x = Dense(64)(single_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.5)(x)
        x = Dense(32)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        total_input = concatenate([prod, forecast, x])

        x = Dense(16)(total_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)

        x = Dense(8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output = Dense(1)(x)

        self.model = Model(inputs=[production_input, forecast_input, single_input], outputs=[output])
        return self.model


    def train_network(self, x_train, y_train, opt='adam'):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        history = self.model.fit(x_train, y_train,
                        batch_size=self.batch_size, validation_split=0.1, callbacks=[checkpoint],
                        epochs = 300, verbose=2)

        return history.history, self.model

    def evaluate(self, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae'])
        self.model.load_weights(self.model_path)
        
        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation, self.model.metrics_names

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))