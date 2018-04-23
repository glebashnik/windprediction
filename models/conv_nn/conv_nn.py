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

    def build_model(self, history_length, num_features, l2_reg=0):
        
        history_input = Input(shape=(history_length, num_features), name='history_input')

        prod_conv = Conv1D(filters=128, kernel_size=2, activation='relu')(history_input)
        # prod_conv = BatchNormalization()(prod_conv)
        prod_pool = AveragePooling1D(pool_size=2)(prod_conv)
        prod_conv = Conv1D(filters=64, kernel_size=1, activation='relu')(prod_pool)
        # prod_conv = BatchNormalization()(prod_conv)
        prod_pool = AveragePooling1D(pool_size=1)(prod_conv)

        flat = Flatten()(prod_pool)
        x = Dense(64, activation='relu')(flat)
        # x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dense(8, activation='relu')(x)
        output = Dense(1)(x)

        self.model = Model(inputs=[history_input], outputs=[output])
        return self.model


    def train_network(self, x_train, y_train, opt='adam'):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=200)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        history = self.model.fit([x_train], [y_train],
                        batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopping, checkpoint],
                        epochs = self.epochs, verbose=2, shuffle=True)

        return history.history, self.model

    def evaluate(self, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae'])
        self.model.load_weights(self.model_path)
        
        evaluation = self.model.evaluate([x_test], y_test)
        return evaluation, self.model.metrics_names

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))