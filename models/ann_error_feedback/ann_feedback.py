from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np
import time

from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Average, Dropout, Concatenate
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2
from keras.utils import generic_utils as keras_generic_utils

def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2


class NN_feedback:

    def __init__(self, batch_size=32, epochs=1000, dropoutrate=0.3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropoutrate
        self.relu_leak = 0.2

    # Build n networks and averages the final dense output
    def build_feedback_model(self, input_dim, model_structure):
        input_layer = Input(shape=(input_dim,))

        error = Input(shape=(1,))

        x = self.dense_block(input_layer, 64, False, 0)
        x = self.dense_block(x, 32, False, 0)
        x = self.dense_block(x, 16, False, 0)
        x = self.dense_block(x, 8, False, 0)

        x = Concatenate()([x, error])

        x = self.dense_block(x, 2, False, 0)
        # for layer in model_structure[1:]:

        #     if layer[0] != 0:
        #         x = self.dense_block(input_data = x, units = layer[0], dropout = layer[1])

        out = Dense(1)(x)

        self.model = Model(inputs=[input_layer, error], outputs=out)
        # self.model.summary()

    def build_simple_model(self, input_dim, model_structure):
        input_layer = Input(shape=(input_dim,))

        # x = self.dense_block(input_layer, 64, False, 0)
        x = self.dense_block(input_layer, 32, False, 0)
        x = self.dense_block(x, 16, False, 0)
        x = self.dense_block(x, 8, False, 0)
        x = self.dense_block(x, 2, False, 0)

        out = Dense(1)(x)

        self.model = Model(inputs=input_layer, outputs=out)
        self.model.summary()

    def dense_block(self, input_data, units, dropout=False, l2_reg=0):

        x = Dense(units=units, activation=None,
                  activity_regularizer=l2(l2_reg))(input_data)
        if dropout:
            x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)
        return LeakyReLU(self.relu_leak)(x)

    def train_network(self, x_train, y_train, opt='adam'):
        self.model.compile(loss='mae', optimizer=opt,
                           metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=500)
        checkpoint = ModelCheckpoint(
            'checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        num_samples = x_train.shape[0]

        print('Batch size: {}'.format(self.batch_size))

        loss_history = []

        for epoch in range(self.epochs):

            #Initialization for beginning of epoch
            pred_error = np.zeros(self.batch_size)
            # progbar = keras_generic_utils.Progbar(num_samples)
            start = time.time()
            

            batch_losses = []

            for batch_i in range(0, num_samples, self.batch_size):

                # Generator that returns random batch, iterates indefinitely with batch size as step size
                x_batch, y_batch = next(
                    gen_batch(x_train, y_train, self.batch_size))

                # Training model on current batch
                # loss = self.model.train_on_batch([x_batch, pred_error], y_batch)
                loss = self.model.train_on_batch(x_batch, y_batch)
                batch_losses.append(loss)
                # Generating predictions for current batch
                batch_pred = self.model.predict(x_batch)

                # Prediction error on current batch for feedback
                pred_error = y_batch - batch_pred

                #Update progressionbar (doesnt work on Linux..??)
                # progbar.add(self.batch_size, values=[("loss", loss)])
                

            #Calculate avg loss for epoch
            avg_epoch_loss = np.average(np.asarray(batch_losses).shape,axis=0)
            loss_history.append(avg_epoch_loss)
            
            print('')
            print('Epoch {}/{}, Time: {}, loss: {}\n'.format(epoch + 1, self.epochs, int(time.time() - start), avg_epoch_loss))


    def evaluate(self, model_path, x_test, y_test):
        self.model.compile(loss='mae', optimizer='adam',
                           metrics=['mae'])
        self.model.load_weights(model_path)

        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation, self.model.metrics_names

    def visualize_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model_architecture.png')

    # RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
