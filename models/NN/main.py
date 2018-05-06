from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import Input, Dense, LeakyReLU, BatchNormalization, Average, Dropout, Concatenate
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

# Generator that returns the data shuffled
def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2


class NN:

    def __init__(self, batch_size=32, epochs=1000, dropoutrate=0.3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropoutrate
        self.relu_leak = 0.2

        self.model_path = os.path.join('models','NN_forest','checkpoint.h5')

    # Build n networks and averages the final dense output
    def build_forest_model(self, input_dim, num_windmills, model_structure):
        input_layer = Input(shape=(input_dim,))

        for i, network in enumerate(model_structure):
            network_list.append(self.dense_block(
                input_data=input_layer, units=network[0][0], dropout=network[0][1]))
            for layer in network[1:]:

                if layer[0] != 0:
                    network_list[i] = self.dense_block(
                        input_data=network_list[i], units=layer[0], dropout=layer[1])

            network_list[i] = Dense(1)(network_list[i])

        out_avg = Average()(network_list)

        self.model = Model(inputs=input_layer, outputs=out_avg)
        return self.model

    # builds a single network
    def build_model(self, input_dim, output_dim):
        input_layer = Input(shape=(input_dim,))

        # x1 = self.dense_block(input_layer, 128, False, 0)
        x1 = self.dense_block(input_layer, 64, False, 0)
        x2 = self.dense_block(x1, 32, False, 0)
        x3 = self.dense_block(x2, 16, False, 0)
        x4 = self.dense_block(x3, 8, False, 0)
        x5 = self.dense_block(x4, 2, False, 0)

        total = Dense(1)(x5)

        self.model = Model(inputs=input_layer, outputs=total)
        self.model.summary()
        return self.model

    # Layers are build with block consisting of Dense, dropout, batchNorm and leakyrelu
    def dense_block(self, input_data, units, dropout=False, l2_reg=0):

        x = Dense(units=units, activation=None,
                  activity_regularizer=l2(l2_reg))(input_data)
        if dropout:
            x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)
        return LeakyReLU(self.relu_leak)(x)


    def train_network(self, x_train, y_train, opt='adam', validation_split=0.15):
        self.model.compile(loss='mae', optimizer=opt,
                           metrics=['mae'])
        
        # Model is stopped if no improvement for 400 epochs. In addition, only 
        # the best model is saved and later used for evaluation
        early_stopping = EarlyStopping(monitor='val_loss', patience=400)
        checkpoint = ModelCheckpoint(
            self.model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        self.model.fit(x=x_train, y=[y_train, y_train_vector], batch_size=self.batch_size, validation_split=validation_split, callbacks=[
                       early_stopping, checkpoint], epochs=self.epochs, verbose=2, shuffle=True)
        return history.history, self.model

    def predict(self, x):
        self.model.compile(loss='mae', optimizer='adam',
                           metrics=['mae'])
        self.model.load_weights(self.model_path)

        return self.model.predict(x)

    def evaluate(self, x_test, y_test, single_targets=False):
        self.model.compile(loss='mae', optimizer='adam',
                           metrics=['mae', 'mse', self.rmse])
        self.model.load_weights(model_path)

        evaluation = self.model.evaluate(x_test, y_test)
        return evaluation, self.model.metrics_names

    def visualize_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model_architecture.png')

    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
