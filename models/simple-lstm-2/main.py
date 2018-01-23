from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

class RNN:

    def __init__(self,datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0, sep=';')
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)

        self.dataset = self.dataset.dropna()

        self.testsplit = 0.9
        self.batch_size = 32
        self.epochs = 1000

        #print(self.dataset)

        data_x = self.dataset.iloc[:,:-1]
        data_y = self.dataset.iloc[:,-1:]

        #Normalizing inputs
        self.x_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        data_x = self.x_scaler.fit_transform(data_x)

        #Normalizing inputs
        # self.y_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        # self.y_scaler.fit(data_y)
        # data_y = self.y_scaler.transform(data_y)

        self.dataset.iloc[:,:-1] = data_x
        self.dataset.iloc[:,-1:] = data_y
        self.data = np.concatenate((data_x, data_y), axis=1)
        
        # Number of timesteps we want to look back and on
        n_in = 6
        n_out = 0

        # Returns an (n_in * n_out) * num_vars NDFrame
        self.timeseries = self.series_to_supervised(data=self.data, n_in=n_in, n_out=n_out, dropnan=True)

        # Converts to numpy representation
        self.timeseries_np = self.timeseries.values

        # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
        self.timeseries_data = self.timeseries_np.reshape(self.timeseries_np.shape[0], n_in+n_out ,self.data.shape[1])

        # Data is everything but the two last rows in the third dimension (which contain the delayed and actual production values)
        self.x_data = self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size, :,:-2]
        self.y_data = self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size,:,-1:]

        #Split for dividing the dataset in a factor of the batch size
        split = self.testsplit * self.x_data.shape[0]
        split -= split % self.batch_size
        split = int(split)

        # Create training and test sets for x
        self.x_train = self.x_data[:split, :, :]
        self.x_test = self.x_data[split:, :, :]

        # Create training and test sets for y
        self.y_train = self.y_data[:split, :,:]
        self.y_test = self.y_data[split:, :, :]

        assert self.x_train.shape[0] % self.batch_size == 0, 'training sample size not divisible by batch size'
        assert self.x_test.shape[0] % self.batch_size == 0, 'testing sample size not divisible by batch size'

    def build_model(self):
        self.model = Sequential()

        # Input layer
        # self.model.add(LSTM(64, activity_regularizer=regularizers.l2(0.01), 
        #                         return_sequences=True, 
        #                         input_shape=(self.x_data.shape[1], self.x_data.shape[2])))
        # #self.model.add(Dropout(0.1)
        # self.model.add(LSTM(32, activation='relu'))
        # self.model.add(TimeDistributed(Dense(1)))


        # BEST MODEL THUS FAR: (N_IN = 6, N_OUT = 1, split=0.9, patience=40)
        self.model.add(LSTM(32, return_sequences=True,
                                batch_input_shape=(self.batch_size, self.x_train.shape[1], self.x_train.shape[2]),
                                stateful=True))        
        self.model.add(LSTM(32, return_sequences=True))
        
        self.model.add(TimeDistributed(Dense(1)))
    
        self.model.summary()

    def train_network(self):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        # Perform early stop if there was not improvement for n epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=50)

        # Save the best model each time
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        for i in range(self.epochs):
            self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs = 1, verbose=2,
                            callbacks=[checkpoint, early_stopping], validation_split=0.2, shuffle=False)
            #Resetting states
            self.model.reset_states()
            print('Epoch: %.d' % i)

    def predict(self):
        # Load the best model found during training
        self.model.load_weights('checkpoint_model.h5')

        # Predict and evaluate
        self.predictions = self.model.predict(self.x_test)
        self.evaluation = self.model.evaluate(self.x_test, self.y_test)

        print('Evaluating with test data')
        print(self.model.metrics_names)
        #print(self.y_scaler.inverse_transform(self.evaluation[1]))
        print(self.evaluation)
        print()

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    # convert series to supervised learning, time series data generation
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def visualize(self):
        #inverse_predictions = self.y_scaler.inverse_transform(self.predictions)
        #inverse_actuals = self.y_scaler.inverse_transform(self.y_test)
        inverse_predictions = self.predictions
        inverse_actuals = self.y_test

        pyplot.plot(inverse_predictions, 'r--', inverse_actuals, 'b--')
        pyplot.show()

if __name__ == '__main__':
    datapath = os.path.join('data', 'advanced_data2.csv')

    nn_network = RNN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()
    nn_network.visualize()
    
    exit(0)