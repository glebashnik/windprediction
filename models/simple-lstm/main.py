from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2


class RNN:

    def __init__(self,datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0,delimiter=(';'))
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)

        self.dataset = self.dataset.dropna(how='any')

        # self.validsplit = 0.7
        self.testsplit = 0.8
        self.batch_size = 128
        self.epochs = 1000

        #-----FABIAN---------

        self.data = self.dataset.values

        #Normalizing data, input and output seperately
        data1, data2 = self.data[:,:-1],self.data[:,-1:]
        self.scaler1 = MinMaxScaler(copy=True,feature_range=(0,1))
        self.scaler2 = MinMaxScaler(copy=True, feature_range=(0,1))
        data1 = self.scaler1.fit_transform(data1)
        data2 = self.scaler2.fit_transform(data2)
        self.data = np.concatenate((data1,data2),axis=1)

        #----EIRIK----
        # data_x = self.dataset.iloc[:,:-1]
        # data_y = self.dataset.iloc[:,-1:]
        #
        # #Normalizing data - only the input
        # self.x_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        # self.x_scaler.fit(data_x)
        # data_x = self.x_scaler.transform(data_x)
        #
        # #Normalizing output with separate scaler in order to facilitate later inverse scaling
        # self.y_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        # self.y_scaler.fit(data_y)
        # data_y = self.y_scaler.transform(data_y)
        #
        #
        # self.dataset.iloc[:,:-1] = data_x
        # self.dataset.iloc[:,-1:] = data_y
        # self.data = self.dataset.values
        #----------------------------
        
        # Number of timesteps we want to look back and on
        n_in = 10
        n_out = 1

        # Returns an (n_in * n_out) * num_vars NDFrame
        self.timeseries = self.series_to_supervised(data=self.data, n_in=n_in, n_out=n_out, dropnan=True)

        # Converts to numpy representation
        self.timeseries_np = self.timeseries.values

        # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
        self.timeseries_data = self.timeseries_np.reshape(self.timeseries_np.shape[0], n_in+n_out ,self.data.shape[1])

        # Data is everything but the two last rows in the third dimension (which contain the delayed and actual production values)
        self.x_data, self.y_data = self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size, :,:-2],self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size,:,-1:]

        #Split for dividing the dataset in a factor of the batch size
        split = self.testsplit * self.x_data.shape[0]
        split -= split % self.batch_size
        split = int(split)

        # Create training, validation and test sets for x
        self.x_train = self.x_data[:split, :, :]
        # self.x_valid = self.x_data[int(self.validsplit * self.x_data.shape[0]):int(self.testsplit * self.x_data.shape[0]), : ,:]
        self.x_test = self.x_data[split:, :, :]
        # Create training, validation and test sets for y
        self.y_train = self.y_data[:split, :,:]
        # self.y_valid = self.y_data[int(self.validsplit * self.y_data.shape[0]):int(self.testsplit * self.y_data.shape[0]), :, :]
        self.y_test = self.y_data[split:, :, :]

        # self.y_train = self.y_data[0:int(self.validsplit * self.y_data.shape[0]), -1:]
        # self.y_valid = self.y_data[int(self.validsplit * self.y_data.shape[0]):int(self.testsplit * self.y_data.shape[0]), -1:]
        # self.y_test = self.y_data[int(self.testsplit * self.y_data.shape[0]):, -1:]

        print(self.x_train.shape)
        print(self.y_test.shape)
        exit(0)

    def build_model(self):

        self.alpha = 0.1
        self.model = Sequential()

        self.model.add(LSTM(11, batch_input_shape=(self.batch_size, self.x_train.shape[1], self.x_train.shape[2]), return_sequences=True))
        # self.model.add(LeakyReLU(0.1))
        # self.model.add(Dropout(0.2))
        self.model.add(LSTM(11, return_sequences=True))
        # self.model.add(Dropout(0.2))
        self.model.add(LSTM(11, return_sequences=True))

        # self.model.add(LSTM(11, return_sequences=True))

        self.model.add(TimeDistributed(Dense(1)))


        # self.model.add(LSTM(128, input_shape=(self.x_data.shape[1], self.x_data.shape[2]), return_sequences=True))
        # self.model.add(Dropout(0.15))
        # #self.model.add(LSTM(64, return_sequences=True))
        # #self.model.add(LSTM(32, return_sequences=True))
        # self.model.add(LSTM(1))

    
        self.model.summary()
        print('Model created')

    def train_network(self):
        self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)
        self.model.compile(loss='mae', optimizer='adadelta',
                           metrics=['mae','mse',self.rmse])
        print("compiled")

        # Perform early stop if there was not improvement for n epochs
        early_stopping = EarlyStopping(monitor='loss', patience=30)

        # Save the best model each time
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')


        #Train the model
        # for i in range(self.epochs):
        #
        #     self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs = 1, verbose=2,
        #                     callbacks=[checkpoint, early_stopping], validation_split=0,shuffle=False)
        #     #Resetting states
        #     self.model.reset_states()
        #     print('Epoch: %.d' % i)


        # self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
        #                callbacks=[checkpoint, early_stopping], validation_split=0, shuffle=False)

        # Train the model
        log = self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs = self.epochs, verbose=2,
                            callbacks=[checkpoint, early_stopping], validation_split=0.2,shuffle=False)


        #print(log.history)

        # with open(os.path.join('results.pickle'), 'wb') as f:
        #     pickle.dump(log.history, f)

        # self.model.save(os.path.join('model.h5'))

        print('--Model trained and saved--')

    def predict(self):
        self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

        #Look only on the last prediction of the time series
        self.y_test_last = self.y_test[:,-1,:]
        self.predictions_last = self.predictions[:,-1,:]

        #Inverse scaling back to original scaling
        self.y_test_last = self.scaler2.inverse_transform(self.y_test_last)
        self.predictions_last = self.scaler2.inverse_transform(self.predictions_last)


        self.evaluation = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)


        # print('Evaluating with test data, normalized')
        # print(self.model.metrics_names)
        # print(self.evaluation)
        # print()

        print('Prediction --vs-- label')
        # print(np.concatenate((self.predictions[0:10], np.reshape(self.y_test[0:10], (10,1))),axis=1))
        print(np.concatenate((self.predictions_last[0:20], self.y_test_last[0:20]),axis=1))
        print()

        print('RMSE ')
        print(np.mean(self.rmse_numpy(self.y_test_last,self.predictions_last)))
        print('MAE')
        print(mean_absolute_error(self.y_test_last,self.predictions_last))


    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def mae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred-y_true))

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
        inverse_predictions = self.y_scaler.inverse_transform(self.predictions)
        inverse_actuals = self.y_scaler.inverse_transform(self.y_test)

        pyplot.plot(inverse_predictions, 'r--', inverse_actuals, 'b--')
        pyplot.show()

if __name__ == '__main__':
    datapath = os.path.join('..','..','data', 'data-2.3.csv')
    # datapath = os.path.join('data', 'cleaned_data.csv')

    nn_network = RNN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()
    nn_network.visualize()