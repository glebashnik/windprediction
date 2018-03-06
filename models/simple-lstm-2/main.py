import os
from pandas import read_csv, DataFrame, concat
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
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2
from sklearn.decomposition import PCA

# Fix on windows bug from the Nov 2017 update
import win_unicode_console
win_unicode_console.enable()

class RNN:

    def __init__(self, datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0, sep=';')
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)

        self.dataset = self.dataset.dropna()

        self.testsplit = 0.8
        self.batch_size = 32
        self.epochs = 100

        data_x = self.dataset.iloc[:, :-1]
        data_y = self.dataset.iloc[:, -1:]

        #Normalizing data
        self.x_scaler = MinMaxScaler(copy=True, feature_range=(0,1))
        data_x = self.x_scaler.fit_transform(data_x)

        self.dataset.iloc[:, :-1] = data_x
        self.dataset.iloc[:, -1:] = data_y
        self.data = np.concatenate((data_x, data_y), axis=1)

        #Extract PCA features and reduce the dimensionality
        data_x = self.extract_PCA_fea
        tures(data_x,n_components = self.PCAcomp)


        # self.dataset.iloc[:,:-1] = data_x
        # self.dataset.iloc[:,-1:] = data_y
        # self.data = self.dataset.values

        self.data = np.concatenate((data_x, data_y), axis = 1)
        
        # Number of timesteps we want to look back and on
        # n_in = 6
        # self.data = np.concatenate((data_x, data_y), axis=1)
        
        # Number of timesteps we want to look back and on
        n_in = 4
        n_out = 1

        print(self.data[0,:])
        print()

        # Returns an (n_in * n_out) * num_vars NDFrame
        self.timeseries = self.series_to_supervised(data=self.data, 
                n_in=n_in, 
                n_out=n_out,
                dropnan=True)

        

        # Converts to numpy representation
        self.timeseries_np = self.timeseries.values

        # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
        self.timeseries_data = self.timeseries_np.reshape(self.timeseries_np.shape[0], n_in+n_out ,self.data.shape[1])

        # i = 0
        # while(True):
        #     print(self.timeseries_data[i,:,:])
        #     o=input()
        #     i += 1
        # Data is everything but the two last rows in the third dimension (which contain the delayed and actual production values)
        self.x_data = self.timeseries_data[:, :, :-1]
        self.y_data = self.timeseries_data[:, :, -1:]

        #Split for dividing the dataset in a factor of the batch size
        split = int(self.testsplit * self.x_data.shape[0])

        # Create training and test sets for x
        self.x_train = self.x_data[:split, :, :]
        self.x_test = self.x_data[split:, :, :]

        # Create training and test sets for y
        self.y_train = self.y_data[:split, :,:]
        self.y_test = self.y_data[split:, :, :]

        print('X_train shape: {}'.format(self.x_train.shape))
        print('y_train shape: {}'.format(self.y_train.shape))
        print('X_test shape: {}'.format(self.x_test.shape))
        print('X_test shape: {}'.format(self.y_test.shape))
     

        assert self.x_train.shape[0] % self.batch_size == 0, 'training sample size not divisible by batch size'
        assert self.x_test.shape[0] % self.batch_size == 0, 'testing sample size not divisible by batch size'

    def build_model(self):
        self.model = Sequential()

        self.model.add(LSTM(32, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))


        # BEST MODEL THUS FAR: (N_IN = 6, N_OUT = 1, split=0.9, patience=40)
        self.model.add(LSTM(32, return_sequences=True,
                                batch_input_shape=(self.batch_size, self.x_train.shape[1], self.x_train.shape[2]),
                                stateful=True))        
        
        self.model.add(LSTM(16, return_sequences=True))
        
        self.model.add(TimeDistributed(Dense(1)))
    
        self.model.summary()

    def train_network(self):
        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        early_stopping = EarlyStopping(monitor='val_loss', patience=15)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='loss', verbose=0, save_best_only=True, mode='min')

        # Creates the closest validation splitt divisible by batch size to 0.2
        samples_split = self.x_train.shape[0]
        val_split = ((0.2 * samples_split - ((0.2 * samples_split) % self.batch_size))/samples_split)


        self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, callbacks=[early_stopping, checkpoint], validation_split=val_split, epochs = self.epochs, verbose=1, shuffle=False)

    def predict(self):
        # Load best found model
        self.model.load_weights(os.path.join('checkpoint_model.h5'))

        self.predictions = self.model.predict(self.x_test, batch_size=self.batch_size)

        self.evaluation = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)

        print('Evaluating with test data')
        print(self.model.metrics_names)
        print(self.evaluation)
        print()

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def extract_PCA_features(self, data, n_components = 10):

        pca = PCA(n_components=n_components)
        data_x_pca = pca.fit_transform(data)

        self.pca_model = pca

        return data_x_pca


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
        lines = pyplot.plot(self.predictions, 'r', self.y_test, 'b')
        pyplot.setp(lines, linewidth=0.5)
        pyplot.show()

if __name__ == '__main__':
    datapath = os.path.join('..','..','data', 'data_bessaker_advanced.csv')

    nn_network = RNN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()
    nn_network.visualize()
