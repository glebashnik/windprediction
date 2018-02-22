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

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


class NETWORK:

    def __init__(self, datapath, pca = False):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0, sep=';')
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)

        # self.dataset = self.dataset.dropna()

        self.testsplit = 0.9
        self.validation_part = 0.10
        self.batch_size = 32
        self.epochs = 200
        self.PCAcomp = 40

        self.dataset = self.dataset.dropna(how='any')
        data_x = self.dataset.iloc[:, :-1]
        data_y = self.dataset.iloc[:, -1:]

        if pca:
        #Extract PCA features and reduce the dimensionality
            data_x = self.extract_PCA_features(data_x,n_components = self.PCAcomp)
            data_x = data_x.reshape(data_x.shape[0], 1 ,data_x.shape[1])

        
        else:
            data_x = data_x.values.reshape(data_x.shape[0], 1 ,data_x.shape[1])

        # self.data = np.concatenate((data_x, data_y), axis = 1)
        # print('Data shape after PCA feature construction: {}'.format(self.data.shape))

        # self.dataset.iloc[:, :-1] = data_x
        # self.dataset.iloc[:, -1:] = data_y
        # self.data = np.concatenate((data_x, data_y), axis=1)

        # Number of timesteps we want to look back and on
        # n_in = 4
        # n_out = 0

        # Returns an (n_in * n_out) * num_vars NDFrame
        # self.timeseries = self.series_to_supervised(data=self.data, 
                # n_in=n_in, 
                # n_out=n_out,
                # dropnan=True)

        # Converts to numpy representation
        # self.timeseries_np = self.timeseries.values

        # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
        # self.timeseries_data = self.timeseries_np.reshape(self.timeseries_np.shape[0], n_in+n_out ,self.data.shape[1])

        # Data is everything but the two last rows in the third dimension (which contain the delayed and actual production values)
        # self.x_data = self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size, :,:-2]
        # self.y_data = self.timeseries_data[:self.timeseries_data.shape[0]-self.timeseries_data.shape[0] % self.batch_size,:,-1:]

        #Split for dividing the dataset in a factor of the batch size
        split = self.testsplit * data_x.shape[0]
        split -= split % self.batch_size
        split = int(split)

        # # Create training and test sets for x
        self.x_train = data_x[:split, :, :]
        self.x_test = data_x[split:, :, :]


        # Create training and test sets for y
        self.y_train = data_y.values[:split, -1]
        self.y_test = data_y.values[split:, -1]


        # assert self.x_train.shape[0] % self.batch_size == 0, 'training sample size not divisible by batch size'
        # assert self.x_test.shape[0] % self.batch_size == 0, 'testing sample size not divisible by batch size'

    def build_lstm(self):
        self.model = Sequential()

        self.model.add(LSTM(32, return_sequences=True,
                                #activation='softsign',
                                batch_input_shape=(self.batch_size, 1, self.x_train.shape[2]),#, self.x_train.shape[2]),
                                stateful=True))
        
        self.model.add(LSTM(16))
        
        self.model.add(Dense(1))
    
        self.model.summary()

    def build_dense(self):

        self.model = Sequential()

        self.model.add(Flatten(input_shape=(1,self.x_train.shape[2])))

        # self.model.add(Dense(128, activation = 'relu'))

        self.model.add(Dense(64, activation = 'relu'))

        # self.model.add(Dropout(0.2))

        # self.model.add(Dense(32, activation = 'relu'))

        # self.model.add(Dropout(0.2))

        self.model.add(Dense(32, activation = 'relu'))


        self.model.add(Dense(16, activation = 'relu'))


        self.model.add(Dense(8, activation = 'relu'))

        self.model.add(Dense(2, activation = 'relu'))

        # self.model.add(Dropout(0.2))

        self.model.add(Dense(1))

        self.model.summary()

    def train_lstm(self):

        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        # Self-written early_stopping
        patience = 20
        epochs_since_best = 0
        best_val_loss = float('inf')

        # Creates the closest validation splitt divisible by batch size to 0.2
        samples_split = self.x_train.shape[0]
        val_split = ((self.validation_part * samples_split - ((self.validation_part * samples_split) % self.batch_size))/samples_split)

        for i in range(self.epochs):
            log = self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size,
             validation_split=val_split, epochs = 1, verbose=2, shuffle=False)
            
            # Early stopping
            epochs_since_best += 1
            if log.history['val_loss'][0] < best_val_loss:
                self.model.save(os.path.join('checkpoint_model.h5'))
                print("Improved model, saving...")
                epochs_since_best = 0
                best_val_loss = log.history['val_loss']
            elif epochs_since_best >= patience:
                print("Exceeded patience, halting training...")
                break
                
            print()
            
            #Resetting states
            self.model.reset_states()
            print('Epoch: %.d' % (i + 1))

        # print(log)

    def train_dense(self):


        self.model.compile(loss='mae', optimizer='adam', metrics=['mae','mse',self.rmse])

        self.model.fit(x=self.x_train, y=self.y_train, epochs = self.epochs, 
        batch_size=self.batch_size, verbose = 2, validation_split = self.validation_part, 
        shuffle=True)


    def nn_pred(self):

        results = self.model.evaluate(self.x_test, self.y_test)

        print(results)

    def predict(self):

        i = int(self.x_test.shape[0]/self.batch_size)

        eval = np.ndarray(shape=(i,4))


        for index in range(i):

            # self.predictions = self.model.predict(self.x_test[index*self.batch_size:(index+1)*self.batch_size], batch_size=self.batch_size)

            self.evaluation = self.model.evaluate(self.x_test[index*self.batch_size:(index+1)*self.batch_size], self.y_test[index*self.batch_size:(index+1)*self.batch_size], batch_size=self.batch_size)

            # print('Evaluating with test data')
            # print(self.model.metrics_names)
            # print(self.evaluation)
            eval[index,:] = self.evaluation
            # print()
 
        final_eval = (eval.mean(axis=0))

        return final_eval, self.model.metrics_names


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
        inverse_predictions = self.predictions
        inverse_actuals = self.y_test
        # inverse_predictions = self.predictions_last
        # inverse_actuals = self.y_test_last

        lines = pyplot.plot(inverse_predictions, 'r', inverse_actuals, 'b')
        pyplot.setp(lines, linewidth=0.5)
        pyplot.show()

    def write_list_to_file(self,file,list):

        for i,item in enumerate(list):
            file.write(" {} ".format(item))

        file.write('\n')

if __name__ == '__main__':
    datapath = os.path.join('..','..','data','Ytre Vikna', 'data_ytrevikna_advanced.csv')

    logfile = open('results.txt','w')

    nn_network = NETWORK(datapath, False)

    # nn_network.build_lstm()
    # nn_network.train_lstm()
    # results, metrics = nn_network.predict()
    # logfile.write('\nLstm no pca\n')
    # nn_network.write_list_to_file(logfile,metrics)
    # nn_network.write_list_to_file(logfile,results)


    # print(results)
    # exit(0)


    nn_network.build_dense()
    nn_network.train_dense()
    # results,_ = nn_network.predict()

    nn_network.nn_pred()

    exit(0)
    logfile.write('\ndense no pca\n')
    nn_network.write_list_to_file(logfile,results)

    nn_network = NETWORK(datapath, pca = False)

    nn_network.build_lstm()
    nn_network.train_lstm()
    results,_ = nn_network.predict()
    logfile.write('\nLstm no pca\n')
    nn_network.write_list_to_file(logfile,results)
    
    nn_network.build_dense()
    nn_network.train_dense()
    results,_ = nn_network.predict()
    logfile.write('\ndense no pca\n')
    nn_network.write_list_to_file(logfile,results)
    
    
    exit(0)
