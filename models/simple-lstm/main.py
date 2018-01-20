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
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2


class RNN:

    def __init__(self,datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0)
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)


        self.validsplit = 0.7
        self.testsplit = 0.9

        self.batch_size = 32
        self.epochs = 1000

        self.data = self.dataset.values

        #Normalizing data
        scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        scaler.fit(self.data)
        self.data = scaler.transform(self.data)

        n_in = 10
        n_out = 3

        print(self.data)
        self.timeseries = self.series_to_supervised(data=self.data, n_in=n_in, n_out=n_out, dropnan=True)

        self.timeseries_np = self.timeseries.values

        self.timeseries_data = self.timeseries_np.reshape(self.timeseries_np.shape[0],n_in+n_out,self.data.shape[1])

        self.x_data, self.y_data = self.timeseries_data[:,:,:-1],self.timeseries_data[:,:,-1]

        self.x_train, self.x_valid, self.x_test = self.x_data[0:int(self.validsplit*self.x_data.shape[0]),:,:], \
                                                self.x_data[int(self.validsplit*self.x_data.shape[0]):int(self.testsplit*self.x_data.shape[0]),:,:], \
                                                self.x_data[int(self.testsplit*self.x_data.shape[0]):,:,:]

        # Keep the two last elements as validation items
        self.y_train, self.y_valid, self.y_test = self.y_data[0:int(self.validsplit*self.y_data.shape[0]),-2:], \
                                                self.y_data[int(self.validsplit*self.y_data.shape[0]):int(self.testsplit*self.y_data.shape[0]),-2:], \
                                                self.y_data[int(self.testsplit*self.y_data.shape[0]):,-2:]

    def build_model(self):

        self.alpha = 0.1
        # input_tensor = Input(shape=(self.x_data.shape[1], self.x_data.shape[2]))
        # print(input_tensor.shape)

        self.model = Sequential()
        self.model.add(LSTM(32, input_shape=(self.x_data.shape[1], self.x_data.shape[2]), return_sequences=True))
        self.model.add(LSTM(2))
    
        #self.model = Model(inputs=input_tensor, outputs=x)
        self.model.summary()
        print('Model created')

    def train_network(self):
        self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
        self.model.compile(loss='mae', optimizer=self.optimizer,
                           metrics=['mae','mse',self.rmse])
        print("compiled")
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        print(self.y_train.shape)
        log = self.model.fit(x=self.x_train,y=self.y_train, batch_size=self.batch_size, epochs = self.epochs,verbose=2,
                            callbacks=[checkpoint,early_stopping], validation_split=0.2,shuffle=True)

        # print(log.history)

        # with open(os.path.join('results.pickle'), 'wb') as f:
        #     pickle.dump(log.history, f)

        # self.model.save(os.path.join('model.h5'))

        # print('--Model trained and saved--')

    def predict(self):
        self.predictions = self.model.predict(self.x_test)


        self.evaluation = self.model.evaluate(self.x_test,self.y_test)

        print('Evaluating with test data')
        print(self.model.metrics_names)
        print(self.evaluation)

        print('Prediction --vs-- label')
        print(np.concatenate((self.predictions[0:10],np.reshape(self.y_test[0:10],(10,1))),axis=1))

        print('evaluating with numpy ')
        print(np.mean(self.rmse_numpy(self.y_test,self.predictions)))


    def visualize_data(self):
        vis_df = read_csv(self.datapath,index_col=0,header=0)
        values = vis_df.values

        #First weather station is selected for visualizastion
        vis_group = [0,1,2,3,20,21]

        i=1
        pyplot.figure()
        for group in vis_group:
            pyplot.subplot(len(vis_group), 1, i)
            pyplot.plot(values[:, group])
            pyplot.title(vis_df.columns[group], y=0.9, loc='right')
            i += 1
        pyplot.show()

        # print(self.dataset.head(5))

    #Custom made rmse loss (aparently missing in keras library)

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

if __name__ == '__main__':
    datapath = os.path.join('data', 'cleaned_data.csv')

    nn_network = RNN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()

    # rnn_network.visualize_data()

    #must get the series to supervised to work
    #