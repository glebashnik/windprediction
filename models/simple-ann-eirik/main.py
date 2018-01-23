from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from keras import backend as K
from keras.layers import *
from keras.models import Model, Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2

class NN:

    def __init__(self, datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0, sep=";")
            # self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)

        # Drop rows without values
        self.dataset = self.dataset.dropna()

        self.testsplit = 0.7
        self.batch_size = 128
        self.epochs = 2500

        # Select only windspeed and direction as well as STORM data, number of working mills, delayed prod
        # Currently only windspeed
        # data_x = pd.concat([
        #     #self.dataset.iloc[:,2:3], 
        #     #self.dataset.iloc[:,6:7], 
        #     self.dataset.iloc[:,10:11], 
        #     self.dataset.iloc[:,14:15], 
        #     self.dataset.iloc[:,18:-1]], 
        #     axis=1)
        # data_x = pd.concat([
        #     self.dataset.iloc[:,0:-3], 
        #     self.dataset.iloc[:,-2:-1]], 
        #     axis=1)
        data_x = self.dataset.iloc[:,0:-1]
        data_y = self.dataset.iloc[:,-1:]

        print(data_x.shape)
        print(data_y.shape)
        #exit(0)

        #Normalizing inputs
        self.x_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        self.x_scaler.fit(data_x)
        self.x_data = self.x_scaler.transform(data_x)

        #Do not normalize outputs
        self.y_data = data_y.values

        # Create training, validation and test sets for x
        self.x_train = self.x_data[0:int(self.testsplit * self.x_data.shape[0]), :]
        self.x_test = self.x_data[int(self.testsplit * self.x_data.shape[0]):, :]

        # Create training, validation and test sets for y
        self.y_train = self.y_data[0:int(self.testsplit * self.y_data.shape[0]), :]
        self.y_test = self.y_data[int(self.testsplit * self.y_data.shape[0]):, :]

    def build_model(self):
        self.model = Sequential()
        alpha = 0.6

        # Input layer
        self.model.add(Dense(32, input_dim=self.x_data.shape[1]))
        self.model.add(Activation('relu'))

        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        #self.model.add(BatchNormalization())

        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        #self.model.add(BatchNormalization())

        # self.model.add(Dense(32))
        # self.model.add(Activation('relu'))
        # #self.model.add(BatchNormalization())
        
        # Output layer
        self.model.add(Dense(1))
    
        self.model.summary()

    def train_network(self):
        self.model.compile(loss='mae', optimizer='adam',metrics=['mae','mse',self.rmse])

        # Perform early stop if there was not improvement for n epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)

        # Train the model
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopping], epochs = self.epochs, verbose=2, shuffle=True)

    def predict(self):
        self.predictions = self.model.predict(self.x_test)
        self.evaluation = self.model.evaluate(self.x_test, self.y_test)

        print('Evaluating with test data..')
        print(self.model.metrics_names)
        print(self.evaluation)
        print()

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def visualize(self):
        pyplot.plot(self.predictions, 'r', self.y_test, 'b')
        pyplot.show()

if __name__ == '__main__':
    np.random.seed(42)

    datapath = os.path.join('data', 'Advanced_data2.csv')

    nn_network = NN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()
    nn_network.visualize()