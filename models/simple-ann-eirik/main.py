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

        self.validsplit = 0.7
        self.testsplit = 0.9
        self.batch_size = 32
        self.epochs = 300

        data_x = self.dataset.iloc[:,0:-3]
        self.y_data = self.dataset.iloc[:,-1:]

        #Normalizing data
        self.x_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        self.x_scaler.fit(data_x)
        self.x_data = self.x_scaler.transform(data_x)

        # #Normalizing output with separate scaler in order to facilitate later inverse scaling
        # self.y_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        # self.y_scaler.fit(data_y)
        # self.y_data = self.y_scaler.transform(data_y)

        # Convert to numpy
        self.data = self.dataset.values

        # Create training, validation and test sets for x
        self.x_train = self.x_data[0:int(self.validsplit * self.x_data.shape[0]), :]
        self.x_valid = self.x_data[int(self.validsplit * self.x_data.shape[0]):int(self.testsplit * self.x_data.shape[0]), :]
        self.x_test = self.x_data[int(self.testsplit * self.x_data.shape[0]):, :]

        # Create training, validation and test sets for y
        self.y_train = self.y_data[0:int(self.validsplit * self.y_data.shape[0]), :]
        self.y_valid = self.y_data[int(self.validsplit * self.y_data.shape[0]):int(self.testsplit * self.y_data.shape[0]), :]
        self.y_test = self.y_data[int(self.testsplit * self.y_data.shape[0]):, :]

    def build_model(self):

        self.alpha = 0.1

        self.model = Sequential()

        self.model.add(Dense(128, input_dim=(self.x_data.shape[1])))
        self.model.add(Dropout(0.3))
        
        self.model.add(Dense(64))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(32))

        self.model.add(Dense(1))
    
        self.model.summary()
        print('Model created')

    def train_network(self):
        self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
        self.model.compile(loss='mae', optimizer=self.optimizer,
                           metrics=['mae','mse',self.rmse])
        print("compiled")

        # Perform early stop if there was not improvement for n epochs
        early_stopping = EarlyStopping(monitor='val_loss', patience=30)

        # Save the best model each time
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        # Train the model
        log = self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batch_size, epochs = self.epochs, verbose=2,
                            callbacks=[checkpoint], validation_split=0.2,shuffle=False)

        self.model.save(os.path.join('model.h5'))

    def predict(self):
        self.predictions = self.model.predict(self.x_test)
        self.evaluation = self.model.evaluate(self.x_test, self.y_test)

        print('Evaluating with test data')
        print(self.model.metrics_names)
        print("Mean absolute error", self.evaluation[1])
        print()

        # print('Prediction --vs-- label')
        # print(np.concatenate((inverse_predictions[0:10], np.reshape(inverse_actuals[0:10], (10,1))),axis=1))
        # print()

        # print('evaluating with numpy ')
        # print(np.mean(self.rmse_numpy(self.y_test,inverse_predictions)))
        # print()

    #RMSE loss function (missing in keras library)
    def rmse_numpy(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))

    def rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def visualize(self):
        inverse_predictions = self.y_scaler.inverse_transform(self.predictions)
        inverse_actuals = self.y_scaler.inverse_transform(self.y_test)

        pyplot.plot(inverse_predictions, 'r', inverse_actuals, 'b')
        pyplot.show()

if __name__ == '__main__':
    datapath = os.path.join('data', 'data-2.3.csv')

    nn_network = NN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()
    nn_network.visualize()