from pandas import read_csv, DataFrame
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.regularizers import l2


class NN:

    def __init__(self,datapath):
        self.datapath = datapath
        try:
            self.dataset = read_csv(self.datapath, index_col=0)
            self.dataset.index.name = 'index'
            print('Data loaded, shape: ' + str(self.dataset.shape))
        except:
            print('No data found on: ' + self.datapath)
            exit(1)


        #Organizing the data
        self.x_data = self.dataset.iloc[:,0:-1].values
        self.y_data = self.dataset.iloc[:,-1].values


        #CHECK BOUNDARIES ON DATA SUCH THAT MAX IS 1 AND MIN IS 0
        #normalizing featurewise
        for i in range(self.x_data.shape[1]):
            #-= MIN
            self.x_data[:,i] /= self.x_data[:,i].max()


        # self.y_data /= self.y_data.max()

        #Splitting into train and test data sets with fixed seed
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data,
                                                                                test_size=0.1, random_state=42)
        #Hyperparameters
        self.batch_size = 32
        self.epochs = 1000
        # vis_group = [0, 1, 2, 3, 20, 21]
        # self.dataset = self.dataset.iloc[:,vis_group]

    def build_model(self):

        self.alpha = 0.1
        self.drop_rate = 0.25
        input_tensor = Input([self.x_data.shape[1]])

        x = Dense(100)(input_tensor)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)

        # x = Dropout(rate=self.drop_rate)(x)

        x = Dense(100)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)

        # x = Dropout(rate=self.drop_rate)(x)

        x = Dense(75)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)

        # x = Dropout(rate=self.drop_rate)(x)

        x = Dense(50)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)

        # x = Dropout(rate=self.drop_rate)(x)

        x = Dense(20)(x)
        x = LeakyReLU(alpha=self.alpha)(x)
        x = BatchNormalization()(x)

        # x = Dropout(rate=self.drop_rate)(x)

        x = Dense(1)(x)
        # x = LeakyReLU(alpha=self.alpha)(x)

        self.model = Model(inputs=input_tensor, outputs=x)
        self.model.summary()
        print('Model created')

    def train_network(self):


        self.optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.005)
        self.model.compile(loss='mae', optimizer=self.optimizer,
                           metrics=['mae','mse',self.rmse])

        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        print(os.path.dirname)
        checkpoint = ModelCheckpoint('checkpoint_model.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min')

        log = self.model.fit(x=self.x_train,y=self.y_train, batch_size=self.batch_size, epochs = self.epochs,verbose=2,
                             callbacks=[checkpoint,early_stopping], validation_split=0.2,shuffle=True)

        print(log.history)

        with open(os.path.join('results.pickle'), 'wb') as f:
            pickle.dump(log.history, f)

        self.model.save(os.path.join('model.h5'))

        print('--Model traiend and saved--')

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
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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

    nn_network = NN(datapath)
    nn_network.build_model()
    nn_network.train_network()
    nn_network.predict()

    # rnn_network.visualize_data()

    #must get the series to supervised to work
    #