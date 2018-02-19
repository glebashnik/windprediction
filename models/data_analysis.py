from pandas import read_csv, DataFrame, concat
import pandas as pd
from matplotlib import pyplot
import os
import _pickle as pickle
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt


def extract_data(datapath, normalize = True):
    datapath = datapath
    try:
        dataset = read_csv(datapath, index_col=0, sep=';')
        dataset.index.name = 'index'
        print('Data loaded, shape: ' + str(dataset.shape))
    except:
        print('No data found on: ' + datapath)
        exit(1)

    dataset = dataset.dropna()

    data_x = dataset.iloc[:,:-1]
    data_y = dataset.iloc[:,-1:]
    #Normalizing data (EXCEPT THE LABEL)
    if normalize:

        x_scaler = MinMaxScaler(copy=True,feature_range=(0,1))
        data_x = x_scaler.fit_transform(data_x)

        dataset.iloc[:,:-1] = data_x
        dataset.iloc[:,-1:] = data_y

    return dataset



def extract_pearson_features(dataset, treshold = 0.5):

    pearson_corr = dataset.corr(method='pearson',min_periods = 1)
    topcorr = pearson_corr.iloc[-1]

    print('\nPearson correlation with total production as target\n')
    print(topcorr)
    print('\n')



    n_dropped = 0

    for i,pear_val in enumerate(topcorr):
        
        if np.abs(pear_val) < treshold:
            print('dropping: ' + topcorr.keys()[i])
            dataset = dataset.drop(topcorr.keys()[i], axis=1)
            n_dropped += 1

    print('\nnumber of features dropped: {}\n'.format(n_dropped))

    return dataset


def extract_PCA_features(dataset, n_components = 10):


    data_x = dataset.iloc[:,:-1]
    data_y = dataset.iloc[:,-1:]


    pca = PCA(n_components=n_components)
    data_x_pca = pca.fit_transform(data_x)

    data = np.concatenate((data_x_pca, data_y),axis = 1)

    return data, pca
    

def prediction(data, testsize = 0.15):

    x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1:], test_size=testsize, random_state=666)

    ols = linear_model.LinearRegression()
    model = ols.fit(x_train,y_train)

    predictions = model.predict(x_test)

    print('MAE: {}\n'.format(mean_absolute_error(y_test,predictions)))

    return predictions, y_test


def visualize(pred_list, legend, scope = 0.2):

    length = int(scope * pred_list[0].shape[0])

    for i,pred in enumerate(pred_list):
        plt.plot(pred[0:length])

    plt.legend(legend, loc='upper right')
    plt.show()


if __name__ == '__main__':
    datapath_old = os.path.join('..','data', 'Advanced_data2.csv')
    datapath_simple = os.path.join('..','data', 'Data_simple.csv')
    datapath_advanced = os.path.join('..','data', 'Data_advanced.csv')

    testsize = 0.15
    n_components = 12
    treshold = 0.7


    dataset = extract_data(datapath_old)

    # Extract PCA features
    PCA_data, PCA_model = extract_PCA_features(dataset,n_components)

    # Only variables with acceptable pearson correlation with output is allowed to pass
    pearson_dataset = extract_pearson_features(dataset, treshold)

    # Perform linear regression
    print('\nPredicting with raw data {} features'.format(dataset.shape[1]))
    raw_pred, _ = prediction(dataset.values, testsize)

    print('Predicting with {} PCA features'.format(n_components))
    PCA_pred, _ = prediction(PCA_data, testsize)
    
    print('Predicting with pearson correlations above {}, gives {} features'.format(treshold, pearson_dataset.shape[1]))
    pearson_pred, y_test = prediction(pearson_dataset.values, testsize)

    # Compare the prediction methods
    visualize([y_test,raw_pred,PCA_pred,pearson_pred],legend=['GT','Raw','PCA','Pearson'],scope = 0.2)

    exit(0)