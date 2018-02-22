import numpy as np

from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler


def process_dataset(dataset, look_back=1, look_ahead=1, testsplit=0.8):
    dataset = dataset.fillna(0)

    # Assuming target value is the last in the dataset, dropping the two last rows because of missing targets
    data_x = dataset.iloc[:-2, :-1]

    # Need to discard the first values because of dropping in time series
    data_y = dataset.iloc[look_back:-2, -1:].values
    data_y = data_y.reshape(data_y.shape[0], 1)

    #Normalizing data
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    data_x = scaler.fit_transform(data_x)

    # Returns an (n_in * n_out) * num_vars NDFrame
    timeseries = series_to_supervised(data=data_x, n_in=look_back, n_out=look_ahead, dropnan=True).values
    
    # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
    timeseries_data = timeseries.reshape(timeseries.shape[0], look_back + look_ahead, data_x.shape[1])
    
    #Split for dividing the dataset in a factor of the batch size
    split = int(testsplit * data_x.shape[0])

    # Create training and test sets
    x_train = timeseries_data[:split, :, :]
    x_test = timeseries_data[split:, :, :]

    y_train = data_y[:split, :]
    y_test = data_y[split:, :]

    print(x_train)
    print(y_train)

    return x_train, x_test, y_train, y_test

def generate_test_set(dataset, look_back=4, look_ahead=0):
    dataset = dataset.fillna(0)

    #Normalizing data
    # scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    # data = scaler.fit_transform(dataset)
    data = dataset

    # Returns an (n_in * n_out) * num_vars NDFrame
    timeseries = series_to_supervised(data=data, n_in=look_back, n_out=look_ahead, dropnan=True)

    # Converts to numpy representation
    timeseries_np = timeseries.values

    # Reshape to three dimensions (number of samples x number of timesteps x number of variables)
    return timeseries_np.reshape(timeseries_np.shape[0], look_back + look_ahead, data.shape[1])

def make_time_series(row_features, look_back_num):
    row_features = row_features.fillna(0).values
    list_of_matrices = []
    num_time_steps = look_back_num
    i = num_time_steps
    
    while i < len(row_features):
        list_of_matrices.append(row_features[(i-num_time_steps):i,:])
        i = i + 1
    return np.array(list_of_matrices).reshape(len(list_of_matrices), 17, look_back_num)

# convert series to supervised learning, time series data generation
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg