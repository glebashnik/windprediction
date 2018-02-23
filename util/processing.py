import numpy as np

from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Returns the dataset split into features and target (assuming target is the last index in the DF)
# Also scales the features into values ranging from 0 to 1
# ALl returned values are NDArrays
def feature_target_split(dataset):
    dataset = dataset.dropna()

    data_x = dataset.iloc[:, :-1]
    data_y = dataset.iloc[:, -1:]

    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    data_x = scaler.fit_transform(data_x)

    return data_x, data_y.values

# Prepares the dataset for use in a dense neural network
# Splits the dataset into features and target, scales and divides into training and test sets
# ALl returned values are NDArrays
def process_dataset_nn(dataset, testsplit=0.8):
    data_x, data_y = feature_target_split(dataset)
    return train_test_split(data_x, data_y, test_size=1-testsplit)

# Prepares the dataset for use in an LSTM network
# Splits the dataset into features and target, creates timeseries based on look-back and look-ahead
# and splits into training and test sets
def process_dataset_lstm(dataset, look_back=1, look_ahead=1, testsplit=0.8):
    data_x, data_y = feature_target_split(dataset)
    data_y = data_y[look_back:, :]
    timeseries_data = create_timeseries(data_x, look_back=look_back, look_ahead=look_ahead)
    
    split = int(testsplit * timeseries_data.shape[0])

    # Create training and test sets
    x_train = timeseries_data[:split, :, :]
    x_test = timeseries_data[split:, :, :]

    y_train = data_y[:split, :]
    y_test = data_y[split:, :]

    return x_train, x_test, y_train, y_test

# Prepares the dataset for use in a stateful LSTM network
# Splits the dataset into features and target, creates timeseries based on look-back and look-ahead,
# fits to the current batch size and splits into training and test sets

# May not work when validation_split in model.fit() is used, due to internal splitting of training set
# into a number that may not be divisible by the batch size
def process_dataset_lstm_stateful(dataset, look_back=1, look_ahead=1, testsplit=0.8, batch_size=32):
    data_x, data_y = feature_target_split(dataset)

    data_y = data_y.reshape(data_y.shape[0], 1)

    timeseries_data = create_timeseries(data_x, look_back=look_back, look_ahead=look_ahead)
    
    #Split for dividing the dataset in a factor of the batch size
    split = testsplit * timeseries_data.shape[0] 
    split -= split % batch_size
    split = int(split)

    test_split = timeseries_data.shape[0] - split
    test_split = test_split - test_split % batch_size
    test_split = int(split + test_split)
    
    # Create training and test sets
    x_train = timeseries_data[:split, :, :]
    x_test = timeseries_data[split:test_split, :, :]

    y_train = data_y[:split, :]
    y_test = data_y[split:test_split:, :]

    return x_train, x_test, y_train, y_test

# Creates a timeseries from a DataFrame
# Returns an (look_back * look_ahead) * num_vars NDFrame
# Drops the first #look_back rows due to these necessarily having NAN values in the timeseries
def create_timeseries(data, look_back=1, look_ahead=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
    for i in range(look_back, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    
    for i in range(0, look_ahead):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values.reshape(agg.shape[0], look_back + look_ahead, data.shape[1])