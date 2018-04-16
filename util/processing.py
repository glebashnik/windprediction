import numpy as np

from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Returns the dataset split into features and target (assuming target is the last index in the DF)
# Also scales the features into values ranging from 0 to 1
# ALl returned values are NDArrays


def feature_target_split(dataset):
    dataset = dataset.dropna()

    data_x = dataset.iloc[:, :-1]
    data_y = dataset.iloc[:, -1:]

    return data_x, data_y.values


def feature_single_target_split(dataset):
    dataset = dataset.dropna()

    data_x = dataset.iloc[:, :-25]
    data_y = dataset.iloc[:, -25:]

    return data_x, data_y.values

# Prepares the dataset for use in a dense neural network
# Splits the dataset into features and target, scales and divides into training and test sets
# If PCA option is True, PCA will be performed on the features as well
# ALl returned values are NDArrays


def process_dataset_nn(dataset, testsplit=0.8, pca=False):
    data_x, data_y = feature_target_split(dataset)
    if pca:
        data_x = extract_PCA_features(data_x, n_components=40)
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=1-testsplit, random_state=1745)

def production_history_data_split(data, production_col_name='Produksjon'):
    production_data = data.filter(regex=production_col_name, axis=1).as_matrix()
    production_data = np.expand_dims(production_data, axis=2)
    rest_data = data.filter(regex='^(?!{})'.format(production_col_name), axis=1)

    return production_data, rest_data

def process_dataset_conv_nn(dataset, production_col_name='Produksjon', testsplit=0.8):
    data_x, data_y = feature_target_split(dataset)

    production_data, rest_data = production_history_data_split(data_x, production_col_name)

    x_prod_train, x_prod_test, y_train, y_test = train_test_split(
        production_data, data_y, test_size=1-testsplit, random_state=1745)

    x_rest_train, x_rest_test, y_train, y_test = train_test_split(
        rest_data, data_y, test_size=1-testsplit, random_state=1745)

    return x_prod_train, x_rest_train, x_prod_test, x_rest_test, y_train, y_test

def process_dataset_nn(dataset, testsplit=0.8, pca=False, single_targets=False):
    if not single_targets:
        data_x, data_y = feature_target_split(dataset)
    else:
        data_x, data_y = feature_single_target_split(dataset)

    if pca:
        data_x = extract_PCA_features(data_x, n_components=40)

    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=1-testsplit, random_state=1745)

    print('Loaded training and test data with shape {} and {}, respectively'.format(
        x_train.shape, y_train.shape))

    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


# Prepares the dataset for use in an LSTM network
# Splits the dataset into features and target, creates timeseries based on look-back and look-ahead
# and splits into training and test sets
# If PCA option is True, PCA will be performed on the features as well
# If stateful option is set, batch_size should be set as well. The training and test will then
# have length equal to a multiple of batch_size
def process_dataset_lstm(dataset, look_back=1, look_ahead=1, testsplit=0.8, stateful=False, batch_size=32, pca=False):
    data_x, data_y = feature_target_split(dataset)
    if pca:
        data_x = extract_PCA_features(data_x, n_components=40)

    data_y = data_y[look_back:, :]
    timeseries_data = create_timeseries(
        data_x, look_back=look_back, look_ahead=look_ahead)

    split = int(testsplit * timeseries_data.shape[0])
    if stateful:
        split = int(split - (split % batch_size))

    test_split = timeseries_data.shape[0] - split
    test_split = test_split - test_split % batch_size
    test_split = int(
        split + test_split) if stateful else timeseries_data.shape[0]

    # Create training and test sets
    x_train = timeseries_data[:split, :, :]
    x_test = timeseries_data[split:test_split, :, :]

    y_train = data_y[:split, :]
    y_test = data_y[split:test_split, :]

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


def extract_PCA_features(data, n_components=10):
    pca = PCA(n_components=n_components)
    data_x_pca = pca.fit_transform(data)
    return data_x_pca
