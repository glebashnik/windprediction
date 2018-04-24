import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import h5py

import xgboost as xgb

from util.processing import process_dataset_lstm, process_dataset_nn, process_dataset_conv_nn
from util.visualization import visualize_loss_history
from util.logging import write_results
from data.dataset_generator import *
from util.data_analysis import *

# from models.simple_lstm.main import RNN as LSTM
# from models.lstm_stateful.main import RNN as StatefulLSTM
from models.simple_ann.main import NN
from models.dense_nn_forest.NN_forest import *
from models.ann_error_feedback.ann_feedback import NN_feedback
from models.nn_dual_loss.nn_dual_loss import NN_dual
from models.lstm_stateful.main import RNN
from models.lstm_stateful.main import RNN
from models.conv_nn.conv_nn import Conv_NN

from keras.utils.vis_utils import plot_model

import xgboost as xgb

import matplotlib.pyplot as plt

# from keras import optimizers
from models.random_forest.random_forest import RandomForest

import h5py

model_path = os.path.join('checkpoint_model.h5')

park = 'Bessaker large'
latest_scream_dataset_path = os.path.join(
    'data', park, 'dataset_20130818-20180420.csv')

dataset = Bessaker_dataset_sparse(latest_scream_dataset_path)
dataset, target = create_dataset_history(dataset, history_length=12)

# Selection of gpu
parser = argparse.ArgumentParser(
    description='Main script for training av evaluating assorted networks')
parser.add_argument('--gpu', required=True,
                    help='Select which GPU to train on')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('Training on GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

# Hyperparameters for training network
testsplit = 0.96
look_back = 6
look_ahead = 1
epochs = 1000
batch_size = 64
lr = 0.001
decay = 1e-6
momentum = 0.9

######################################


# print('Beginning model training on the path: {}'.format(model_path))
# print('Data loaded with {} attributes\n'.format(len(dataset.columns)))

# Dense
# x_train, x_test, y_train, y_test = process_dataset_nn(
#     dataset,
#     testsplit=testsplit
# )


def visualize_training_buckets(file_path):

    history = h5py.File(file_path, 'r')

    buckets = history['buckets'].value
    evaluations = history['evaluations'].value

    plt.plot(buckets[1:], evaluations[1:], 'bo')
    plt.title('Network training loss for different dataset sizes')
    plt.ylabel('MAE')
    plt.xlabel('Dataset sizes')
    # plt.legend(metrics, loc='upper right')
    # if (start != None) and (end != None):
    #     plt.xlim(start, end)
    plt.show()


opt = [
    # optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True),
    # optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
    # 'rmsprop',
    'adam'
]

opt_name = [
    # 'SGD lr: {} decay: {} momentum: {} '.format(lr,decay,momentum),
    # 'Adam lr: {} decay: {} '.format(lr,momentum),
    # 'rmsprop',
    'adam'
]

# Define networks, dimensions: (models,networks,layers)
best_network = [
    [(32, False), (16, False), (8, True), (2, False)],
    [(128, False), (64, True), (32, False), (8, False)],
    [(64, False), (16, False), (6, True), (0, False)],
    [(16, False), (8, False), (8, True), (0, False)],
]

network_forest = [
    [(32, False), (16, False), (8, False), (2, False)],
    [(128, False), (64, False), (32, False), (8, False)],
    [(64, False), (16, False), (6, False), (0, False)],
    [(16, False), (8, False), (8, False), (0, False)],
    [(32, False), (16, False), (8, False), (2, False)],
    [(64, False), (64, False), (32, False), (16, False)],
    [(12, False), (6, False), (3, False), (0, False)],
    [(69, False), (128, False), (32, False), (4, False)],
]


feedback_network = [(32, False), (16, True), (8, False), (2, False)]
dropouts = [0.2, 0.3, 0.4, 0.5]

lstm_layers = [64, 32, 16, 8]


def execute_network_simple(dataset, note, epochs, dropoutrate=0, opt='adam', write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    print('Training with {} features'.format(num_features))

    network = NN_dual(model_path=model_path, batch_size=32, epochs=epochs,
                      dropoutrate=dropoutrate)

    model_architecture = network.build_model(
        input_dim=num_features, output_dim=num_targets)
    model_architecture.summary()

    hist_loss, model = network.train_network(
        x_train=x_train, y_train=y_train, opt=opt)

    evaluation, metric_names = network.evaluate(x_test, y_test)

    if write_log:
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)

    return evaluation


def execute_network_advanced(dataset, note, layers, epochs, dropoutrate=0.3, opt='adam', write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    network = NN_dual(model_path=model_path, batch_size=32, epochs=epochs,
                      dropoutrate=dropoutrate)
    model_architecture = network.build_forest_model(
        input_dim=num_features, model_structure=layers)
    model_architecture.summary()

    hist_loss, model = network.train_network(
        x_train=x_train, y_train=y_train, opt=opt)

    evaluation, metric_names = network.evaluate(
        x_test, y_test, single_targets=False)

    if write_log:
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)


def execute_network_lstm(dataset, note, layers, epochs, dropoutrate=0.3, opt='adam', write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_lstm(
        dataset, look_back=6, look_ahead=1, testsplit=testsplit, stateful=True, batch_size=batch_size)

    input_shape = x_train.shape[1:]
    num_features = x_train.shape[2]

    lstm_network = RNN(batch_size, epochs)

    model_architecture = lstm_network.build_model_general(
        input_shape=input_shape, layers=layers)

    lstm_network.train_network(
        x_train=x_train, y_train=y_train)

    evaluation, metric_names = lstm_network.evaluate(
        x_test, y_test)

    if write_log:
        write_results(park, model_architecture, note, num_features,
                      None, evaluation, metric_names, epochs, opt, dropoutrate, ahed=1, back=6)


def execute_random_forest(dataset, notes):

    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    forest = RandomForest_featureimportance()

    forest.train(x_train, y_train)

    evaluation = forest.test(x_test, y_test)

    print('Test evaluation on random forest: {}'.format(evaluation))


def execute_xgb(dataset, notes):
    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit)

    training_data = xgb.DMatrix(x_train, y_train)
    test_data = xgb.DMatrix(x_test, y_test)

    params = {'eval_metric':'mae', 'max_depth':10, 'subsample':0.5, 'eta':0.1, 'n_estimators':1000}
    num_round = 1
    model = xgb.train(params, training_data, num_round)
    pred = model.predict(test_data)
    print("MAE is ", np.average(np.abs(pred-y_test)))

def execute_conv_network(dataset, target, note, write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_conv_nn(dataset, target, testsplit=testsplit)
    
    history_length = np.shape(x_train)[1]
    num_features = np.shape(x_train)[2]

    network = Conv_NN(epochs=epochs, batch_size=batch_size, model_path=model_path)

    model_architecture = network.build_model(history_length, num_features)
    model_architecture.summary()
    hist_loss, model = network.train_network(x_train, y_train, opt=opt)

    evaluation, metric_names = network.evaluate(x_test, y_test)

    if write_log:
        dropoutrate = 0 
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)

    # Creates model, trains the network and saves the evaluation in a txt file.
    # Requires a specified network and training hyperparameters


# ========== Comment in the model you want to run here ==========
# execute_network_simple(
#     dataset, 'Training with 38700 samples', epochs, write_log=True, single_targets=False)

# dataset = dataset[0:8000]

# execute_network_simple(
#     dataset, 'Training with 8000 samples', epochs, write_log=True, single_targets=False)

# exit(0)
# execute_network_advanced(
#     dataset, 'training on network forest', best_network, epochs, write_log=True)

# execute_network_advanced(
    # dataset, 'training best network forest, only 38700', best_network, epochs, write_log=True)

# visualize_training_buckets('training_data_buckets_1st_2000e.hdf5')

# evaluation = execute_network_simple(
#     dataset, 'Training simple network with new dataset and dropout', epochs, write_log=True)
# exit(0)
# data_buckets = [200, 500, 1000, 2000,
#                 4000, 6000, 10000, 16000, 20000, 24000, 28000, dataset.shape[0]]
# evaluation_list = []
# for i, bucket in enumerate(data_buckets):
#     subdataset = dataset[0:bucket]

#     print(subdataset.shape[0])

#     evaluation = execute_network_simple(
#         subdataset, 'Training simple network with {} samples'.format(subdataset.shape[0]), epochs, write_log=True)

#     evaluation_list.append(evaluation[0])

# with h5py.File('large test buckets.hdf5', 'w') as f:
#     print('Saving all model evaluations in h5 file')
#     f.create_dataset('buckets', data=data_buckets)
#     f.create_dataset('evaluations', data=evaluation_list)

# exit(0)

# execute_network_advanced(
#     dataset, 'training on network forest', network_forest, epochs, write_log=True)

# dataset = feature_importance(
#     dataset, scope=3000, num_features=48, print_=False)


# execute_network_simple(
#     dataset, 'Training on feature importance adv dataset', epochs, write_log=True)

# execute_network_advanced(
#     dataset, 'Training on feature importance adv dataset', layers[0], epochs, write_log=True)
# execute_network_advanced(
#     dataset, 'Training on feature importance adv dataset', layers[1], epochs, write_log=True)

execute_conv_network(dataset, target, 'Conv network', write_log=True)

# execute_xgb(dataset, 'XG BOOST')