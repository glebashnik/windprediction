import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime

from util.processing import process_dataset_lstm, process_dataset_nn
from util.visualization import visualize_loss_history
from util.logging import write_results
from data.dataset_generator import *
from util.data_analysis import *

# from models.simple_lstm.main import RNN as LSTM
# from models.lstm_stateful.main import RNN as StatefulLSTM
from models.simple_ann.main import NN
from models.dense_nn_forest.NN_forest import NN_forest
from models.ann_error_feedback.ann_feedback import NN_feedback
from models.nn_dual_loss.nn_dual_loss import NN_dual

# from keras import optimizers
from models.random_forest.main import RandomForest

import h5py


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# datapath = os.path.join('data','Ytre Vikna', 'data_ytrevikna_advanced.csv')
# datapath = os.path.join('data','Skomakerfjellet', 'data_skomakerfjellet_advanced.csv')
park = 'Bessaker Vindpark'

datapath = os.path.join('data', park,
                        'data_bessaker_advanced.csv')

tek_path = os.path.join('rawdata', 'vindkraft 130717-160218 TEK met.csv')
arome_path = os.path.join('rawdata', 'vindkraft 130717-160218 arome.csv')
model_path = os.path.join('checkpoint_model.h5')

# dataset = generate_bessaker_dataset(tek_path, arome_path)
dataset = generate_bessaker_dataset_extra(tek_path, arome_path)
# dataset = generate_bessaker_dataset_single_target(tek_path, arome_path)


# # Extracting indices of most important features
# dataset = dataset.drop(['BESS-Bessakerfj.-GS-T4015A3 -0104'], axis=1)
# dataset = feature_importance(
#     dataset, scope=3000, num_features=40, print_=True)
# exit(0)

# visualize_loss_history('M03-D21_h20-m07-s13')
# exit(0)

# Hyperparameters for training network
testsplit = 0.7
look_back = 6
look_ahead = 1
epochs = 3000
batch_size = 64
lr = 0.001
decay = 1e-6
momentum = 0.9


print('Beginning model training on the path: {}'.format(model_path))
print('Data loaded with {} atributes\n'.format(len(dataset.columns)))

# Dense
# x_train, x_test, y_train, y_test = process_dataset_nn(
#     dataset,
#     testsplit=testsplit
# )


opt = [
    # optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True),
    # optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
    #'rmsprop',
    'adam'
]

opt_name = [
    #'SGD lr: {} decay: {} momentum: {} '.format(lr,decay,momentum),
    #'Adam lr: {} decay: {} '.format(lr,momentum),
    #'rmsprop',
    'adam'
]

# Define networks, domensions: (models,networks,layers)
best_network = [
    [(32, False), (16, False), (8, False), (2, False)],
    [(128, False), (64, False), (32, False), (8, False)],
    [(64, False), (16, False), (6, False), (0, False)],
    [(16, False), (8, False), (8, False), (0, False)],
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


def execute_network_simple(dataset, note, epochs, dropoutrate=0, opt='adam', write_log=False, single_targets=False):

    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit, single_targets=single_targets)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    network = NN_dual(model_path=model_path, batch_size=32, epochs=epochs,
                      dropoutrate=dropoutrate)

    if not single_targets:
        model_architecture = network.build_model(
            input_dim=num_features, output_dim=num_targets)
    else:
        model_architecture = network.build_model_single_targets(
            input_dim=num_features, output_dim=num_targets)

    hist_loss, model = network.train_network(
        x_train=x_train, y_train=y_train, opt=opt)

    evaluation, metric_names = network.evaluate(x_test, y_test, single_targets)

    if write_log:
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)


def execute_network_advanced(dataset, note, layers, epochs, dropoutrate=0.3, opt='adam', write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_nn(
        dataset, testsplit=testsplit)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    network = NN_dual(model_path=model_path, batch_size=32, epochs=epochs,
                      dropoutrate=dropoutrate)
    model_architecture = network.build_forest_model(
        input_dim=num_features, model_structure=layers)

    hist_loss, model = network.train_network(
        x_train=x_train, y_train=y_train, opt=opt)

    evaluation, metric_names = network.evaluate(
        x_test, y_test, single_targets=False)

    if write_log:
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)

# Creates model, trains the network and saves the evaluation in a txt file.
# Requires a specified network and training hyperparameters


execute_network_simple(
    dataset, 'Regular network, dataset advanced', epochs, write_log=True, single_targets=False)
execute_network_advanced(
    dataset, 'training on network forest', best_network, epochs, write_log=True)
execute_network_advanced(
    dataset, 'training on network forest', network_forest, epochs, write_log=True)
exit(0)

dataset = feature_importance(
    dataset, scope=3000, num_features=48, print_=False)

execute_network_simple(
    dataset, 'Training on feature importance adv dataset', epochs, write_log=True)

execute_network_advanced(
    dataset, 'Training on feature importance adv dataset', layers[0], epochs, write_log=True)
execute_network_advanced(
    dataset, 'Training on feature importance adv dataset', layers[1], epochs, write_log=True)
