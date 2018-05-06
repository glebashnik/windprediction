#Import libraries
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import h5py

# Import utilities
from util.processing import process_dataset_lstm, process_dataset_nn, process_dataset_conv_nn, process_dataset_nn_last_month
from util.visualization import visualize_loss_history
from util.processing import process_dataset_lstm, process_dataset_nn
from util.visualization import compare_predictions
from util.logging import write_results

# Import models
from models.NN.main import NN
from models.NN_error_feedback.main import NN_feedback
from models.lstm_stateful.main import LSTM_stateful
from models.lstm.main import LSTM
from models.conv_nn.main import Conv_NN
from models.random_forest import RandomForest



os.environ['CUDA_VISIBLE_DEVICES'] = '0'


tek_path = os.path.join('data/raw', 'vindkraft 130717-160218 TEK met.csv')
arome_path = os.path.join(
    'data/raw', 'vindkraft 130717-160218 arome korr winddir.csv')
modelpath = os.path.join('checkpoint_model.h5')


# dataset = generate_bessaker_large_dataset_scratch(
#     os.path.join('data', 'Bessaker large'))
datapath = os.path.join('data','Ytre Vikna', 'data_ytrevikna_advanced.csv')
# datapath = os.path.join('data','Skomakerfjellet', 'data_skomakerfjellet_advanced.csv')
# datapath = os.path.join('data','Bessaker', 'data_bessaker_advanced.csv')
park = 'Bessaker large'
latest_scream_dataset_path = os.path.join(
    'data', park, 'dataset_20130818-20180420.csv')
dataset_bess = Bessaker_dataset(latest_scream_dataset_path)
dataset_vals = Valsnes_dataset(latest_scream_dataset_path)

# datapath = os.path.join('data','Ytre Vikna', 'data_ytrevikna_advanced.csv')
# datapath = os.path.join('data','Skomakerfjellet', 'data_skomakerfjellet_advanced.csv')
# datapath = os.path.join('data', park)

tek_out_path = os.path.join('data', 'tek_out.csv')
# tek_path = os.path.join('rawdata', 'vindkraft 130717-160218 TEK met.csv')
# arome_path = os.path.join('rawdata', 'vindkraft 130717-160218 arome.csv')
model_path = os.path.join('checkpoint_model.h5')

# tek_out_path = os.path.join('data', 'tek_out.csv')
# dataset = generate_bessaker_dataset_single_target(tek_path, arome_path)

# dataset = generate_bessaker_large_dataset(tek_out_path, history_length=12)
# dataset = dataset.dropna()


# dataset = generate_bessaker_large_dataset(datapath)

num_features = len(dataset.columns) -1
gfile = open('results.txt','w')

# Selection of gpu
parser = argparse.ArgumentParser(
    description='Main script for training av evaluating assorted networks')
parser.add_argument('--gpu', required=True,
                    help='Select which GPU to train on')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
print('Training on GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))

# Hyperparameters for training network
testsplit = 0.7
look_back = 6
look_ahead = 1
epochs = 1
batch_size = 64
lr = 0.001
decay = 1e-6
momentum=0.9

print('Beginning model training on the path: {}'.format(model_path))
print('Data loaded with {} attributes\n'.format(len(dataset.columns)))


# Define networks, dimensions: (models,networks,layers)
best_network = [
    [(32, False), (16, False), (8, True), (2, False)],
    [(128, False), (64, True), (32, False), (8, False)],
    [(64, False), (16, False), (6, True), (0, False)],
    [(16, False), (8, False), (8, True), (0, False)],
]

    [[(32,False),(16,True),(8,False),(2,False)],
    [(64,False),(64,True),(32,False),(16,False)],
    [(12,False),(6,True),(3,False),(0,False)],
    [(69,False),(128,True),(32,False),(4,False)]],
    ]

feedback_network = [(32,False),(16,True),(8,False),(2,False)]

dropouts = [0.2, 0.3, 0.4, 0.5]

lstm_layers = [64, 32, 16, 8]


def execute_network_simple(dataset, note, epochs, dropoutrate=0.25, opt='adam', write_log=False):

    x_train, x_test, y_train, y_test = process_dataset_nn_last_month(
        dataset)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    print('Training with {} features'.format(num_features))

    network = NN(model_path=model_path, batch_size=32, epochs=epochs,
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


def execute_network(x_train, x_test, y_train, y_test, layers, epochs, dropoutrate, opt = 'adam', optname='adam'):
    nn_network = NN_forest(batch_size=32, epochs=epochs, dropoutrate=dropoutrate)

    nn_network.build_model(input_dim=num_features,model_structure=layers)

    # nn_network.visualize_model()
    nn_network.train_network(x_train=x_train, y_train=y_train,opt=opt)
    print('training finished')
    
    evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, epochs, optname, dropoutrate)


def execute_network_simple(x_train, x_test, y_train, y_test, layers, epochs, dropoutrate, opt = 'adam', optname='adam'):
    
    network = NN_feedback(batch_size=32, epochs=epochs, dropoutrate=dropoutrate)
    network.build_simple_model(input_dim=num_features,model_structure=layers)
    
    network.train_network(x_train=x_train, y_train=y_train,opt=opt)
    print('training finished')
    
    evaluation, metric_names = network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, epochs, optname, dropoutrate)


def execute_network_feedback(x_train, x_test, y_train, y_test, layers, epochs, dropoutrate, opt = 'adam', optname='adam'):
    
    network = NN_feedback(batch_size=32, epochs=epochs, dropoutrate=dropoutrate)
    network.build_model(input_dim=num_features,model_structure=layers)
    
    # network.visualize_model()
    network.train_network(x_train=x_train, y_train=y_train,opt=opt)
    print('training finished')
    
    evaluation, metric_names = network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, epochs, optname, dropoutrate)


execute_network_simple(x_train, x_test, y_train, y_test, feedback_network, epochs, 0.3)
exit(0)

execute_network_feedback(x_train, x_test, y_train, y_test, feedback_network, epochs, 0.3)


for model in layers:

<<<<<<< HEAD

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


# def execute_random_forest(dataset, notes):

#     x_train, x_test, y_train, y_test = process_dataset_nn(
#         dataset, testsplit=testsplit)

#     num_features = x_train.shape[1]
#     num_targets = y_train.shape[1]

#     forest = RandomForest_featureimportance()

#     forest.train(x_train, y_train)

#     evaluation = forest.test(x_test, y_test)

#     print('Test evaluation on random forest: {}'.format(evaluation))


def execute_conv_network(dataset, note, write_log=False):

    # Dette ble kjempe stykt, bare å si fra om noen kan en god løsning på dette
    x_prod_train, x_rest_train, x_prod_test, x_rest_test, y_train, y_test = process_dataset_conv_nn(
        dataset, production_col_name='Produksjon')

    history_length = np.shape(x_prod_train)[1]
    rest_input_dim = x_rest_train.shape[1]

    network = Conv_NN(epochs=epochs, batch_size=batch_size,
                      model_path=model_path,)

    model_architecture = network.build_model(history_length, rest_input_dim)
    model_architecture.summary()

    model = network.train_network(x_prod_train, x_rest_train, y_train, opt=opt)

    evaluation, metric_names = network.evaluate(
        x_prod_test, x_rest_test, y_test)
    print(evaluation)

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

# evaluation = execute_network_simple(
#     dataset_bess, 'Simple network, testing on last month', epochs, write_log=True)

# evaluation = execute_network_simple(
#     dataset_vals, 'Simple network, testing on last month', epochs, write_log=True)

# visualize_training_buckets('large test buckets.hdf5')


data_buckets = [2000,
                4000, 6000, 10000, 16000, 20000, 24000, 28000, dataset_vals.shape[0]]
evaluation_list = []
for i, bucket in enumerate(data_buckets):
    subdataset = dataset_vals[0:bucket]

    print(subdataset.shape[0])

    evaluation = execute_network_simple(
        subdataset, 'Training simple network with {} samples'.format(subdataset.shape[0]), epochs, write_log=True)

    evaluation_list.append(evaluation[0])

with h5py.File('slim network, last attempt.hdf5', 'w') as f:
    print('Saving all model evaluations in h5 file')
    f.create_dataset('buckets', data=data_buckets)
    f.create_dataset('evaluations', data=evaluation_list)

evaluation = execute_network_simple(
    dataset_bess, 'Simple - bess (month)', epochs, write_log=True)
# execute_network_advanced(
#     dataset_bess, 'Advanced - bess (month)', best_network, epochs, write_log=True)
evaluation = execute_network_simple(
    dataset_vals, 'Simple - vals (month)', epochs, write_log=True)
# execute_network_advanced(
#     dataset_vals, 'Advanced - vals (month)', best_network, epochs, write_log=True)
exit(0)

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


# for i, opt in enumerate(opt):
#     for layer_set in layers:
#         execute_network(x_train, x_test, y_train, y_test, batch_size, layer_set, epochs, opt, opt_name[i])

# model = RandomForest()
# model.train(x_train, y_train)
# print(model.predict(x_test, y_test))
