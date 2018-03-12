import os
import pandas as pd
import matplotlib.pyplot as plt

from util.processing import process_dataset_lstm, process_dataset_nn
from util.visualization import compare_predictions
from util.logging import write_results

from models.simple_lstm_2.main import RNN as LSTM
from models.lstm_stateful.main import RNN as StatefulLSTM
from models.simple_ann_eirik.main import NN
from models.random_forest.main import RandomForest

from keras import optimizers

datapath = os.path.join('data','Skomakerfjellet', 'pred-compare.csv')
modelpath = os.path.join('checkpoint_model.h5')

try:
    dataset = pd.read_csv(datapath, sep=';')
except:
    print('No data found on: ' + datapath)
    exit(1)

logfile = 'results.txt'
testsplit = 0.8
look_back = 6
look_ahead = 1
epochs = 1000
batch_size= 32
lr = 0.001
decay = 1e-6
momentum=0.9

#LSTM
# x_train, x_test, y_train, y_test = process_dataset_lstm(
#     dataset, 
#     look_back=look_back, 
#     look_ahead=look_ahead, 
#     testsplit=testsplit, 
#     batch_size=batch_size,
#     stateful=False
# )

#Dense 
x_train, x_test, y_train, y_test = process_dataset_nn(
    dataset, 
    testsplit=testsplit
)

opt = [
    #optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True),
    #optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),
    #'rmsprop',
    'adam'
]

opt_name = [
    #'SGD lr: {} decay: {} momentum: {} '.format(lr,decay,momentum),
    #'Adam lr: {} decay: {} '.format(lr,momentum),
    #'rmsprop',
    'adam'
]

#layers = [[128, 16], [128, 16], [128, 16], [128, 16], [128, 16]] 
#layers = [[64, 32, 16, 8], [64, 32, 16, 8], [32, 16, 8], [32, 16, 8]]
layers = [[128, 32], [128, 32], [128, 32], [128, 32]]
#layers = [[32, 16], [32, 16], [64, 32], [64, 32]]


def execute_network(x_train, x_test, y_train, y_test, batch_size, layers, epochs, opt, optname):
    #LSTM
    # nn_network = LSTM(batch_size=batch_size, epochs=epochs)
    # nn_network.build_model_general((x_train.shape[1], x_train.shape[2]), layers)

    #Stateful LSTM
    # nn_network = StatefulLSTM(batch_size=batch_size, epochs=epochs)
    # nn_network.build_model_general((x_train.shape[1], x_train.shape[2]), layers)
    
    #Dense NN
    nn_network = NN(batch_size=batch_size, epochs=epochs)
    nn_network.build_model_general(x_train.shape[1], layers)
    
    nn_network.train_network(x_train, y_train, opt=opt)
    evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, look_back, look_ahead, epochs, optname)

for i, opt in enumerate(opt):
    for layer_set in layers:
        execute_network(x_train, x_test, y_train, y_test, batch_size, layer_set, epochs, opt, opt_name[i])

# model = RandomForest()
# model.train(x_train, y_train)
# print(model.predict(x_test, y_test))