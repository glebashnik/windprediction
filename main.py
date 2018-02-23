import os
import pandas as pd

import matplotlib.pyplot as plt
from util.processing import process_dataset_lstm_stateful, process_dataset_lstm, process_dataset_nn
from util.visualization import compare_predictions

from models.simple_lstm_2.main import RNN as LSTM
from models.lstm_stateful.main import RNN as StatefulLSTM
from models.simple_ann_eirik.main import NN

datapath = os.path.join('data', 'data_yttrevikna_advanced.csv')
modelpath = os.path.join('checkpoint_model.h5')

try:
    dataset = pd.read_csv(datapath, index_col=0, sep=';')
except:
    print('No data found on: ' + datapath)
    exit(1)

x_train, x_test, y_train, y_test = process_dataset_lstm(dataset, look_back=6, look_ahead=1, testsplit=0.8)
#x_train, x_test, y_train, y_test = process_dataset_nn(dataset, testsplit=0.8)

nn_network = LSTM(batch_size=32, epochs=200)
#nn_network = NN(batch_size=32, epochs=10000)

nn_network.build_model((x_train.shape[1], x_train.shape[2]))
#nn_network.build_model(x_train.shape[1])

nn_network.train_network(x_train, y_train)
evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)

print(metric_names)
print(evaluation)