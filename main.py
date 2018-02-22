import os
import pandas as pd

import matplotlib.pyplot as plt
from util.processing import process_dataset, generate_test_set, make_time_series, write_results
from util.visualization import compare_predictions
from models.simple_lstm_2.main import RNN

from keras import optimizers

datapath = os.path.join('data','Ytre Vikna', 'data_ytrevikna_advanced.csv')
modelpath = os.path.join('checkpoint_model.h5')

try:
    dataset = pd.read_csv(datapath, index_col=0, sep=';')
except:
    print('No data found on: ' + datapath)
    exit(1)

x_train, x_test, y_train, y_test = process_dataset(dataset, look_back=4, look_ahead=1, testsplit=0.8)

# nn_network = RNN(batch_size=32, epochs=100)



# nn_network.build_model((x_train.shape[1], x_train.shape[2]))
# nn_network.train_network(x_train, y_train)
# evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)

logfile = open('results.txt','w')
testsplit = 0.8
look_back = 4
look_ahead = 1
epochs = 200
batch_size= 32
lr = 0.001
decay = 1e-6
momentum=0.9

opt = [optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True),
optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)]

opt_name = ['SGD lr: {} decay: {} momentum: {} '.format(lr,decay,momentum),
'Adam lr: {} decay: {} '.format(lr,momentum)]

layers = [[32,16],[64,32],[16,16],[20,20],[128,64]]


def execute_network(x_train, x_test, y_train, y_test, layers, epochs, opt, optname):

    nn_network = RNN(batch_size=32, epochs=epochs)
    nn_network.build_model_general((x_train.shape[1], x_train.shape[2]),layers)
    nn_network.train_network(x_train, y_train,0.01,opt)
    evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)
    write_results(logfile,layers,evaluation,metric_names,look_back,look_ahead, epochs, optname)




for i,opt in enumerate(opt):
    for j in range(len(layers)):
        
        execute_network(x_train, x_test, y_train, y_test, layers[j],epochs, opt, opt_name[i])

logfile.close()