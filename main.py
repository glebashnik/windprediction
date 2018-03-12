import os
import pandas as pd
import matplotlib.pyplot as plt

from util.processing import process_dataset_lstm, process_dataset_nn
from util.visualization import compare_predictions
from util.logging import write_results

from models.simple_lstm.main import RNN as LSTM
from models.lstm_stateful.main import RNN as StatefulLSTM
from models.simple_ann.main import NN
from models.dense_nn_forest.NN_forest import NN_forest
from models.ann_error_feedback.ann_feedback import NN_feedback

from keras import optimizers
from models.random_forest.main import RandomForest

datapath = os.path.join('data','Ytre Vikna', 'data_ytrevikna_advanced.csv')
# datapath = os.path.join('data','Skomakerfjellet', 'data_skomakerfjellet_advanced.csv')
# datapath = os.path.join('data','Bessaker', 'data_bessaker_advanced.csv')


# datapath = os.path.join('data','Skomakerfjellet', 'pred-compare.csv')
modelpath = os.path.join('checkpoint_model.h5')

try:
    dataset = pd.read_csv(datapath, sep=';')
except:
    print('No data found on: ' + datapath)
    exit(1)



num_features = len(dataset.columns) -1
x_train, x_test, y_train, y_test = process_dataset_nn(dataset, testsplit=0.7)

print('Beginning model training on the path:')
print(datapath)
print('Number of features: {}\n\n'.format(num_features))

#Hyperparameters for training network
logfile = open('results.txt','w')

testsplit = 0.8
look_back = 6
look_ahead = 1
epochs = 5000
batch_size= 64
lr = 0.001
decay = 1e-6
momentum=0.9


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

layers =[
    [[(32,False),(16,True),(8,False),(2,False)],
    [(128,False),(64,True),(32,False),(8,False)],
    [(64,False),(16,True),(6,False),(0,False)],
    [(16,True),(8,False),(8,False),(0,False)]],

    [[(32,False),(16,True),(8,False),(2,False)],
    [(64,False),(64,True),(32,False),(16,False)],
    [(12,False),(6,True),(3,False),(0,False)],
    [(69,False),(128,True),(32,False),(4,False)]],
    ]

feedback_network = [(32,False),(16,True),(8,False),(2,False)]

dropouts = [0.2, 0.3, 0.4, 0.5]

def execute_network(x_train, x_test, y_train, y_test, layers, epochs, dropoutrate, opt = 'adam', optname='adam'):
    nn_network = NN_forest(batch_size=32, epochs=epochs, dropoutrate=dropoutrate)

    nn_network.build_model(input_dim=num_features,model_structure=layers)

    # nn_network.visualize_model()
    nn_network.train_network(x_train=x_train, y_train=y_train,opt=opt)
    print('training finished')
    
    evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, epochs, optname, dropoutrate)


def execute_network_NN_feedback(x_train, x_test, y_train, y_test, layers, epochs, dropoutrate, opt = 'adam', optname='adam'):
    network = NN_feedback(batch_size=32, epochs=epochs, dropoutrate=dropoutrate)

    network.build_model(input_dim=num_features,model_structure=layers)
    # network.visualize_model()
    network.train_network(x_train=x_train, y_train=y_train,opt=opt)
    print('training finished')
    
    evaluation, metric_names = network.evaluate(modelpath, x_test, y_test)
    write_results(logfile, layers, evaluation, metric_names, epochs, optname, dropoutrate)


execute_network_NN_feedback(x_train, x_test, y_train, y_test, feedback_network, epochs, 0.3)
exit(0)


for model in layers:

    for dropoutrate in dropouts:

        execute_network(x_train, x_test, y_train, y_test, model, epochs, dropoutrate)

        print('Network executed')


# for i, opt in enumerate(opt):
#     for layer_set in layers:
#         execute_network(x_train, x_test, y_train, y_test, batch_size, layer_set, epochs, opt, opt_name[i])

# model = RandomForest()
# model.train(x_train, y_train)
# print(model.predict(x_test, y_test))