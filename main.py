import os
import pandas as pd

import matplotlib.pyplot as plt
from util.processing import process_dataset, generate_test_set, make_time_series
from util.visualization import compare_predictions
from models.simple_lstm_2.main import RNN

datapath = os.path.join('data', 'Data_advanced.csv')
modelpath = os.path.join('checkpoint_model.h5')
datapath_test = os.path.join('data', 'test.csv')

try:
    dataset_train = pd.read_csv(datapath, index_col=0, sep=';')
    #dataset_test = pd.read_csv(datapath_test, index_col=0, sep=';')
except:
    print('No data found on: ' + datapath)
    exit(1)

x_train, x_test, y_train, y_test = process_dataset(dataset_train, look_back=4, look_ahead=0, testsplit=0.8)

nn_network = RNN(batch_size=1, epochs=100)
nn_network.build_model((x_train.shape[1], x_train.shape[2]))
nn_network.train_network(x_train, y_train)
evaluation, metric_names = nn_network.evaluate(modelpath, x_test, y_test)

print(metric_names)
print(evaluation)

#model_path = os.path.join('checkpoint_model.h5')

#x_test = dataset_test.iloc[:, :-1]
#y_test = dataset_test['Target'].reset_index(drop=True)

#print(x_test)

#x_test = process_dataset(x_test, look_back=4, look_ahead=0)

#print(x_test)
#print(y_test)

# for i, sample in enumerate(x_test):
#     sample = sample.reshape(1, sample.shape[0], sample.shape[1])
#     prediction = nn_network.predict(model_path, sample)
#     print("Prediction: ", prediction, " -- Actual: ", y_test.iloc[i])

#predictions = nn_network.predict(model_path, x_test)
# plt.plot(predictions, 'r', y_test, 'b')
# plt.show()


#pd.DataFrame(predictions).to_csv('predictions.csv', sep=';')
