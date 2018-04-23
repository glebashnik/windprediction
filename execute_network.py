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

def execute_conv_network(dataset, note, write_log=False):

    # Dette ble kjempe stykt, bare å si fra om noen kan en god løsning på dette
    x_prod_train, x_rest_train, x_prod_test, x_rest_test, y_train, y_test = process_dataset_conv_nn(dataset, production_col_name='Produksjon')
    
    history_length = np.shape(x_prod_train)[1]
    rest_input_dim = x_rest_train.shape[1]
    num_features = rest_input_dim+history_length


    network = Conv_NN(epochs=epochs, batch_size=batch_size, model_path=model_path,)

    model_architecture = network.build_model(history_length, rest_input_dim)
    model_architecture.summary()

    hist_loss, model = network.train_network(x_prod_train, x_rest_train, y_train, opt=opt)

    evaluation, metric_names = network.evaluate(
        x_prod_test, x_rest_test, y_test)
    
    if write_log:
        dropoutrate = 0 
        write_results(park, model_architecture, note, num_features,
                      hist_loss, evaluation, metric_names, epochs, opt, dropoutrate)
