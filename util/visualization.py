<<<<<<< HEAD
from matplotlib import pyplot
=======
import os
import sys
# import matplotlib.pyplot as plt
# import h5py
# import numpy as np
>>>>>>> cd087be07de353aae2bc8a7580e1149ab0e04a02

<<<<<<< HEAD

# def compare_predictions(self, x, y):
#     lines = pyplot.plot(x, 'r', y, 'b')
#     pyplot.setp(lines, linewidth=0.5)
#     pyplot.show()


def visualize_loss_history(network_sample, start=None, end=None):

    history = h5py.File(os.path.join('result_log',
                                     network_sample, 'results_'+network_sample+'.h5'), 'r')

    metrics = list(history.keys())

    loss = (history[metrics[0]].value)
    val_loss = (history[metrics[1]].value)
    # gen_logloss = (history[metrics[2]].value)
    # gen_loss = (history[metrics[3]].value)

    # ax = plt.figure(figsize=(20, 15))

    plt.plot(history[metrics[0]].value)
    plt.plot(history[metrics[1]].value)
    # plt.plot(history[metrics[2]].value)
    # plt.plot(history[metrics[3]].value)
    plt.title('Network training loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(metrics, loc='upper right')
    if (start != None) and (end != None):
        plt.xlim(start, end)
    plt.show()


def visualize_training_buckets(file_path):

    history = h5py.File(file_path, 'r')

    buckets = history['buckets'].value
    evaluations = history['buckets'].value

    plt.plot(buckets, evaluations, 'bo')
    plt.title('Network training loss for different dataset sizes')
    plt.ylabel('MAE')
    plt.xlabel('Dataset sizes')
    # plt.legend(metrics, loc='upper right')
    # if (start != None) and (end != None):
    #     plt.xlim(start, end)
    plt.show()


if __name__ == '__main__':
    print('Visualization script')
    exit(0)
    try:
        network = sys.argv[1]
    except IndexError:
        print('no network given')
        network = 'M04-D16_h18-m03-s37'

    # visualize_loss_history(network)
    # visualize_training_buckets(os.path.join('..','training_data_buckets.hdf5'))
=======
def compare_predictions(self, x, y):
    lines = pyplot.plot(x, 'r', y, 'b')
    pyplot.setp(lines, linewidth=0.5)
    pyplot.show()
>>>>>>> 6277f428e59c037e60730473841be17577ffdd0a
