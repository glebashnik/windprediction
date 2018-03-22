import matplotlib.pyplot as plt
import h5py
import os


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


if __name__ == '__main__':

    try:
        network = sys.argv[1]
    except IndexError:
        print('no network given')
        network = 'M03-D21_h19-m51-s15'

    visualize_loss_history(network)
