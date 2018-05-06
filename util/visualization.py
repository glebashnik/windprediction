from matplotlib import pyplot


# Visualizes training loss history of a trained model, the h5 file is 
# stored in the 'result_log' folder
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
    evaluations = history['evaluations'].value

    plt.plot(buckets[3:], evaluations[3:], 'bo')
    plt.title('Network training loss for different dataset sizes, 2500 Epochs')
    plt.ylabel('MAE')
    plt.xlabel('number of samples')
    # plt.legend(metrics, loc='upper right')
    # if (start != None) and (end != None):
    #     plt.xlim(start, end)
    plt.show()



if __name__ == '__main__':
    print('Visualization script')
    try:
        network = sys.argv[1]
    except IndexError:
        print('no network given')
        network = 'M04-D16_h18-m03-s37'

    # visualize_loss_history(network)
    # visualize_training_buckets(os.path.join('..','training_data_buckets.hdf5'))
