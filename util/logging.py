# Writes results to log
def write_results(filename, layers, results, metrics, look_ahead, look_back, epochs, optimizer):
    file = open(filename, 'a')
    
    file.write('\nLayers:\n')

    # for i,item in enumerate(layers):
    #     file.write(":{} ".format(item))

    file.write('\nLook-ahead: {} Look-back: {}'.format(look_ahead, look_back))

    file.write('\nOptimizer: ' + optimizer)

    file.write('\nDropoutrate: {}'.format(dropoutrate))

    file.write('\nTrained {} epochs\n'.format(epochs))
    
    for i,item in enumerate(metrics):
        file.write(item + ', ')
    file.write('\n')

    for i,item in enumerate(results):
        file.write("{}".format(item) + ', ')

    file.write('\n')
    file.close()