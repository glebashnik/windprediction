# Writes results to log
<<<<<<< HEAD
def write_results(file,layers,results,metrics, epochs=0, optimizer='adam', dropoutrate= 0, ahed=None, back=None):
    file.write('\nNN_forest:\n')
=======
def write_results(filename, layers, results, metrics, look_ahead, look_back, epochs, optimizer):
    file = open(filename, 'a')
    
    file.write('\nLayers:\n')
>>>>>>> f883e113d852bdc3773595162bb1c816fbdb0e4a

    # for i,item in enumerate(layers):
    #     file.write(":{} ".format(item))

<<<<<<< HEAD
    if (ahed != None) and (back != None): file.write('\nLookback: {} Lookahed: {}'.format(ahed,back))
=======
    file.write('\nLook-ahead: {} Look-back: {}'.format(look_ahead, look_back))
>>>>>>> f883e113d852bdc3773595162bb1c816fbdb0e4a

    file.write('\nOptimizer: ' + optimizer)

    file.write('\nDropoutrate: {}'.format(dropoutrate))

    file.write('\nTrained {} epochs\n'.format(epochs))
    
    for i,item in enumerate(metrics):
<<<<<<< HEAD
        file.write('  ' + item)
=======
        file.write(item + ', ')
>>>>>>> f883e113d852bdc3773595162bb1c816fbdb0e4a
    file.write('\n')

    for i,item in enumerate(results):
        file.write("{}".format(item) + ', ')

    file.write('\n')
    file.close()