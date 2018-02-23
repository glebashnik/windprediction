# Writes results to log
def write_results(file,layers,results,metrics,ahed,back, epochs, optimizer):
    file.write('\nLayers:\n')

    for i,item in enumerate(layers):
        file.write("LSTM:{} ".format(item))

    file.write('\nLookback: {} Lookahed: {}'.format(ahed,back))

    file.write('\nOptimizer: ' + optimizer)

    file.write('\nTrained {} epochs\n'.format(epochs))
    
    for i,item in enumerate(metrics):
        file.write(item)
    file.write('\n')

    for i,item in enumerate(results):
        file.write(" {} ".format(item))

    file.write('\n')