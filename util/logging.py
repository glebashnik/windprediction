# Writes results to log
def write_results(file,layers,results,metrics, epochs=0, optimizer='adam', dropoutrate= 0, ahed=None, back=None):
    file.write('\nNN_forest:\n')

    # for i,item in enumerate(layers):
    #     file.write(":{} ".format(item))

    if (ahed != None) and (back != None): file.write('\nLookback: {} Lookahed: {}'.format(ahed,back))

    file.write('\nOptimizer: ' + optimizer)

    file.write('\nDropoutrate: {}'.format(dropoutrate))

    file.write('\nTrained {} epochs\n'.format(epochs))
    
    for i,item in enumerate(metrics):
        file.write('  ' + item)
    file.write('\n')

    for i,item in enumerate(results):
        file.write("{}".format(item) + ', ')

    file.write('\n')
    file.close()