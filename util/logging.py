# When a model is trained the results and specs of the network is saved in a txt file, located
# in the 'result_log' folder


def write_results(file,layers,results,metrics, epochs=0, optimizer='adam', dropoutrate= 0, ahed=None, back=None):
    file.write('\nNN_forest:\n')

    if (ahed != None) and (back != None): file.write('\nLookback: {} Lookahed: {}'.format(ahed,back))

    file.write('\nOptimizer: ' + optimizer)

    file.write('\nDropoutrate: {}'.format(dropoutrate))

    file.write('\nTrained {} epochs\n'.format(epochs))
    
    for i,item in enumerate(metrics):
        file.write('  ' + item)
    file.write('\n')

    for i,item in enumerate(results):
        file.write("{}".format(item) + ', ')

    # Save loss history as h5 file
    if not hist_loss == None:
        with h5py.File(log_name_h5, 'w') as hf:
            hf.create_dataset('mae_loss', data=hist_loss['loss'])
            hf.create_dataset('mae_val_loss', data=hist_loss['val_loss'])

    # Save hyperparameter info (and what not) in txt file
    print()
    if model_arch != None:
        model_arch.summary(print_fn=lambda x: logfile.write(x + '\n'))

    logfile.write('\n\n' + park + '\n')

    logfile.write(note)

    logfile.write('\nFeatures: {}'.format(num_features))

    if (ahed != None) and (back != None):
        logfile.write('\nLookback: {} Lookahed: {}'.format(ahed, back))

    logfile.write('\nOptimizer: ' + optimizer)

    logfile.write('\nDropoutrate: {}'.format(dropoutrate))

    logfile.write('\nTrained {} epochs\n'.format(epochs))

    logfile.write('{} test evaluation\n'.format(metrics[-1]))

    logfile.write('{}\n'.format(results))

    logfile.write('{} training loss\n'.format(metrics[-1]))

    if not hist_loss == None:
        logfile.write('{}'.format(hist_loss['loss'][-1]))

    logfile.write('\n')
    logfile.close()
