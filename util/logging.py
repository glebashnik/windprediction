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

<<<<<<< HEAD
    # Save model to folder
    copyfile('checkpoint_model.h5', os.path.join(
        basename, 'checkpoint_model.h5'))

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
=======
    file.write('\n')
    file.close()
>>>>>>> 6277f428e59c037e60730473841be17577ffdd0a
