import time
import datetime
import os
import h5py
from shutil import copyfile
# Writes results to log


def write_results(park, model_arch, note, num_features, hist_loss, results, metrics, epochs, optimizer='adam', dropoutrate=0, ahed=None, back=None):

    # Initialize folders and paths
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('M%m-D%d_h%H-m%M-s%S')

    basename = os.path.join('result_log', st)

    if not os.path.isdir(basename):
        os.mkdir(basename)

    log_name = os.path.join(basename, 'results_{}.txt'.format(st))
    log_name_h5 = os.path.join(basename, 'results_{}.h5'.format(st))
    logfile = open(log_name, 'w')

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

    logfile.write('\nOptimizer: ' + optimizer[0])

    logfile.write('\nDropoutrate: {}'.format(dropoutrate))

    logfile.write('\nTrained {} epochs\n'.format(epochs))

    logfile.write('{} test evaluation\n'.format(metrics[-1]))

    logfile.write('{}\n'.format(results))

    logfile.write('{} training loss\n'.format(metrics[-1]))

    if not hist_loss == None:
        logfile.write('{}'.format(hist_loss['loss'][-1]))

    logfile.write('\n')
    logfile.close()
