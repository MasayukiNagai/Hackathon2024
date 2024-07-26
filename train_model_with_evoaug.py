import os
import sys
import argparse
import gc

import numpy as np

import keras
from keras import backend as K
import tensorflow as tf

from model_architectures import NormalNN, MHA_NN, EvoAug_NN
from helper import IOHelper

# import evoaug_tf
# from evoaug_tf import evoaug, augment


def argument_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--input', '-i', type=str, metavar='INPUT', required=True,
        dest='input', help='Required; Specify an input file path.')
    arg_parser.add_argument(
        '--param', '-p', type=str, metavar='PARAM', required=True,
        dest='param', help='Required; Specify a param config file.')
    arg_parser.add_argument(
        '--outdir', '-o', type=str, metavar='OUTDIR', required=False,
        dest='outdir', default='./results',
        help='Optional; Specify an output directory. (default: ./results)')
    args = arg_parser.parse_args()
    return args


def write_summary(outpath, scores):
    metrics = ['MSE', 'PCC', 'SCC']
    with open(outpath, 'w') as f:
        items = ['train_MSE', 'train_PCC', 'train_SCC',
                 'valid_MSE', 'valid_PCC', 'valid_SCC']
        header = '\t'.join(items)
        f.write(header + '\n')
        items = []
        for split in ['train', 'valid']:
            items += [f'{scores[split][m]:.5f}' for m in metrics]
        line = '\t'.join(items)
        f.write(line + '\n')


def main(h5path, params, outdir):
    K.clear_session()
    gc.collect()

    # Load data from the input h5 file
    X_train, Y_train, X_valid, Y_valid = \
        IOHelper.load_onehot_from_h5(h5path)

    _, num_inputs, _ = X_train.shape
    _, num_outputs = Y_train.shape

    # num_inputs = 1001
    # num_outputs = 18

    # Initialize a NN model
    # my_model = NormalNN(params, num_inputs, num_outputs)
    # my_model = MHA_NN(params, num_inputs, num_outputs)
    my_model = EvoAug_NN(params, num_inputs, num_outputs).model
    # print(my_model.model.summary())

    history = my_model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(X_valid, Y_valid))


    # Train the model
    my_history = my_model.train_model(
        X_train, Y_train, X_valid, Y_valid,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        early_stop=params['early_stop'])

    # Evaluate the model performance on the three data splits
    train_scores = my_model.evaluate_model(X_train, Y_train, 'train')
    valid_scores = my_model.evaluate_model(X_valid, Y_valid, 'validation')
    scores = {
        'train': train_scores,
        'valid': valid_scores}

    # Save the trained model
    my_model.save_model(f'model', outdir, 'h5')

    summary_path = os.path.join(outdir, 'summary.txt')
    write_summary(summary_path, scores)


if __name__ == '__main__':
    args = argument_parser()
    h5path = args.input
    paramfile = args.param
    outdir = args.outdir
    params = IOHelper.parse_conf(paramfile)
    print(params)

    # h5path = '/shared/hackathon/dataset1.h5'
    # outdir = './results'

    if not os.path.exists(h5path):
        sys.exit(f'Error: the specified path does not exist: {h5path}')
    IOHelper.makedir(outdir)

    print(f'Using gpu = {os.environ["CUDA_VISIBLE_DEVICES"]}')
    main(h5path, params, outdir)




history = model.fit(x_train, y_train,
                    epochs=100,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_valid, y_valid),
                    callbacks=[es_callback, reduce_lr])

exp_name = 'resbind_fly'
save_path = os.path.join(output_dir, exp_name+"_evoaug.h5")
model.save_weights(save_path)
