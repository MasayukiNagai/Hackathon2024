import os
import argparse

from keras import backend as K
import tensorflow as tf

from model_architectures import SimpleNN, NormalNN
from helper import IOHelper


def argument_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--input', '-i', type=str, metavar='INPUT', required=True,
        dest='data', help='Required; Specify an input folder path.')
    arg_parser.add_argument(
        '--param', '-p', type=str, metavar='PARAM', required=True,
        dest='param', help='Required; Specify a param config file.')
    arg_parser.add_argument(
        '--outdir', '-o', type=str, metavar='OUTDIR', required=False,
        dest='outdir', default='./results',
        help='Optional; Specify an output directory. (default: ./results)')
    arg_parser.add_argument(
        '--gpu', '-gpu', type=str, metavar='GPU', required=False, dest='gpu',
        help='Optional; Specify a gpu device by its index (e.g., "0").')
    args = arg_parser.parse_args()
    return args


def main(h5path, params, outdir):
    # Initialize a NN model
    my_model = NormalNN(params)

    # Load data from the input h5 file
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
        IOHelper.load_onehot_from_h5(h5path)

    # Train the model
    my_history = my_model.train_model(
        X_train, Y_train, X_valid, Y_valid,
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        early_stop=params['early_stop'])

    # Evaluate the model performance on the three data splits
    my_model.evaluate_model(X_train, Y_train, 'train')
    my_model.evaluate_model(X_valid, Y_valid, 'validation')
    my_model.evaluate_model(X_test, Y_test, 'test')

    # Save the trained model
    my_model.save_model('trained_model', outdir, 'h5')


if __name__ == '__main__':
    # args = argument_parser()
    # h5path = args.infile
    # params = args.paramfile
    # outdir = args.outdir
    # os.exists
    # os.makedirs

    h5path = '/home/nagai/projects/Hackathon2024/data/Accessibility_models_training_data_h5/fold01_onehot.h5'
    outdir = '/home/nagai/projects/Hackathon2024/data/models/'

    params = {
        'batch_size': 128,
        'epochs': 1,
        'early_stop': 5,
        'lr': 0.005,
        'padding':'same',

        'num_conv_layers': 4,

        'num_filters1': 256,
        'kernel_size1': 7,
        'activation1': 'relu',
        'max_pool1': 3,

        'num_filters2': 120,
        'kernel_size2': 3,
        'activation2': 'relu',
        'max_pool2': 3,

        'num_filters3': 60,
        'kernel_size3': 3,
        'activation3': 'relu',
        'max_pool3': 3,

        'num_filters4': 60,
        'kernel_size4': 3,
        'activation4': 'relu',
        'max_pool4': 3,

        'num_dense_layers': 2,

        'dense_neurons5': 64,
        'activation5': 'relu',
        'dropout_prob5': 0.4,

        'dense_neurons6': 256,
        'activation6': 'relu',
        'dropout_prob6': 0.4,}

    print(f'Using gpu = {os.environ["CUDA_VISIBLE_DEVICES"]}')
    # print(tf.config.list_physical_devices("GPU"))
    # print(tf.config.experimental.list_physical_devices('GPU'))

    main(h5path, params, outdir)
