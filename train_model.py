import os
import sys
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


def write_summary(outpath, all_scores):
    metrics = ['MSE', 'PCC', 'SCC']
    with open(outpath, 'w') as f:
        items = ['fold', 'train_MSE', 'train_PCC', 'train_SCC',
                 'valid_MSE', 'valid_PCC', 'valid_SCC',
                 'test_MSE', 'test_PCC', 'test_SCC']
        header = '\t'.join(items)
        f.write(header + '\n')
        for scores in all_scores:
            items = [scores['fold']]
            for split in ['train', 'valid', 'test']:
                items += [f'{scores[split][m]:.5f}' for m in metrics]
            line = '\t'.join(items)
            f.write(line + '\n')


def main(datadir, params, outdir, celltype=None):
    folds = ['fold01', 'fold02', 'fold03', 'fold04', 'fold05',
             'fold06', 'fold07', 'fold08', 'fold09', 'fold10']
    all_celltypes = [
        'amnioserosa', 'brain', 'CNS', 'epidermis', 'fatbody',
        'glia', 'hemocytes', 'malpighiantube', 'midgut', 'pharynx',
        'plasmatocytes', 'PNS', 'salivarygland', 'somaticmuscle', 'trachea',
        'unknown', 'ventralmidline', 'visceralmuscle']
    if celltype:
        assert celltype in all_celltypes, f'Invalid celltype: {celltype}'
        idx = all_celltypes.index(celltype)
    num_outputs = 1 if celltype else len(all_celltypes)

    all_scores = []
    for fold in folds:
        print(f'########## {fold}, celltype={celltype} ##########')

        # Initialize a NN model
        my_model = NormalNN(params, num_outputs)

        # Load data from the input h5 file
        h5path = os.path.join(datadir, f'{fold}_onehot.h5')
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = \
            IOHelper.load_onehot_from_h5(h5path)

        # If predicting a single cell type, take out the relevant part
        if celltype:
            Y_train = Y_train[:, idx]
            Y_valid = Y_valid[:, idx]
            Y_test = Y_test[:, idx]

        # Train the model
        my_history = my_model.train_model(
            X_train, Y_train, X_valid, Y_valid,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            early_stop=params['early_stop'])

        # Evaluate the model performance on the three data splits
        train_scores = my_model.evaluate_model(X_train, Y_train, 'train')
        valid_scores = my_model.evaluate_model(X_valid, Y_valid, 'validation')
        test_scores = my_model.evaluate_model(X_test, Y_test, 'test')
        test_scores['fold'] = fold
        scores = {
            'fold': fold,
            'train': train_scores,
            'valid': valid_scores,
            'test': test_scores}
        all_scores.append(scores)

        # Save the trained model
        my_model.save_model(f'model_{fold}', outdir, 'h5')
        K.clear_session()

    summary_path = os.path.join(outdir, 'summary.txt')
    write_summary(summary_path, all_scores)


if __name__ == '__main__':
    # args = argument_parser()
    # datadir = args.data
    # paramfile = args.param
    # outdir = args.outdir
    # gpu = args.gpu
    # params = IOHelper.parse_conf(paramfile)

    datadir = '/home/nagai/projects/Hackathon2024/data/Accessibility_models_training_data_h5/'
    outdir = '/home/nagai/projects/Hackathon2024/data/models/'
    celltype = None  # 'CNS'

    if not os.path.exists(datadir):
        sys.exit(f'Error: the specified path does not exist: {datadir}')
    IOHelper.makedir(outdir)

    params = {
        'batch_size': 128,
        'epochs': 20,
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

    main(datadir, params, outdir, celltype)
