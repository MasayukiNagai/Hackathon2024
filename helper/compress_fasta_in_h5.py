import os
import IOHelper

def prepare_input(datadir, fold, split):
    print(f'Retriving fold={fold}, split={split}')
    file_seq = os.path.join(datadir, f'fold{fold}_sequences_{split}.fa.gz')
    input_sequence_data = IOHelper.get_fastas_from_file(file_seq, uppercase=True)
    input_sequence_onehot = IOHelper.convert_one_hot(input_sequence_data)
    print(f'Shape of the sequence one-hot: {input_sequence_onehot.shape}')
    # X = np.nan_to_num(input_sequence_onehot)

    file_activity = os.path.join(datadir, f'fold{fold}_sequences_activity_{split}.txt.gz')
    input_activity_data = IOHelper.get_activity_from_file(file_activity)
    print(f'Shape of the activity data: {input_activity_data.shape}')

    return input_sequence_onehot, input_activity_data


def main(datadir):
    folds = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    for fold in folds:
        X_train, Y_train = prepare_input(datadir, fold, 'Train')
        X_val, Y_val = prepare_input(datadir, fold, 'Val')
        X_test, Y_test = prepare_input(datadir, fold, 'Test')
        filepath = os.path.join(datadir, f'fold{fold}_onehot.h5')
        IOHelper.save_onehot_in_h5(filepath, X_train, Y_train, X_val, Y_val, X_test, Y_test, comp_level=4)


if __name__ == '__main__':
    datadir = '/home/nagai/projects/Hackathon2024/data/Accessibility_models_training_data/'
    main(datadir)
