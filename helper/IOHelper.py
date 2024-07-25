import os
import sys
import gzip
import configparser

import numpy as np
import h5py


def get_fastas_from_file(fasta_path, uppercase=False):
    """
    Reads a FASTA file and returns the sequences as a NumPy array.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file.
    uppercase : bool, optional
        If True, converts sequences to uppercase. Default is False.

    Returns
    -------
    np.ndarray
        Array of sequences from the FASTA file.
    """
    fastas = []
    # headers = []
    seq = None
    header = None
    file_opener = gzip.open if fasta_path.endswith('.gz') else open
    with file_opener(fasta_path, 'rt', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith(">"):
                if seq is not None and header is not None:
                    fastas.append(seq)
                    # headers.append(header)
                seq = ""
                header = line[1:]
            else:
                if seq is not None:
                    seq += line.upper() if uppercase else line
                else:
                    seq = line.upper() if uppercase else line
    fastas.append(seq)
    # headers.append(header)

    return np.array(fastas)


def get_activity_from_file(activity_path):
    """
    Reads a TSV file and returns the data as a NumPy array.

    Parameters
    ----------
    activity_path : str
        Path to the TSV file.

    Returns
    -------
    np.ndarray
        Array of data from the TSV file, skipping the header.
    """
    file_opener = gzip.open if activity_path.endswith(".gz") else open
    with file_opener(activity_path, 'rt') as file:
        data = np.genfromtxt(file, delimiter='\t', skip_header=1)

    return data


def convert_one_hot(sequences, alphabet="ACGT", uncertain_N=True):
    """
    Convert a flat array of sequences to one-hot representation.
    **Important**: all letters in `sequences` *must* be contained in `alphabet`,
    and all sequences must have the same length.

    Parameters
    ----------
    sequences : numpy.ndarray of strings
        The array of strings. Should be one-dimensional.
    alphabet : str
        The alphabet of the sequences.

    Returns
    -------
    Numpy array of sequences in one-hot representation. The shape of this array is
    `(len(sequences), len(sequences[0]), len(alphabet))`.

    Examples
    --------
    >>> one_hot(["TGCA"], alphabet="ACGT")
    array([[[0., 0., 0., 1.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.]]])
    """
    A = len(alphabet)
    alphabet += 'N' # to capture non-sense characters

    sequences = np.asanyarray(sequences)
    assert sequences.ndim == 1, "array of sequences must be one-dimensional."

    n_sequences = sequences.shape[0]
    seq_len = len(sequences[0])

    # Unpack strings into 2D array, where each point has one character.
    s = np.zeros((n_sequences, seq_len), dtype="U1")
    for i in range(n_sequences):
        s[i] = list(sequences[i])

    # Make an integer array from the string array.
    pre_onehot = np.zeros(s.shape, dtype=np.uint8)
    for i, letter in enumerate(alphabet):
        # do nothing on 0 because array is initialized with zeros.
        if i:
            pre_onehot[s == letter] = i

    # create one-hot representation
    n_classes = len(alphabet)
    one_hot = np.eye(n_classes, dtype=np.int8)[pre_onehot]

    # remove nonsense character
    one_hot = one_hot[:,:,:A]

    # replace positions with N with 0.25 if true
    if uncertain_N:
        for n,x in enumerate(one_hot):
            index = np.where(np.sum(x, axis=-1) == 0)[0]
            one_hot[n,index,:] = 0.25
    return one_hot


def save_onehot_in_h5(filepath, X_train, Y_train, X_val, Y_val, X_test, Y_test, comp_level=4):
    """
    Save one-hot encoded arrays and labels to an HDF5 file with gzip.

    Parameters
    ----------
    filepath : str
        Path to the output HDF5 file.
    X_train : numpy.ndarray
        One-hot encoded training data.
    Y_train : numpy.ndarray
        Training labels.
    X_val : numpy.ndarray
        One-hot encoded validation data.
    Y_val : numpy.ndarray
        Validation labels.
    X_test : numpy.ndarray
        One-hot encoded test data.
    Y_test : numpy.ndarray
        Test labels.
    comp_level : int
        gzip compression level. An integer from 0 to 9, default is 4.
    """
    print(f'Saving data in h5 (gzip compression level = {comp_level})')
    with h5py.File(filepath, 'w') as h5f:
        h5f.create_dataset('x_train', data=X_train, dtype='int8',
                           compression='gzip', compression_opts=comp_level)
        h5f.create_dataset('y_train', data=Y_train, dtype='float32',
                           compression='gzip', compression_opts=comp_level)
        h5f.create_dataset('x_valid', data=X_val, dtype='int8',
                           compression='gzip', compression_opts=comp_level)
        h5f.create_dataset('y_valid', data=Y_val, dtype='float32',
                           compression='gzip', compression_opts=comp_level)
        # h5f.create_dataset('X_test', data=X_test, dtype='int8',
        #                    compression='gzip', compression_opts=comp_level)
        # h5f.create_dataset('Y_test', data=Y_test, dtype='float32',
        #                    compression='gzip', compression_opts=comp_level)


def load_onehot_from_h5(filepath):
    """
    Load one-hot encoded arrays and labels from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the input HDF5 file.

    Returns
    -------
    X_train : numpy.ndarray
        One-hot encoded training data.
    Y_train : numpy.ndarray
        Training labels.
    X_val : numpy.ndarray
        One-hot encoded validation data.
    Y_val : numpy.ndarray
        Validation labels.
    X_test : numpy.ndarray
        One-hot encoded test data.
    Y_test : numpy.ndarray
        Test labels.
    """
    print(f'Loading onehot encoded data from "{filepath}"')
    with h5py.File(filepath, 'r') as h5f:
        X_train = h5f['x_train'][:]
        Y_train = h5f['y_train'][:]
        X_val = h5f['x_valid'][:]
        Y_val = h5f['y_valid'][:]
        # X_test = h5f['X_test'][:]
        # Y_test = h5f['Y_test'][:]
    return X_train, Y_train, X_val, Y_val


def makedir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.mkdir(dirpath)
        except OSError:
            sys.exit(f'Error: Failed to make a directory "{dirpath}"')
    else:
        pass


def parse_conf(conffile):
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(conffile, 'UTF-8')
    param_dict = dict(conf.items('parameters'))
    for k, v in param_dict.items():
        if v.lower() == 'none':
            param_dict[k] = None
    return param_dict
